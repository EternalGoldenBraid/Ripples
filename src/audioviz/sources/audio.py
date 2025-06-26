from typing import List, Tuple, Union

import numpy as np
import cupy as cp
from loguru import logger as log

from .base import ExcitationSourceBase
from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.utils.signal_processing import map_audio_freq_to_visual_freq

class AudioExcitation(ExcitationSourceBase):
    """
    Generates ripple excitations based on top-K dominant audio frequencies.

    Parameters:
        processor (AudioProcessor): Source of spectral audio data.
        position (Tuple[float, float]): Normalized (0-1) position of the emitter in the plane.
        max_frequency (float): Clipping frequency to avoid aliasing.
        amplitude (float): Default amplitude for fallback (may be overridden by energy weights).
        speed (float): Propagation speed in m/s.
        resolution (Tuple[int, int]): Resolution of the field in (H, W).
        decay_alpha (float): Controls spatial falloff.
        backend (np or cp): Backend array module.
        name (str): Display name.
        audio_processor_chl_idxs (List[int] | int): Channels to read from.

    Returns:
        2D wave excitation field.

    TODO:
        - Support for directional excitation from stereo/multichannel input.
        - Unify amplitude source: drop static amplitude in favor of energy-weighted top-K.
        - Normalize coordinates to world units (meters).

    Caveats:
        - Operates in pixel space.
        - Only energy-weighted top-K frequency amplitudes are supported for now.
    """

    def __init__(self, processor: AudioProcessor, 
                 position: Tuple[float, float],
                 max_frequency: float, amplitude: float,
                 speed:float, resolution: Tuple[int, int],
                 backend: Union[np.ndarray, cp.ndarray],
                 decay_alpha: float = 0.0,
                 name: str = "Audio Excitation",
                 audio_processor_chl_idxs: Union[List[int], int] = -1 # All channels
                ):
        """
        TODO:
            - Amplitude: It's updated in the QTSlider owned by WaveVisualizer but defined here?
        """
        self.processor = processor
        self.audio_processor_chl_idxs = audio_processor_chl_idxs
        self.xp = backend

        if position [0] < 0 or position[0] > 1.0 or \
            position[1] < 0 or position[1] > 1.0:
            err = f"Position {position} must be in range [0, 1] for both x and y coordinates."
            log.error(err)
            raise ValueError(err)

        position = (
            int(position[0] * resolution[1]),
            int(position[1] * resolution[0])
        )

        self.position = position
        self.amplitude = amplitude
        self.speed = speed
        self.resolution = resolution
        self.decay_alpha = decay_alpha
        self.max_frequency: float = max_frequency

        self.dx = 1.0 / resolution[0]
        self.dy = 1.0 / resolution[1]

        self.xs, self.ys = self.xp.meshgrid(
            self.xp.arange(resolution[1]),  # width
            self.xp.arange(resolution[0])   # height
        )

        self.name: str = name

    def __call__(self, t: float, chl_idx: int = 0) -> Union[np.ndarray, cp.ndarray]:
        """
        Note:
            I think chl_idx == 0 is the microphone with scarlet studio.
        """

        top_k_freqs = self.processor.current_top_k_frequencies[chl_idx] # from N, k -> k
        assert len(top_k_freqs) == self.processor.num_top_frequencies
        top_k_amps = self.processor.current_top_k_energies[chl_idx]
        freqs = self.xp.array(
            [f for f in top_k_freqs if f is not None and self.xp.isfinite(f)],
            dtype=self.xp.float32)
        # freqs = self.xp.array(freqs, dtype=self.xp.float32)
        amps = self.xp.array(top_k_amps, dtype=self.xp.float32)

        # Ensure 2D shape: (N, k)
        if freqs.ndim == 1:
            freqs = freqs[None, :]  # expand to (1, k)
            amps = amps[None, :]  # expand to (1, k)
        elif freqs.ndim != 2:
            raise ValueError(f"Expected freqs to be 1D or 2D, got shape {freqs.shape}")

        # freqs = map_audio_freq_to_visual_freq(freqs, self.max_frequency)


        excitation = self.xp.zeros(self.resolution, dtype=self.xp.float32)

        if len(freqs) == 0:
            return excitation

        N, k = freqs.shape

        x0 = self.position[0]
        y0 = self.position[1]

        xs = self.xs[None, :, :]
        ys = self.ys[None, :, :]

        r_pixels = self.xp.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)
        r_meters = r_pixels * self.dx

        decay = self.xp.exp(-self.decay_alpha * r_meters)

        freqs = self.xp.clip(freqs, 1e-3, self.max_frequency)
        wavelengths = self.speed / freqs 
        phases = 2 * self.xp.pi * freqs * t

        # for computing:
        #   ripple[n, k, h, w] = amplitude * decay[n, 1, h, w] * sin(phase[n, k, 1, 1] - 2π * r[n, 1, h, w] / wavelength[n, k, 1, 1])
        r = r_meters[:, None, :, :] # shape: (N, 1, H, W)
        decay = decay[:, None, :, :] # shape: (N, 1, H, W)
        wavelengths = wavelengths[:, :, None, None] # shape: (N, k, 1, 1)
        phases = phases[:, :, None, None] # shape: (N, k, 1, 1)
        amps = amps[:, :, None, None]   # shape: (N, k, 1, 1)

        propagation_limit = self.speed * t
        mask = r <= propagation_limit

        # ripple = self.amplitude * decay * self.xp.sin(phases - 2 * self.xp.pi * r / wavelengths)
        ripple =  decay * amps * self.xp.sin(phases - 2 * self.xp.pi * r / wavelengths)
        # ripple *= mask

        return ripple.sum(axis=(0, 1))

    def set_amplitude(self, value: float):
            self.amplitude = value

    def set_decay_alpha(self, value: float):
        self.decay_alpha = value

    def set_position_x(self, x: int):
        self.position = (x, self.position[1])

    def set_position_y(self, y: int):
        self.position = (self.position[0], y)

    def get_controls(self):
        return [
            ("amplitude", {
                "label": "Amplitude",
                "min": 0,
                "max": 500,
                "init": int(self.amplitude * 100),
                "tooltip": "Strength of audio excitation",
                "on_change": lambda val: self.set_amplitude(val / 100.0)
            }),
            ("decay_alpha", {
                "label": "Decay α",
                "min": 0,
                "max": 200,  # Maps to 0–20
                "init": int(self.decay_alpha * 10),
                "tooltip": "How fast the ripple decays from the source",
                "on_change": lambda val: self.set_decay_alpha(val / 10.0)
            }),
            ("position_x", {
                "label": "Source X",
                "min": 0,
                "max": self.resolution[1] - 1,
                "init": self.position[0],
                "tooltip": "X coordinate of the source",
                "on_change": self.set_position_x
            }),
            ("position_y", {
                "label": "Source Y",
                "min": 0,
                "max": self.resolution[0] - 1,
                "init": self.position[1],
                "tooltip": "Y coordinate of the source",
                "on_change": self.set_position_y
            })
        ]
