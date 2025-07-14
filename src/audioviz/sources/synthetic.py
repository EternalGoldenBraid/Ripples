from typing import Optional, Union, Tuple, Dict, List, Callable, Union

from audioviz.sources.base import ExcitationSourceBase
import numpy as np
import cupy as cp
from loguru import logger as log

CONTROLS_TYPE = List[Dict[str, Dict[str, Union[str, int, float, Callable]]]]

class SyntheticPointExcitation(ExcitationSourceBase):
    """
    Emits a single frequency ripple from a fixed point, useful for debugging and testing.

    Parameters:
        dx float: Grid spacing in meters.
        resolution (Tuple[int, int]): Resolution of the ripple field (H, W).
        position (Tuple[int, int]): Pixel coordinates of emission source.
        amplitude (float): Amplitude of the synthetic sine wave.
        frequency (float): Frequency in Hz.
        decay_alpha (float): Spatial falloff.
        speed (float): Propagation speed in m/s.
        backend (np or cp): Numpy-compatible backend.
        name (str): Display name.

    Returns:
        2D wave excitation array at time `t`.

    TODO:
        - Extend to support geometric primitives (circles, triangles).
        - Generalize position to world-space.

    Caveats:
        - Emits only a single tone.
        - No propagation mask or boundary behavior implemented.
    """

    def __init__(self, dx: float, resolution: Tuple[int, int], position, amplitude:float=0.0,
                 frequency:float=1.0, decay_alpha:float=0.0, speed:float=340.0,
                 backend=np, name: str = "Synthetic Point Excitation", nominal_peak: float=10.0):
        super().__init__(name=name, nominal_peak=nominal_peak)

        self.dx = dx
        self.amplitude = amplitude
        self.frequency = frequency
        self.decay_alpha = decay_alpha
        self.speed = speed
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

        self.xs, self.ys = self.xp.meshgrid(
            self.xp.arange(resolution[1]),
            self.xp.arange(resolution[0])
        )

    def __call__(self, t: float) -> Dict[str, Union[np.ndarray, cp.ndarray]]:
        x0, y0 = self.position
        r_pix = self.xp.sqrt((self.xs - x0)**2 + (self.ys - y0)**2)
        r_m = r_pix * self.dx
        decay = self.xp.exp(-self.decay_alpha * r_m)

        wavelength = self.speed / self.frequency
        phase = 2 * self.xp.pi * self.frequency * t
        ripple = self.gain * self.amplitude * decay *self.xp.sin(phase - 2 * self.xp.pi * r_m / wavelength)

        propagation_limit = self.speed * t
        mask = r_m <= propagation_limit
        ripple *= mask

        # if not hasattr(self, 'log_counter_'):
        #     self.log_counter_ = 0
        # self.log_counter_ += 1
        # if self.log_counter_  == 10:
        #     log.debug(f"min: {ripple.min()}, max: {ripple.max()}, mean: {ripple.mean()}")
        #     self.log_counter_ = 0

        return {'excitation': ripple}

    def _set_amplitude(self, val): 
        self.amplitude = val if val < self.nominal_peak else self.nominal_peak
        # log.debug( f"Set Amplitude to {self.amplitude:.3f}")

    def _set_frequency(self, val): 
        self.frequency = val
        # log.debug( f"Set Frequency to {self.frequency:.3f} Hz ({val} Hz)")

    def _set_decay_alpha(self, val): 
        self.decay_alpha = val
        # log.debug( f"Set Decay α to {self.decay_alpha:.3f} ({val})")

    def get_controls(self):
        return [
            ("amplitude", {
                "label": "Amplitude",
                "min": 0, "max": int(self.nominal_peak*100), "init": int(self.amplitude * 100),
                "tooltip": "Controls amplitude of synthetic ripple",
                "on_change": lambda val: self._set_amplitude(val / 100.0)
            }),
            ("frequency", {
                "label": "Frequency (Hz)",
                "min": 1, "max": 1000, "init": int(self.frequency),
                "tooltip": "Controls frequency of synthetic ripple",
                "on_change": lambda val: self._set_frequency(val)
            }),
            ("decay_alpha", {
                "label": "Decay α",
                "min": 0, "max": 200, "init": int(self.decay_alpha * 10),
                "tooltip": "Controls spatial falloff of the excitation",
                "on_change": lambda val: self._set_decay_alpha(val / 10.0)
            }),
            ("gain", {
                    "type": "slider",
                    "label": "Gain",
                    "min": 0, "max": 400, "init": int(self.gain*100),
                    "on_change": lambda v: self._set_gain(v/100)
                }),
        ]
