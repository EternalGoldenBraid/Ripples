from typing import Tuple, Union, Optional, List, Any, Iterator
from pathlib import Path
import numpy as np
import cupy as cp
import cv2
from PyQt5.QtCore import Qt

from .base import ExcitationSourceBase
from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.utils.utils import resize_array

class HeartVideoExcitation(ExcitationSourceBase):
    """Excitation driven by a 2-D mask (image frame) that pulses over time.

    Parameters
    ----------
    source : Union[str, Path, np.ndarray]
        Path to an image / video file **or** an already-loaded numpy image.
    resolution : Tuple[int, int]
        (H, W) of the ripple field – used for resizing.
    position : Tuple[float, float]
        Normalised (x, y) ∈ [0, 1] where the mask centre will be placed.
    amplitude : float
        Peak excitation strength.
    backend : np / cupy
        Choose np or cp depending on CPU / GPU mode.
    name : str
        Source label shown in UI.
    """

    def __init__(self,
                 source : Union[str, Path, np.ndarray],
                 backend,
                 resolution : Tuple[int, int],
                 position   : Tuple[float, float] = (0.5, 0.5),
                 name       : str = "Image Pulse",
                 audio_processor: Optional[AudioProcessor] = None,
                 ):

        super().__init__(name=name)

        self.xp = backend
        self.output_resolution = resolution
        self.position   = position              # normalised (x, y)
        self.audio_processor = audio_processor
        self.follow_audio: bool = True if audio_processor is not None else False
        self.manual_freq: float = 1.0  # Hz

        self.load_video(source) 
        self.video_resolution: Tuple[int, int] = (self.frames[0].shape[0], self.frames[0].shape[1])  # (Hf, Wf)
        self.out: Union[np.ndarray, cp.ndarray] = self.xp.zeros(self.output_resolution, dtype=self.xp.float32)
        self.num_frames = len(self.frames)

        self.amplitude: float = 1.0
        self._prev_mask: Optional[Union[np.ndarray, cp.ndarray]] = None

    def load_video(self, source: Union[str, Path, np.ndarray]):

        if isinstance(source, (str, Path)):
            p = Path(source)

            if not p.exists():
                raise FileNotFoundError(f"Video file {p} does not exist!")

            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise IOError(f"Cannot open video {p}")
        
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                gray = self.xp.asarray(gray, dtype=self.xp.float32)
                frames.append(gray)
            if not frames:
                raise ValueError("No frames decoded from video!")
            self.frames = frames                                    # list of (Hf,Wf) arrays
        else:                                   # numpy clip already supplied
            self.frames = [self.xp.asarray(source, dtype=self.xp.float32) / source.max()]
        
    def get_controls(self):
        H, W = self.output_resolution
        return [
            ("follow_audio", {                 # NEW checkbox
                "type": "checkbox",
                "label": "Follow Audio",
                "init": self.follow_audio,
                "tooltip": "If checked, heart speed syncs to audio.",
                "on_change": lambda v: setattr(self, "follow_audio", v == Qt.Checked if self.audio_processor else False)
            }),
            ("manual_freq", {                  # slider
                "type": "slider",
                "label": "Manual Frequency (Hz)",
                "min": 1, "max": 10, "init": int(self.manual_freq),
                "on_change": lambda v: setattr(self, "manual_freq", v)
            }),
            ("pos_x", {
                "label": "X pos",
                "min": 0, "max": W-1, "init": int(self.position[0] * (W-1)),
                "tooltip": "Horizontal placement",
                "on_change": lambda v: self._set_pos_x(v / (W-1))
            }),
            ("pos_y", {
                "label": "Y pos",
                "min": 0, "max": H-1, "init": int(self.position[1] * (H-1)),
                "tooltip": "Vertical placement",
                "on_change": lambda v: self._set_pos_y(v / (H-1))
            }),
            ("amplitude", {
                "label": "Amplitude",
                "min": 0, "max": 500, "init": int(self.amplitude * 100),
                "tooltip": "Controls amplitude of synthetic ripple",
                "on_change": lambda val: self.set_amplitude(val / 100.0)
            }),
        ]

    # ---------------------------------------------------------------- setters
    def set_amplitude(self, val): self.amplitude = val
    def _set_pos_x(self, x: float): self.position = (x, self.position[1])
    def _set_pos_y(self, y: float): self.position = (self.position[0], y)

    # ---------------------------------------------------------------- runtime
    def __call__(self, t: float, channel_idx=1) -> np.ndarray:
        freq = (self.audio_processor.current_top_k_frequencies[channel_idx][0] \
                if self.follow_audio else self.manual_freq)

        phase = (t * freq) % 1.0                 # in [0,1)
        frame_idx = int(phase * self.num_frames) # 0..num_frames-1
        
        frame = self.frames[frame_idx]           # NumPy or CuPy slice

        mask  = self.xp.asarray(frame, dtype=self.xp.float32)
        H_out, W_out = self.output_resolution
        
        # resize mask to ¼ of the field width
        target_w = int(W_out * 0.25)
        target_h = int(mask.shape[0] * target_w / mask.shape[1])
        mask_resized = resize_array(mask, target_h, target_w, xp=self.xp)
        
        # placement slice (again use field size)
        cx = int(self.position[0] * (W_out - 1))
        cy = int(self.position[1] * (H_out - 1))
        h, w = mask_resized.shape
        top    = max(0, cy - h // 2)
        left   = max(0, cx - w // 2)
        bottom = min(H_out, top  + h)
        right  = min(W_out, left + w)
        mask_crop = mask_resized[:bottom-top, :right-left]

        delta = mask_crop - self._prev_mask  if self._prev_mask is not None else mask_crop
        self._prev_mask = mask_crop.copy()
        self.out[:] = 0.0
        self.out[top:bottom,left:right] = delta * self.amplitude

        return self.out
