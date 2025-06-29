from typing import Tuple, Union, Optional, List, Any, Iterator
from pathlib import Path
import numpy as np
import cv2
from PyQt5.QtCore import Qt

from .base import ExcitationSourceBase
from audioviz.audio_processing.audio_processor import AudioProcessor

class ImagePulseExcitation(ExcitationSourceBase):
    """Excitation driven by a 2-D mask (image frame) that pulses over time.

    Parameters
    ----------
    source : Union[str, Path, np.ndarray]
        Path to an image / video file **or** an already-loaded numpy image.
    resolution : Tuple[int, int]
        (H, W) of the ripple field – used for resizing.
    position : Tuple[float, float]
        Normalised (x, y) ∈ [0, 1] where the mask centre will be placed.
    pulse_freq : float
        Sinusoidal pulse frequency in Hz.
    amplitude : float
        Peak excitation strength.
    backend : np / cupy
        Choose np or cp depending on CPU / GPU mode.
    name : str
        Source label shown in UI.
    """

    def __init__(self,
                 source : Union[str, Path, np.ndarray],
                 resolution : Tuple[int, int],
                 position   : Tuple[float, float] = (0.5, 0.5),
                 backend    = np,
                 name       : str = "Image Pulse",
                 audio_processor: Optional[AudioProcessor] = None,
                 ):

        super().__init__(name=name)
        self.xp = backend
        self.resolution = resolution
        self.position   = position              # normalised (x, y)
        self.out: np.ndarray = self.xp.zeros(resolution, dtype=self.xp.float32)
        self.audio_processor = audio_processor
        self.follow_audio: bool = True if audio_processor is not None else False
        self.manual_freq: float = 1.0  # Hz

        # --- Frame iterator --------------------------------------------------
        self._frame_iter : Iterator[np.ndarray]
        self._is_video   : bool

        self.load_video(source) 
        self.num_frames = len(self.frames)

    def load_video(self, source: Union[str, Path, np.ndarray]):

        if isinstance(source, (str, Path)):
            p = Path(source)
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise IOError(f"Cannot open video {p}")
        
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                frames.append(gray)
            if not frames:
                raise ValueError("No frames decoded from video!")
            self.frames = frames                                    # list of (Hf,Wf) arrays
        else:                                   # numpy clip already supplied
            self.frames = [source.astype(np.float32) / source.max()]
        
    def get_controls(self):
        H, W = self.resolution
        return [
            ("follow_audio", {                 # NEW checkbox
                "type": "checkbox",
                "label": "Follow Audio",
                "init": self.follow_audio,
                "tooltip": "If checked, heart speed syncs to audio.",
                "on_change": lambda v: setattr(self, "follow_audio", v == Qt.Checked)
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
            ("gain", {
                    "type": "slider",
                    "label": "Gain",
                    "min": 0, "max": 400, "init": int(self.gain*100),
                    "on_change": lambda v: self._set_gain(v/100)
                }),
        ]

    # ---------------------------------------------------------------- setters
    def set_pulse_freq(self, f: float): self.pulse_freq = f
    def _set_pos_x(self, x: float): self.position = (x, self.position[1])
    def _set_pos_y(self, y: float): self.position = (self.position[0], y)

    # ---------------------------------------------------------------- runtime
    def __call__(self, t: float, channel_idx=1) -> np.ndarray:
        freq = (self.audio_processor.current_top_k_frequencies[channel_idx][0] \
                if self.follow_audio else self.manual_freq)
        freq = (self.audio_processor.current_top_k_frequencies[channel_idx][0] \
                if self.follow_audio else self.manual_freq)

        phase = (t * freq) % 1.0                 # in [0,1)
        frame_idx = int(phase * self.num_frames) # 0..num_frames-1
        
        frame = self.frames[frame_idx]           # NumPy or CuPy slice

        mask  = self.xp.asarray(frame, dtype=self.xp.float32)

        # Resize mask to ~¼ of field diagonal & keep aspect
        H, W = self.resolution
        target_w = int(W * 0.25)
        target_h = int(mask.shape[0] * target_w / mask.shape[1])
        mask_resized = cv2.resize(mask, (target_w, target_h),
                                  interpolation=cv2.INTER_AREA)

        # Compute placement slice
        cx = int(self.position[0] * (W-1))
        cy = int(self.position[1] * (H-1))
        h, w = mask_resized.shape
        top  = max(0, cy - h//2)
        left = max(0, cx - w//2)
        bottom = min(H, top + h)
        right  = min(W, left + w)
        mask_crop = mask_resized[:bottom-top, :right-left]

        # Build excitation field
        self.out[:] = 0.0  # Clear output field
        self.out[top:bottom, left:right] = mask_crop
        return self.out
