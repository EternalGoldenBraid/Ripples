from typing import Tuple, Union, Optional, List, Any, Iterator, Dict
from pathlib import Path
from itertools import cycle


import numpy as np
import cupy as cp
import cv2
from PyQt5.QtCore import Qt
from loguru import logger as log


from .base import ExcitationSourceBase
from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.utils.utils import resize_array
from audioviz.types import ArrayType

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
                 amplitude: float = 0.0,
                 position   : Tuple[float, float] = (0.5, 0.5),
                 name       : str = "heart",
                 audio_processor: Optional[AudioProcessor] = None,
                 ):

        super().__init__(name=name, nominal_peak=1)
        self.fps: Optional[float] = None
        self.frames: List[Union[np.ndarray, cp.ndarray]] = []
        self.frame: Optional[ArrayType] = None

        self.xp = backend
        self.output_resolution = resolution
        self.position   = position              # normalised (x, y)
        self.audio_processor = audio_processor
        self.follow_audio: bool = True if audio_processor is not None else False

        self.load_video(source) 
        self.video_resolution: Tuple[int, int] = (self.frames[0].shape[0], self.frames[0].shape[1])  # (Hf, Wf)
        self.out: Union[np.ndarray, cp.ndarray] = self.xp.zeros(self.output_resolution, dtype=self.xp.float32)
        self.num_frames = len(self.frames)
        assert self.fps is not None, "Video FPS must be set!"
        assert self.num_frames > 0, "No frames loaded from video source!"
        # self.manual_freq: float = self.fps/self.num_frames
        self.manual_freq: float = 1
        log.info(
            f"""
            Heart video loaded: {self.num_frames} frames at {self.fps} FPS, resolution {self.video_resolution}, frequency {self.manual_freq} Hz.
            """
        )

        self.amplitude: float = amplitude
        self._prev_mask: Optional[Union[np.ndarray, cp.ndarray]] = None

    def load_video(self, source: Union[str, Path, np.ndarray]):

        if isinstance(source, (str, Path)):
            p = Path(source)

            if not p.exists():
                raise FileNotFoundError(f"Video file {p} does not exist!")

            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise IOError(f"Cannot open video {p}")
        
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30   # fallback if metadata is missing
            frames_np: List[np.ndarray] = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                gray = self.xp.asarray(gray, dtype=self.xp.float32)
                gray -= gray.mean()           # centre to ~0
                gray /= (gray.std() + 1e-6)    # optional: unit variance
                frames_np.append(gray)
            if not frames_np:
                raise ValueError("No frames decoded from video!")

            self.frames: ArrayType = self.xp.asarray(np.stack(frames_np), dtype=self.xp.float32)
            self._frame_buffer: ArrayType = self.xp.zeros_like(self.frames[0])  # first frame as a template
            self._mask_buffer: ArrayType = self.xp.zeros_like(self.frames[0])  # first frame as a template
        else:                                   # numpy clip already supplied
            self.frames = self.xp.asarray(source, dtype=self.xp.float32) / source.max()

        self.frame_iter: Iterator[Union[np.ndarray, cp.ndarray]] = cycle(self.frames)
        
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
                "min": 1, "max": 100, "init": int(self.manual_freq),
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
                "min": 0, "max": 50000, "init": int(self.amplitude * 100),
                "tooltip": "Controls amplitude of synthetic ripple",
                "on_change": lambda val: self.set_amplitude(val / 100.0)
            }),
            ("gain", {
                "label": "Gain",
                "min": 0,
                "max": 1000,
                "init": int(self.gain* 100),
                "tooltip": "Strength of audio excitation",
                "on_change": lambda val: self._set_gain(val / 100.0)
            }),
        ]

    # ---------------------------------------------------------------- setters
    def set_amplitude(self, val): self.amplitude = val
    def _set_pos_x(self, x: float): self.position = (x, self.position[1])
    def _set_pos_y(self, y: float): self.position = (self.position[0], y)


    def extract_mask(self, frame, mode: str = "raw", threshold: float = 0.5) -> ArrayType:
        """
        Returns a static binary mask from the current frame.
        
        Parameters:
        -----------
        mode : str
            "raw"     → resized + centered intensity mask
            "binary"  → thresholded mask
            "contour" → binary mask of closed contour(s) using OpenCV
        
        threshold : float
            Applied if mode is "binary" or "contour"
        
        Returns:
        --------
        mask : ArrayType (float32 or bool)
            Mask in output field shape (H, W)
        """
        mask = self.xp.asarray(frame, dtype=self.xp.float32)
    
        H_out, W_out = self.output_resolution
        target_w = int(W_out * 0.25)
        target_h = int(mask.shape[0] * target_w / mask.shape[1])
        mask_resized = resize_array(mask, target_h, target_w, xp=self.xp)
    
        cx = int(self.position[1] * (W_out - 1))
        cy = int(self.position[0] * (H_out - 1))
        h, w = mask_resized.shape
        top    = max(0, cy - h // 2)
        left   = max(0, cx - w // 2)
        bottom = min(H_out, top  + h)
        right  = min(W_out, left + w)
    
        out = self.xp.zeros((H_out, W_out), dtype=self.xp.float32)
        mask_crop = mask_resized[:bottom-top, :right-left]
        out[top:bottom, left:right] = mask_crop - self.xp.mean(mask_crop)
    
        if mode == "raw":
            return out
    
        bin_mask = (out > threshold).astype(self.xp.uint8)
    
        if mode == "binary":
            return bin_mask
    
        elif mode == "contour":
            if self.xp != np:
                bin_mask = cp.asnumpy(bin_mask)
    
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(bin_mask, dtype=np.uint8)
            cv2.drawContours(filled, contours, -1, 1, thickness=-1)  # fill all contours
    
            return filled.astype(bool)
    
        else:
            raise ValueError(f"Unknown mode '{mode}' for extract_mask")

    # ---------------------------------------------------------------- runtime
    def __call__(self, t: float, channel_idx=1) -> Dict[str, Any]:
        # freq = (self.audio_processor.current_top_k_frequencies[channel_idx][0] \
        #         if self.follow_audio else self.manual_freq)
        #
        # phase = (t * freq) % 1.0                 # in [0,1)
        # frame_idx = int(phase * self.num_frames) # 0..num_frames-1
        # 
        # frame = self.frames[frame_idx]           # NumPy or CuPy slice

        self.out[:] = 0.0  # reset output field
        self._frame_buffer[:] = next(self.frame_iter)

        mask  = self.xp.asarray(self._frame_buffer[:], dtype=self.xp.float32)
        H_out, W_out = self.output_resolution
        
        # resize mask to ¼ of the field width
        target_w = int(W_out * 0.25)
        target_h = int(mask.shape[0] * target_w / mask.shape[1])
        mask_resized = resize_array(mask, target_h, target_w, xp=self.xp)
        
        # placement slice (again use field size)
        cx = int(self.position[1] * (W_out - 1))
        cy = int(self.position[0] * (H_out - 1))
        h, w = mask_resized.shape
        top    = max(0, cy - h // 2)
        left   = max(0, cx - w // 2)
        bottom = min(H_out, top  + h)
        right  = min(W_out, left + w)
        mask_crop = mask_resized[:bottom-top, :right-left]
        mask_crop = mask_crop - self.xp.mean(mask_crop)

        self.out[top:bottom, left:right] = self.amplitude * mask_crop
        overlay_mask = self.out > 0.0

        # Extract contour mask
        # TODO Can we not use gpu for this?
        if self.xp != np:
            bin_mask = cp.asnumpy(overlay_mask).astype(np.uint8)
        else:
            bin_mask = overlay_mask.astype(np.uint8)


        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(bin_mask, dtype=np.uint8)
        # cv2.drawContours(filled, contours, -1, 1, thickness=-1)  # fill all contours
        cv2.drawContours(filled, contours, -1, 1, thickness=1)
    
        edge_mask = filled.astype(bool)
        edge_overlay = filled.astype(np.float32)
        edge_overlay = cv2.GaussianBlur(edge_overlay, (3, 3), sigmaX=1)
        edge_overlay -= edge_overlay.mean()
        edge_overlay /= (np.abs(edge_overlay).max() + 1e-6)  # now in roughly [-1, 1]


        if not hasattr(self, 'log_counter_'):
            self.log_counter_ = 0
        self.log_counter_ += 1
        if self.log_counter_  == 10:
            log.debug(f"min: {self.out.min()}, max: {self.out.max()}, mean: {self.out.mean()}")
            self.log_counter_ = 0

        self.out[:] = .0

        return {
            'excitation': self.out,
                'overlay': {'mask': edge_overlay},
                'boundary': {'region': edge_mask},
        }

    def get_boundary_mask(self) -> ArrayType:
        shape = self.output_resolution
        self.mask = np.zeros(shape, dtype=bool)
        # Wall max a circle of ones in the center of the grid
        radius = min(shape) // 4  # Example radius
        center = (shape[0] // 2, shape[1] // 2)
        # Set to ones on the perimeter of the circle not on the inside
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                    self.mask[i, j] = True

        return self.mask
