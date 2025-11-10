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
    mode : str
        "contour_ecg_edge"  → contour + ECG edge
        "contour_ecg_center"→ contour + ECG center
        "contour_only"      → contour only, no ECG
        "contour_no_excitation" → contour, no ECG, no amplitude
        Default is "contour_ecg_edge".
    """

    def __init__(self,
                 video_path: Union[str, Path],
                 backend,
                 resolution : Tuple[int, int],
                 ecg_path: Optional[Union[str, Path]] = None,
                 amplitude: float = 0.0,
                 position   : Tuple[float, float] = (0.5, 0.5),
                 name       : str = "heart",
                 audio_processor: Optional[AudioProcessor] = None,
                 # mode: str = "contour_ecg_edge"
                 mode: str = "contour_ecg_center"
                 # mode: str = "contour_only"
                 # mode: str = "contour_no_excitation"
                 ):

        super().__init__(name=name, nominal_peak=1)
        self.fps: Optional[float] = None
        self.frames: List[Union[np.ndarray, cp.ndarray]] = []
        self.frame: Optional[ArrayType] = None

        # self.xp = backend
        self.xp = np # For now do all in CPU here due to opencv
        self.output_resolution = resolution
        self.position   = position              # normalised (x, y)
        self.audio_processor = audio_processor
        self.follow_audio: bool = True if audio_processor is not None else False
        self.mode: str = mode

        self.load_video(video_path) 
        self.video_resolution: Tuple[int, int] = (self.frames[0].shape[0], self.frames[0].shape[1])  # (Hf, Wf)
        self.out: ArrayType = self.xp.zeros(self.output_resolution, dtype=self.xp.float32)
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

        if ecg_path is not None:
            self.load_ecg(ecg_path)

        self.frame_idx: int = 0

        self.amplitude: float = amplitude

        self._edge_buffer: ArrayType = self.xp.zeros(self.output_resolution, dtype=self.xp.uint8)
        self._filled_buffer: ArrayType = self.xp.zeros(self.output_resolution, dtype=self.xp.uint8)

    def load_ecg(self, ecg_path: Union[str, Path]) -> None:
        self.ecg = np.load(ecg_path).astype(np.float32)
        if len(self.ecg) != self.num_frames:
            log.warning(
                f"""
                ECG length {len(self.ecg)} does not match number of video frames {self.num_frames}
                """
            )
            if len(self.ecg) < self.num_frames:
                self.ecg = np.pad(self.ecg, (0, self.num_frames - len(self.ecg)), mode='edge')
                log.warning( f"Padding ECG to match video frames: new length {len(self.ecg)}")
            elif len(self.ecg) > self.num_frames:
                self.ecg = self.ecg[:self.num_frames]
                log.warning( f"Truncating ECG to match video frames: new length {len(self.ecg)}")
        log.info(f"✅ ECG loaded with {len(self.ecg)} samples.")

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


    def extract_mask(self, frame, mode: str = "raw", threshold: float = 0.0) -> ArrayType:
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
            cv2.drawContours(filled, contours, -1, 1, thickness=1)
    
            return filled.astype(bool)
    
        else:
            raise ValueError(f"Unknown mode '{mode}' for extract_mask")

    # ---------------------------------------------------------------- runtime
    def __call___old(self, t: float, channel_idx=1) -> Dict[str, Any]:
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

    def __call__(self, t: float, channel_idx=1) -> Dict[str, Any]:
        self.out[:] = 0.0
        self.frame_idx += 1
        self.frame_idx %= self.num_frames

        self._frame_buffer[:] = next(self.frame_iter)
    
        if self.mode == "contour_only":
            return self._excite_from_contour_only()
    
        elif self.mode == "contour_ecg_edge":
            assert hasattr(self, "ecg"), "ECG data must be loaded for 'contour_ecg_edge' mode"
            return self._excite_ecg_edge()
    
        elif self.mode == "contour_ecg_center":
            assert hasattr(self, "ecg"), "ECG data must be loaded for 'contour_ecg_center' mode"
            return self._excite_ecg_center()
    
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

    def extract_masks(
        self,
        frame: np.ndarray,
        threshold: float = 0.0,
        edge_thickness: int = 3,) -> Dict[str, ArrayType]:
        """Return filled heart mask **and** edge mask for a single grayscale frame.

        1. Thresholds + recenters frame → `filled` boolean mask (H, W).
        2. Finds external contours → draws into reusable `_edge_buffer` with the requested thickness.

        Returns
        -------
        dict with keys:
            'filled' : bool ndarray (H, W)
            'edge'   : bool ndarray (H, W)
        """
        # ---------- step 1: filled mask (resize + place) ----------
        H_out, W_out = self.output_resolution
        mask = self.xp.asarray(frame, dtype=self.xp.float32)

        target_w = int(W_out * 0.125)
        target_h = int(mask.shape[0] * target_w / mask.shape[1])
        mask_resized = resize_array(mask, target_h, target_w, xp=self.xp)
        mask_resized = self.xp.asnumpy(mask_resized) if self.xp != np else mask_resized

        cx = int(self.position[1] * (W_out - 1))
        cy = int(self.position[0] * (H_out - 1))
        h, w = mask_resized.shape
        top    = max(0, cy - h // 2)
        left   = max(0, cx - w // 2)
        bottom = min(H_out, top  + h)
        right  = min(W_out, left + w)

        filled = self._filled_buffer
        filled[:] = 0
        filled[top:bottom, left:right] = (mask_resized[:bottom-top, :right-left] > threshold).astype(np.uint8)

        # ---------- step 2 : edge mask (largest, hole-free) ----------
        edge   = self._edge_buffer      # reusable uint8 buffers (0/1)
        filled = self._filled_buffer    #    “
        edge[:]   = 0
        # keep filled as it is – we only need its size
        
        # 1. find *all* contours, including internal rings
        contours, _ = cv2.findContours(
            filled,                    # binary image (uint8 0/1)
            cv2.RETR_EXTERNAL,         # only outer boundaries
            cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 2. pick the biggest one
            biggest = max(contours, key=cv2.contourArea)
        
            # 3. paint it *solid* into `filled` (wipes holes)
            filled[:] = 0
            cv2.drawContours(filled, [biggest], -1, 1, thickness=-1)   # -1  → fill
        
            # 4. draw only the edge into `edge`
            cv2.drawContours(edge,   [biggest], -1, 1, thickness=edge_thickness)
        
        # convert to bool for the caller
        edge_bool   = edge.astype(bool)
        filled_bool = filled.astype(bool)
        return {"filled": filled_bool, "edge": edge_bool}

 # ---------------------------------------------------------------- excitation modes
    def _excite_ecg_edge(self) -> Dict[str, Any]:
        masks = self.extract_masks(self._frame_buffer[:])
        edge_bool = masks["edge"]
        edge_f32 = edge_bool.astype(np.float32)

        # Smooth once; reuse buffer
        edge_f32 = cv2.GaussianBlur(edge_f32, (3, 3), sigmaX=1)
        edge_f32 -= edge_f32.mean()
        max_val = self.xp.abs(edge_f32).max()
        if max_val > 0:
            edge_f32 /= max_val

        amp = self.ecg[self.frame_idx]
        self.out[:] = self.amplitude * amp * edge_f32
        return {
            "excitation": self.out,
            "overlay": {"mask": edge_f32},
            "boundary": {"region": masks["filled"]},
        }

    def _excite_ecg_center(self) -> Dict[str, Any]:
        masks = self.extract_masks(self._frame_buffer[:])
        filled = masks["filled"]
        edge   = masks["edge"]

        ys, xs = np.nonzero(filled)
        if len(xs) == 0:
            return {
                "excitation": self.out,
                "overlay": {"mask": np.zeros_like(self.out)},
                "boundary": {"region": edge},
            }

        cx = int(xs.mean()); cy = int(ys.mean())
        sigma = np.max(np.sqrt((xs - cx)**2 + (ys - cy)**2)) / 3.0

        if not hasattr(self, "yy"):
            H, W = self.output_resolution
            self.yy, self.xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        dx = self.xx - cx; dy = self.yy - cy
        gaussian = np.exp(-(dx*dx + dy*dy) / (2 * sigma**2))
        gaussian *= filled

        amp = self.ecg[self.frame_idx]
        excitation = gaussian * amp * self.amplitude

        overlay = gaussian.astype(np.float32)
        overlay -= overlay.mean(); max_val = np.abs(overlay).max()
        if max_val > 0:
            overlay /= max_val

        self.out[:] = self.xp.asarray(excitation, dtype=self.xp.float32)
        return {
            "excitation": self.out,
            "overlay": {"mask": overlay},
            "boundary": {"region": edge},
        }

    def _excite_from_contour_only(self) -> Dict[str, Any]:
        """Excite *only* along the heart boundary (edge).

        The filled interior is ignored – we press a thin wall onto the water surface, 
        like a cookie‑cutter.  The edge mask is optionally blurred so the excitation 
        isn’t a single‑pixel spike.
        """
        masks = self.extract_masks(self._frame_buffer[:])
        edge_bool = masks["edge"]   # boolean mask (H, W)

        # Convert to float32 and smooth a little so the wave starts gently
        edge_f32 = edge_bool.astype(np.float32)
        edge_f32 = cv2.GaussianBlur(edge_f32, (3, 3), sigmaX=1)
        edge_f32 -= edge_f32.mean()
        max_val = np.abs(edge_f32).max()
        if max_val > 0:
            edge_f32 /= max_val

        # Write excitation directly to GPU/CPU buffer
        self.out[:] = self.amplitude * self.xp.asarray(edge_f32, dtype=self.xp.float32)

        return {
            "excitation": self.out,
            "overlay": {"mask": edge_f32},
            "boundary": {"region": edge_bool},
        }
