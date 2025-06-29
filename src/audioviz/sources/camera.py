import cv2
from threading import Thread, Lock
from typing import Optional

import numpy as np
import cupy as cp

class CameraSource:
    """
    Simple live camera/video source.

    Attributes:
        cap (cv2.VideoCapture): OpenCV capture object.
        latest_frame (Optional[np.ndarray]): Last grabbed frame.
        running (bool): Whether the camera loop is running.
    """

    def __init__(self, camera_index: int = 0, width: Optional[int] = None, height: Optional[int] = None, video_path: Optional[str] = None):
        """
        Parameters:
            camera_index (int): Index of webcam (default 0).
            width (Optional[int]): Target width if resizing.
            height (Optional[int]): Target height if resizing.
            video_path (Optional[str]): Path to video file instead of live camera.
        """
        self.cap = cv2.VideoCapture(video_path if video_path else camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera or video at index/path: {camera_index if not video_path else video_path}")

        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        self._lock = Lock()
        self._thread: Optional[Thread] = None

    def start(self):
        """Start background thread to continuously grab frames."""
        self.running = True
        self._thread = Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        """Background loop to update latest frame."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self._lock:
                self.latest_frame = frame.copy()

    def read(self) -> Optional[np.ndarray]:
        """Get latest frame safely (thread-safe copy)."""
        with self._lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        """Stop camera and release resources."""
        self.running = False
        if self._thread is not None:
            self._thread.join()
        self.cap.release()

    def __del__(self):
        self.stop()
