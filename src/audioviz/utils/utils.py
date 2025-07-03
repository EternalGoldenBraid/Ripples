from typing import Union, Optional
from functools import wraps
from contextlib import contextmanager
import time

from loguru import logger as log
import numpy as np
import cv2

def resize_array(img, target_h: int, target_w: int, xp):
    """
    Resize 2-D array `img` to (target_h, target_w) preserving dtype.

    Parameters
    ----------
    img : np.ndarray | cp.ndarray
        Single-channel image on CPU or GPU.
    target_h, target_w : int
        Desired output size.
    xp : module
        `numpy`  or `cupy`; use `backend` from your visualizer/source.

    Returns
    -------
    resized : np.ndarray | cp.ndarray
        Image of shape (target_h, target_w) on the same backend.
    """
    if xp is np:                                 # ---- CPU path
        return cv2.resize(img, (target_w, target_h),
                          interpolation=cv2.INTER_AREA)

    # -------- GPU path (CuPy) ------------------
    import cupyx.scipy.ndimage as cnd            # lazy-import cupyx

    zoom_y = target_h / img.shape[0]
    zoom_x = target_w / img.shape[1]
    # order=1 → bilinear, matches cv2.INTER_AREA quality for downsizing
    return cnd.zoom(img, zoom=(zoom_y, zoom_x), order=1)

@contextmanager
def timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info(f"⏱️ {label} took {elapsed:,.2f} s")
