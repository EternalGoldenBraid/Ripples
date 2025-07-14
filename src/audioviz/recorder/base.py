from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

# ───────────────────────────── project‑internal imports ─────────────────────

# ===========================================================================
#  Recorder plug‑in API
# ===========================================================================

class RecorderBase(ABC):
    """Minimal interface every recorder must implement."""

    @abstractmethod
    def start(self, fps: int) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    # Called from GUI thread (video) and audio callback respectively
    def feed_video(self, rgb_frame: np.ndarray) -> None: ...  # noqa: D401,E704
    def feed_audio(self, pcm_block: np.ndarray) -> None: ...  # noqa: D401,E704
