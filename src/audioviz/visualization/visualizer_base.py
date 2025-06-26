from typing import Optional, Dict

from abc import abstractmethod
from PyQt5 import QtCore, QtWidgets
from audioviz.audio_processing.audio_processor import AudioProcessor

from loguru import logger

class VisualizerBase(QtWidgets.QWidget):
    """
    Abstract base class for visualizers that update in response to incoming data.

    Parameters:
        update_interval_ms (int): Interval between updates in milliseconds.
        parent (QtWidgets.QWidget): Optional parent widget.

    Attributes:
        timer (QTimer): Handles periodic updates.

    Methods:
        update_visualization(): Abstract method for updating visuals.

    Notes:
        - Any subclass must implement `update_visualization()`.
        - Can be embedded in complex Qt GUI layouts.
    """


    def __init__(self, 
                 update_interval_ms: int = 50, parent=None,
                 **kwargs: Optional[Dict]):
        super().__init__(parent)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(update_interval_ms)
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start()

    @abstractmethod
    def update_visualization(self):
        """Update the visualization based on new processor data."""
        pass
