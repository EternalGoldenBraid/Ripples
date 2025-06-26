import time

from PyQt5 import QtWidgets
import pyqtgraph as pg
# pg.setConfigOptions(useOpenGL=True) 
# pg.setConfigOptions(antialias=True)
import numpy as np
import librosa as lr
from matplotlib import cm
from matplotlib.colors import Normalize
from loguru import logger

from .visualizer_base import VisualizerBase
from audioviz.audio_processing.audio_processor import AudioProcessor

class SpectrogramVisualizer(VisualizerBase):
    def __init__(self,
                 processor: AudioProcessor,
                 cmap: cm.colors.Colormap,
                 norm: Normalize,
                 waveform_plot_duration: float,
                 parent: QtWidgets.QWidget = None):

        super().__init__(parent=parent)

        self.processor: AudioProcessor = processor

        self.cmap: cm.colors.Colormap = cmap
        self.norm: Normalize = norm
        self.waveform_plot_duration: float = waveform_plot_duration

        self.spectrogram_plot_item: pg.PlotItem
        self.spectrogram_view: pg.ImageView
        self.spectrogram_y_labels = processor.freq_bins
        self.waveform_plot: pg.PlotWidget
        self.waveform_curve: pg.PlotDataItem
        
        self.waveform_y_range: tuple = (-1, 1)

        if not hasattr(processor, 'n_spec_bins'):
            raise AttributeError(
                f"{self.__class__.__name__}: " \
                "requires the processor to have 'n_spec_bins' attribute.")
        self.spectrogram_y_range: tuple = (0, processor.n_spec_bins)

        self.init_ui()

    def init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
    
        # Spectrogram plot using ImageItem
        self.spectrogram_plot_item = pg.PlotItem(title="Spectrogram")
        self.spectrogram_plot_item.getViewBox().invertY(False)
    
        # Set axis labels
        self.spectrogram_plot_item.setLabel('bottom', "Time (frames)")
        self.spectrogram_plot_item.setLabel('left', "Hertz")
        self.spectrogram_plot_item.setYRange(*self.spectrogram_y_range)
        # Set freq labels
        self.spectrogram_plot_item.setYRange(0, self.processor.n_spec_bins)
    
        self.spectrogram_image_item = pg.ImageItem()
        self.spectrogram_plot_item.addItem(self.spectrogram_image_item)
    
        # Pack into GraphicsLayoutWidget
        spectrogram_widget = pg.GraphicsLayoutWidget()
        spectrogram_widget.addItem(self.spectrogram_plot_item)
    
        layout.addWidget(spectrogram_widget)
    
        # Waveform plot
        self.waveform_plot = pg.PlotWidget(title="Waveform")
        self.waveform_plot.setLabel("bottom", "Samples")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.setYRange(-1, 1)
        self.waveform_curve = self.waveform_plot.plot(pen="y")
        layout.addWidget(self.waveform_plot)

    def update_visualization(self) -> None:
        """
        Currently averaging spectrograms across channels.
        """
        snapshot = self.processor.get_latest_snapshot()
        if snapshot is None:
            return  # No new data yet
        
        audio, spectrograms = snapshot
        
        spectrogram = np.mean(spectrograms, axis=0)
        time_start_spectrogram = time.time()
        # Process spectrogram
        spectrogram_db = lr.power_to_db(spectrogram, ref=1.0)
        rgba_img = self.cmap(self.norm(spectrogram_db.T))
        rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)
        
        self.spectrogram_image_item.setImage(
            rgb_img,
            autoLevels=False,
            autoRange=False,
            levels=(0, 255)
        )
        time_end_spectrogram = time.time()
        # print(f"Spectrogram processing time: {time_end_spectrogram - time_start_spectrogram:.4f} seconds")
        
        time_start_waveform = time.time()
        # Process waveform
        mean_waveform = np.mean(audio, axis=1)
        
        n_samples = int(self.waveform_plot_duration * self.processor.sr)
        if len(mean_waveform) > n_samples:
            mean_waveform = mean_waveform[-n_samples:]
        
        self.waveform_curve.setData(mean_waveform)
        time_end_waveform = time.time()
        # print(f"Waveform processing time: {time_end_waveform - time_start_waveform:.4f} seconds")
