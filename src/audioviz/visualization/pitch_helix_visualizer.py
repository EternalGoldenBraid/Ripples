from typing import List

from PyQt5 import QtWidgets
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector

from audioviz.visualization.visualizer_base import VisualizerBase
from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.utils.guitar_profiles import GuitarProfile  

NOTE_NAMES: List[str] = ['C', 'C#', 'D', 'D#', 'E', 'F', 
              'F#', 'G', 'G#', 'A', 'A#', 'B']

class PitchHelixVisualizer(VisualizerBase):
    def __init__(self,
                 processor: AudioProcessor,
                 guitar_profile: GuitarProfile,
                 parent: QtWidgets.QWidget = None):

        super().__init__(processor, parent=parent)

        # Frequency mapping settings
        self.min_freq: float = guitar_profile.lowest_frequency()
        self.max_freq: float = guitar_profile.highest_frequency()

        # Helix settings
        self.radius: float = 10.0
        self.pitch: float = 3  # Rise per full turn (2*pi)
        octaves_span = np.log2(self.max_freq / self.min_freq)
        self.turns: int = octaves_span


        # Init OpenGL view
        self.view: gl.GLViewWidget = gl.GLViewWidget()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.view)

        self.view.opts['distance'] = 40
        self.view.orbit(45, 60)  # Nice initial angle

        # Create the static helix backbone
        self.create_helix()

        # Create scatter plot for dynamic active notes
        self.scatter: gl.GLScatterPlotItem = gl.GLScatterPlotItem()
        self.add_note_labels()
        self.create_semitone_nodes()
        self.view.addItem(self.scatter)


    def create_semitone_nodes(self) -> None:
        """Create hollow target nodes at every playable semitone."""
        self.semitone_positions = []
        self.semitone_scatter = gl.GLScatterPlotItem()
        self.view.addItem(self.semitone_scatter)
    
        # Calculate number of semitone steps
        lowest = self.min_freq
        highest = self.max_freq
    
        num_semitones = int(np.round(12 * np.log2(highest / lowest)))
    
        freqs = [lowest * (2 ** (i / 12)) for i in range(num_semitones + 1)]
    
        positions = []
        colors = []
    
        for freq in freqs:
            xyz = self.frequency_to_xyz(freq)
            positions.append(xyz)
    
            # Hollow (dim) default color
            colors.append((255, 255, 0, 0.2))  # Yellow with alpha
    
        self.semitone_positions = np.array(positions)
        self.semitone_colors = np.array(colors)
    
        self.semitone_scatter.setData(
            pos=self.semitone_positions,
            color=self.semitone_colors,
            size=18.0
        )


    def add_note_labels(self) -> None:
        """Add floating labels for pitch classes around the spiral."""
        from pyqtgraph.opengl.items.GLTextItem import GLTextItem
    
        for i, name in enumerate(NOTE_NAMES):
            # Calculate theta for this note
            theta = 2 * np.pi * (i / 12)  # 12 semitones per circle
    
            # Radius slightly larger than spiral
            label_radius = self.radius * 1.1
    
            # Place label at base layer (z=0)
            x = label_radius * np.cos(theta)
            y = label_radius * np.sin(theta)
            z = 0  # Could offset later for multiple octaves
    
            text_item = GLTextItem(pos=(x, y, z), text=name, color=(255,255,255,255))
            self.view.addItem(text_item)

    def create_helix(self) -> None:
        """Create the static background helix."""
        theta: float = np.linspace(0, 2 * np.pi * self.turns, 1000)
        x: float  = self.radius * np.cos(theta)
        y: float  = self.radius * np.sin(theta)
        z: float  = self.pitch * theta

        pts: np.ndarray = np.vstack([x, y, z]).T

        line = gl.GLLinePlotItem(pos=pts, color=(0.5, 0.5, 0.5, 1.0), width=1.0, antialias=True)
        self.view.addItem(line)

    def frequency_to_xyz(self, freq: float) -> np.ndarray:
    
        """Map a frequency (Hz) to (x, y, z) on the helix based on lowest guitar frequency."""
        # Calculate semitone distance from the guitar's lowest frequency
        semitones_from_min = 12 * np.log2(freq / self.min_freq)
    
        # Calculate turn on spiral
        # One full circle = 12 semitones (octave)
        theta = 2 * np.pi * (semitones_from_min / 12)
    
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        z = self.pitch * theta
    
        return np.array([x, y, z])

    def update_visualization(self) -> None:
        dominant_freq = self.processor.current_top_k_frequencies[0]
        if dominant_freq is None:
            print("No dominant frequency detected.")
            return
        else:
            print(f"Dominant frequency: {dominant_freq:.2f} Hz")
    
        points = []
        colors = []
        if self.min_freq <= dominant_freq <= self.max_freq:
            xyz = self.frequency_to_xyz(dominant_freq)
            points.append(xyz)
            colors.append((1.0, 0.2, 0.2, 1.0))  # bright red
    
        if points:
            pts_arr = np.array(points)
            colors_arr = np.array(colors)
            self.scatter.setData(pos=pts_arr, color=colors_arr, size=10.0)
        else:
            self.scatter.setData(pos=np.zeros((0, 3)))
