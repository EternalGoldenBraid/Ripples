from typing import Optional

import cupy as cp
import pyqtgraph.opengl as gl

import numpy as np
import matplotlib.cm as cm
from PyQt5 import QtWidgets
import pyqtgraph as pg
from PyQt5.QtWidgets import QSlider, QLabel
from PyQt5.QtCore import Qt
import PyQt5.QtCore as QtCore
from loguru import logger as log

from audioviz.engine import RippleEngine
from audioviz.ripple_ui import ControlPanel


class RippleWaveVisualizer(QtWidgets.QWidget):
    def __init__(self,
                 plane_size_m: tuple,
                 dx: float,
                 speed: float,
                 damping: float,
                 use_gpu: bool = True):
        super().__init__()

        self.control_panel: Optional[ControlPanel] = None

        self.log_counter_ = 0
        self.use_gpu = use_gpu
        self.dx = dx
        Nx = int(plane_size_m[0] / dx)
        Ny = int(plane_size_m[1] / dx)
        self.resolution = (Ny, Nx)

        self.engine = RippleEngine(
            resolution=self.resolution,
            dx=dx,
            speed=speed,
            damping=damping,
            use_gpu=use_gpu,
        )

        self.timer = QtCore.QTimer()
        self.timer.setInterval(16)  # ~60 FPS
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start()

        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        toggle_button = QtWidgets.QPushButton("Show Controls")
        toggle_button.clicked.connect(self.toggle_controls)
        layout.addWidget(toggle_button)

        self.image_item = pg.ImageItem()
        cmap = cm.get_cmap("inferno")
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        self.image_item.setLookupTable(lut)

        self.plot = pg.PlotItem()
        self.plot.setTitle("Ripple Simulation")
        self.plot.addItem(self.image_item)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.addItem(self.plot)
        layout.addWidget(self.plot_widget)

        self.colorbar = pg.HistogramLUTItem()
        self.colorbar.setImageItem(self.image_item)
        self.plot_widget.addItem(self.colorbar)
        self.colorbar.gradient.setColorMap(pg.ColorMap(
            pos=np.linspace(0, 1, lut.shape[0]),
            color=lut,
        ))

        self.setLayout(layout)

    def update_visualization(self):
        self.engine.time += self.engine.dt
        self.engine.update(self.engine.time)

        Z = self.engine.get_field()
        Z_vis = cp.asnumpy(Z) if self.use_gpu else Z

        self.image_item.setImage(Z_vis, autoLevels=False)

        self.log_counter_ += 1
        if self.log_counter_ == 30:
            log.debug(f"min: {Z_vis.min()}, max: {Z_vis.max()}, mean: {Z_vis.mean()}")
            self.log_counter_ = 0


    def toggle_controls(self):
        if self.control_panel is None:
            from audioviz.ripple_ui import ControlPanel
            self.control_panel = ControlPanel(self.engine)
            self.control_panel.resize(300, 600)
        self.control_panel.show()


class RippleWaveVisualizer3D(QtWidgets.QWidget):
    def __init__(self,
                 plane_size_m: tuple,
                 dx: float,
                 speed: float,
                 damping: float,
                 use_gpu: bool = True):
        super().__init__()

        self.control_panel: Optional[ControlPanel] = None

        self.use_gpu = use_gpu
        self.dx = dx
        Nx = int(plane_size_m[0] / dx)
        Ny = int(plane_size_m[1] / dx)
        self.resolution = (Ny, Nx)
        self.x = np.linspace(-1, 1, Nx)
        self.y = np.linspace(-1, 1, Ny)

        self.engine = RippleEngine(
            resolution=self.resolution,
            dx=dx,
            speed=speed,
            damping=damping,
            use_gpu=use_gpu,
        )

        self._init_ui()

        from matplotlib import cm
        self.cmap = cm.get_cmap("inferno")

        self.timer = QtCore.QTimer()
        self.timer.setInterval(16)  # ~60 FPS
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start()

    # Normalize z between 0 and 1
    def get_colors(self, Z_vis):
        z_norm = (Z_vis - Z_vis.min()) / (Z_vis.max() - Z_vis.min() + 1e-8)
        colors = self.cmap(z_norm)
        colors = colors[:, :, :4]  # RGBA
        return colors


    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        toggle_button = QtWidgets.QPushButton("Show Controls")
        toggle_button.clicked.connect(self.toggle_controls)
        layout.addWidget(toggle_button)

        self.view = gl.GLViewWidget()
        self.view.setWindowTitle('3D Ripple Surface')
        self.view.setCameraPosition(distance=30, elevation=30, azimuth=45)
        layout.addWidget(self.view)

        self.surface = gl.GLSurfacePlotItem(computeNormals=False)
        self.surface.setGLOptions('opaque')
        self.surface.opts['smooth'] = True
        self.view.addItem(self.surface)

        self.setLayout(layout)

    def update_visualization(self):
        self.engine.time += self.engine.dt
        self.engine.update(self.engine.time)

        Z = self.engine.get_field()
        Z_vis = cp.asnumpy(Z) if self.use_gpu else Z
        colors = self.get_colors(Z_vis)

        Ny, Nx = Z_vis.shape

        self.surface.setData(x=self.x, y=self.y, z=Z_vis, colors=colors)

    def toggle_controls(self):
        if self.control_panel is None:
            self.control_panel = ControlPanel(self.engine)
            self.control_panel.resize(300, 600)
        self.control_panel.show()
