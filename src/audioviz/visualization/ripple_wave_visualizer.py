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

class AdaptiveEMA:

    def __init__(self):
        self.vmin_ema: Optional[float] = None
        self.vmax_ema: Optional[float] = None

    def __call__(self, Z_vis: np.ndarray,
                 Z: np.ndarray, 
                    alpha: float = 0.1,
                    log_scale: bool = False,
                    mask: Optional[np.ndarray] = None
                    ) -> None:
        """
        Applies adaptive normalization to a 2D array using EMA of percentiles.
        
        Parameters:
        -----------
        Z         : np.ndarray      – input array (2D, typically the wave field)
        alpha     : float            – smoothing factor for EMA [0–1]
        log_scale : bool             – apply log1p(abs(Z)) * sign(Z) before scaling
        
        Returns:
        --------
        Z_norm    : np.ndarray       – normalized image in [-1, 1]
        """
        Z_proc = Z if mask is None else Z[mask]
        if log_scale:
            Z_proc = np.log1p(np.abs(Z_proc)) * np.sign(Z_proc)
        else:
            Z_proc = Z_proc

        # Percentile-based bounds
        vmin_t, vmax_t = np.percentile(Z_proc, [1, 99])
        # vmin_t, vmax_t = np.percentile(Z_proc, [1, 99])
        # vmin_t, vmax_t = np.percentile(Z_proc, [5, 95])

        # Update EMA bounds
        if self.vmin_ema is None or self.vmax_ema is None:
            # Initialize with the first values
            new_vmin, new_vmax = (1-alpha)*vmin_t, (1-alpha)*vmax_t
        else:
            new_vmin = (1 - alpha) * self.vmin_ema + alpha * vmin_t
            new_vmax = (1 - alpha) * self.vmax_ema + alpha * vmax_t

        # Normalize to [-1, 1]
        Z_norm = np.clip((Z_proc - new_vmin) / (new_vmax - new_vmin + 1e-6), 0, 1)
        Z_norm = Z_norm * 2 - 1

        self.vmin_ema, self.vmax_ema = new_vmin, new_vmax
        Z_vis[mask] = Z_norm



class RippleWaveVisualizer(QtWidgets.QWidget):
    def __init__(self,
                 plane_size_m: tuple,
                 dx: float,
                 speed: float,
                 damping: float,
                 use_gpu: bool = True):
        super().__init__()

        self.normalize: AdaptiveEMA = AdaptiveEMA()

        self.control_panel: Optional[ControlPanel] = None

        self.log_counter_ = 0
        self.use_gpu = use_gpu
        self.dx = dx
        Nx = int(plane_size_m[0] / dx)
        Ny = int(plane_size_m[1] / dx)
        self.resolution = (Ny, Nx)
        self.Z_ema = np.zeros(self.resolution, dtype=np.float32)
        self.Z_vis = np.ones(self.resolution, dtype=np.float32)*-1
        self.Z = np.zeros(self.resolution, dtype=np.float32)

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

        # Normalization
        self.vmin_ema = None
        self.vmax_ema = None
        self.alpha = 0.9  # TODO add to controls


    def update_visualization(self):
        self.Z_vis[:] = -1.0  # Reset visualization array
        self.engine.time += self.engine.dt

        return_dict: Dict = self.engine.update(self.engine.time)
        overlay = return_dict.get("overlay", None)
        alpha = return_dict.get("alpha", 0.0)
        boundary_mask = return_dict.get("boundary_mask", None)
        self.Z[:] = cp.asnumpy(self.engine.get_field()) if self.use_gpu else self.engine.get_field()

        # Z_vis = cp.asnumpy(Z) if self.use_gpu else Z
        # Z_vis = Z_vis - np.mean(Z_vis)
        # if overlay is not None:
        #     Z_vis = (1-alpha) * Z_vis + alpha * cp.asnumpy(overlay)
        # self.image_item.setImage(Z_vis, autoLevels=False)

        self.normalize(Z_vis=self.Z_vis, Z=self.Z,
                    # alpha=self.alpha,
                    alpha=0.8,
                    log_scale=False, mask=boundary_mask
        )
        if overlay is not None:
            self.Z_vis[:] = (1-alpha) * self.Z_vis + alpha * cp.asnumpy(overlay)
        self.image_item.setImage(self.Z_vis, levels=(-1.0,1.2))

        # # --- fast residual image ---------------------------------
        # Z_raw = Z if not self.use_gpu else cp.asnumpy(Z)
        # self.Z_ema = 0.98 * self.Z_ema + 0.02 * Z_raw
        # Z_residual = Z_raw - self.Z_ema
        # Z_vis = self.adaptive_normalize(Z_residual, alpha=0.05)
        # self.image_item.setImage(Z_vis, levels=(-1, 1))


        self.log_counter_ += 1
        if self.log_counter_ == 30:
            log.debug(
                f"""
                Z_vis: min: {self.Z_vis.min()}, max: {self.Z_vis.max()}, mean: {self.Z_vis.mean()} 
                Z: min: {Z.min()}, max: {Z.max()}, mean: {Z.mean()}
                """
            )
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
        self.normalize: AdaptiveEMA = AdaptiveEMA()

        self.use_gpu = use_gpu
        self.dx = dx
        Nx = int(plane_size_m[0] / dx)
        Ny = int(plane_size_m[1] / dx)
        self.resolution = (Ny, Nx)
        self.x = np.linspace(-1, 1, Nx)
        self.y = np.linspace(-1, 1, Ny)
        self.Z_vis = np.ones(self.resolution, dtype=np.float32)*-1
        self.Z = np.zeros(self.resolution, dtype=np.float32)

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
    def get_colors(self, Z_vis: np.ndarray, Z: np.ndarray, mask):
        # z_norm = (Z_vis - Z_vis.min()) / (Z_vis.max() - Z_vis.min() + 1e-8)
        self.normalize(Z_vis=Z_vis, Z=Z, alpha=0.8, log_scale=False, mask=mask)
        colors = self.cmap(Z_vis)
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
        self.Z_vis[:] = -1.0  # Reset visualization array
        self.engine.time += self.engine.dt

        return_dict: Dict = self.engine.update(self.engine.time)
        overlay = return_dict.get("overlay", None)
        alpha = return_dict.get("alpha", 0.0)
        boundary_mask = return_dict.get("boundary_mask", None)

        self.Z[:] = cp.asnumpy(self.engine.get_field()) if self.use_gpu else self.engine.get_field()
        self.Z_vis[:] = self.Z[:]  
        self.Z_vis -= np.mean(self.Z_vis)
        colors = self.get_colors(Z_vis=self.Z_vis, Z=self.Z, mask=boundary_mask)

        self.surface.setData(x=self.x, y=self.y, z=self.Z_vis, colors=colors)

    def toggle_controls(self):
        if self.control_panel is None:
            self.control_panel = ControlPanel(self.engine)
            self.control_panel.resize(300, 600)
        self.control_panel.show()
