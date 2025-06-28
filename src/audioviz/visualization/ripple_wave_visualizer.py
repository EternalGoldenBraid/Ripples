from typing import Optional, Tuple, Dict, Any

import numpy as np
import cupy as cp
import matplotlib.cm as cm
from PyQt5 import QtWidgets
import pyqtgraph as pg
from PyQt5.QtWidgets import QSlider, QLabel
from PyQt5.QtCore import Qt
from loguru import logger as log


from audioviz.visualization.visualizer_base import VisualizerBase
from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.sources.base import ExcitationSourceBase
from audioviz.physics.wave_propagator import WavePropagatorCPU, WavePropagatorGPU

class RippleWaveVisualizer(VisualizerBase):
    """
    Simulates 2D ripple propagation in response to multiple excitation sources.

    Parameters:
        plane_size_m (Tuple[float, float]): Size of the wave simulation surface in meters.
        resolution (Tuple[int, int]): Pixel resolution of the field (H, W).
        speed (float): Wave propagation speed (m/s).
        damping (float): Wave energy loss factor per frame.
        use_synthetic (bool): Whether to inject synthetic excitation (deprecated).
        use_gpu (bool): Use CuPy for GPU acceleration.

    Attributes:
        excitation_sources (Dict[str, ExcitationSourceBase]): Registered sources.
        propagator (WavePropagatorCPU | WavePropagatorGPU): Wave simulation engine.
        Z (array): Current wave state.
        dt (float): Time step.
        dx, dy (float): Spatial resolution in meters.
        image_item (ImageItem): Render target.

    Methods:
        update_visualization(): Apply all sources, simulate wave propagation, update view.
        add_excitation_source(): Register new external source.
        _init_wave_controls(): UI controls for wave physics (damping, speed).
        get_controls(): Returns UI sliders for external source parameters.

    TODO:
        - Support field-of-view scaling separate from physical simulation size.
        - Decouple resolution from simulation domain (support zoom and crop).
        - Add source grouping or toggling in UI.

    Caveats:
        - Source coordinates are still in pixel space.
        - UI assumes single main plot; no support for 3D or overlays.
    """
    def __init__(self,
                 plane_size_m: Tuple[float, float],
                 resolution: Tuple[int, int],
                 speed: float,
                 damping: float,

                 use_synthetic: bool = True,

                 use_gpu: bool = False,
                 **kwargs):

        super().__init__(**kwargs)

        self.excitation_sources: Dict[str, ExcitationSourceBase] = {}
        self.use_gpu = use_gpu
        self.xp = cp if use_gpu else np

        self.plane_size_m = plane_size_m
        self.resolution = resolution
        self.excitation = self.xp.zeros(self.resolution, dtype=np.float32)
        self.speed = speed
        self.time = 0.0


        self.dx = self.plane_size_m[0] / self.resolution[0]
        self.dy = self.plane_size_m[1] / self.resolution[1]
        self.dt = (max(self.dx, self.dy) / speed) * 1 / np.sqrt(2)

        propagator_kwargs = {
            "shape": self.resolution,
            "dx": self.dx,
            "dt": self.dt,
            "speed": self.speed,
            "damping": damping
        }
        if use_gpu:
            self.propagator = WavePropagatorGPU(**propagator_kwargs)
        else:
            self.propagator = WavePropagatorCPU(**propagator_kwargs)

        self.Z = self.xp.zeros(self.resolution, dtype=self.xp.float32)

        self.max_frequency = self.speed / (2 * max(self.dx, self.dy))

        # Grid for ripple
        self.source_positions = []
        np.random.seed(42)
        x = np.random.randint(0, resolution[0])
        y = np.random.randint(0, resolution[1])
        self.source_position: Tuple[int, int] = (x, y)

        self.xs, self.ys = self.xp.meshgrid(
            self.xp.arange(self.resolution[1]),
            self.xp.arange(self.resolution[0])
        )


        ## Create the plot widget and image item ##

        layout = QtWidgets.QVBoxLayout(self)

        self.image_item = pg.ImageItem()
        # cmap = cm.get_cmap("seismic")
        cmap = cm.get_cmap("inferno")
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        self.image_item.setLookupTable(lut)

        self.plot = pg.PlotItem()
        self.plot.setTitle("Ripple Simulation")
        self.plot.addItem(self.image_item)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.addItem(self.plot)
        layout.addWidget(self.plot_widget)

        ## Controls ##
        self._init_wave_controls(layout)  # Add damping/speed sliders here
        self.control_layout = layout  # So other sources can append here

    def _init_wave_controls(self, layout: QtWidgets.QVBoxLayout):

        def _update_speed(val: float):
            self.speed = val
            self.speed_label.setText(f"Speed: {val:.1f}")
            self.dt = (max(self.dx, self.dy) / self.speed) * 1 / np.sqrt(2)
            self.propagator.dt = self.dt
            self.propagator.c = self.speed
            self.propagator.c2_dt2 = (self.speed * self.dt / self.dx)**2

        def _update_damping(val: float):
            self.propagator.damping = val
            self.damping_label.setText(f"Damping: {val:.3f}")

        # Damping
        self.damping_label = QLabel("Damping: 0.999")
        damping_slider = QSlider(Qt.Horizontal)
        damping_slider.setMinimum(0)
        damping_slider.setMaximum(1000)
        damping_slider.setValue(999)
        damping_slider.valueChanged.connect(
            lambda val: _update_damping(val / 1000)
        )
        layout.addWidget(QLabel("Wave Damping"))
        layout.addWidget(self.damping_label)
        layout.addWidget(damping_slider)

        # Speed
        self.speed_label = QLabel(f"Speed: {self.speed:.1f}")
        speed_slider = QSlider(Qt.Horizontal)
        speed_slider.setMinimum(1)
        speed_slider.setMaximum(1000)
        speed_slider.setValue(int(self.speed))
        speed_slider.valueChanged.connect(
            lambda val: _update_speed(val)
        )
        layout.addWidget(QLabel("Wave Speed (m/s)"))
        layout.addWidget(self.speed_label)
        layout.addWidget(speed_slider)


    def add_source_controls(self, source: ExcitationSourceBase) -> None:
        """Create a QGroupBox with all UI controls declared by a source."""

        group = QtWidgets.QGroupBox(source.name)
        group_layout = QtWidgets.QVBoxLayout(group)

        for key, cfg in source.get_controls():
            ctrl_type = cfg.get("type", "slider")      # default → slider
            tooltip   = cfg.get("tooltip", "")

            if ctrl_type == "checkbox":
                widget = QtWidgets.QCheckBox(cfg["label"])
                widget.setChecked(bool(cfg["init"]))
                widget.setToolTip(tooltip)
                widget.stateChanged.connect(cfg["on_change"])
                group_layout.addWidget(widget)

            elif ctrl_type == "slider":                # legacy + default
                label = QLabel(cfg["label"])
                label.setToolTip(tooltip)

                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(cfg["min"])
                slider.setMaximum(cfg["max"])
                slider.setValue(cfg["init"])
                slider.valueChanged.connect(cfg["on_change"])

                val_lbl = QLabel(str(cfg["init"]))

                def _update(lbl=val_lbl, pname=key):
                    def inner(v):
                        fmt = "{:.2f}" if ("amplitude" in pname or "alpha" in pname) else "{}"
                        lbl.setText(fmt.format(v / 100 if ("amplitude" in pname or "alpha" in pname) else v))
                    return inner
                slider.valueChanged.connect(_update())

                group_layout.addWidget(label)
                group_layout.addWidget(val_lbl)
                group_layout.addWidget(slider)

            else:
                raise ValueError(
                    f"Unknown control type '{ctrl_type}' for source '{source.name}'.\n"
                    "Supported types: 'slider', 'checkbox'."
                )

        group.setLayout(group_layout)
        self.control_layout.addWidget(group)


    def add_source_controls_old(self, source: ExcitationSourceBase):
        group = QtWidgets.QGroupBox(source.name)
        group_layout = QtWidgets.QVBoxLayout(group)
    
        for param_name, config in source.get_controls():
            ctrl_type = config.get("type", "slider")

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(config["min"])
            slider.setMaximum(config["max"])
            slider.setValue(config["init"])
            slider.valueChanged.connect(config["on_change"])
    
            label = QLabel(config["label"])
            label.setToolTip(config.get("tooltip", ""))
            value_label = QLabel(str(config["init"]))
    
            def update_label(val, lbl=value_label, pname=param_name):
                fmt = "{:.2f}" if "amplitude" in pname or "alpha" in pname else "{}"
                lbl.setText(fmt.format(val / 100 if "amplitude" in pname or "alpha" in pname else val))
    
            slider.valueChanged.connect(update_label)
    
            group_layout.addWidget(label)
            group_layout.addWidget(value_label)
            group_layout.addWidget(slider)
    
        group.setLayout(group_layout)
        self.control_layout.addWidget(group)


    def add_excitation_source(self, source: ExcitationSourceBase):
        """Add an excitation source to the visualizer."""

        if source.name in self.excitation_sources:
            err = f"Excitation source '{source.name}' already exists."
            log.error(err)
            raise ValueError(err)

        self.excitation_sources[source.name] = source
        log.info(f"Added excitation source '{source.name}'.")

        self.add_source_controls(source)

    def update_visualization(self):
        self.time += self.dt
        weight: float = 1/len(self.excitation_sources)

        for name, source in self.excitation_sources.items():
            self.excitation[:] += weight*source(self.time)

            # log.debug(f"Excitation source '{name}' at time {self.time:.3f}s: {source(self.time).shape} shape")

        self.propagator.add_excitation(self.excitation)
        self.propagator.step()
        self.Z[:] = self.propagator.get_state()

        Z_vis = cp.asnumpy(self.Z) if self.use_gpu else self.Z
        max_abs = np.max(np.abs(Z_vis))
        self.image_item.setLevels([-max_abs, max_abs])
        self.image_item.setImage(Z_vis, autoLevels=False)
        self.excitation[:] = 0  # Reset excitation for next step



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = RippleWaveVisualizer()
    widget.setWindowTitle("Ripple Wave (Synthetic)")
    widget.resize(600, 600)
    widget.show()
    sys.exit(app.exec())
