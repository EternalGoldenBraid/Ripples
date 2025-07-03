from typing import Dict
import json
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QSlider, QLabel
from PyQt5.QtCore import Qt
from audioviz.sources.base import ExcitationSourceBase
from audioviz.engine import RippleEngine

class ControlPanel(QtWidgets.QWidget):
    def __init__(self, engine: RippleEngine):
        super().__init__()
        self.engine = engine
        self.setWindowTitle("Ripple Controls")
        self.layout = QtWidgets.QVBoxLayout(self)

        # Save/load buttons at top
        save_button = QtWidgets.QPushButton("Save Preset")
        save_button.clicked.connect(self.save_preset)
        self.layout.addWidget(save_button)

        load_button = QtWidgets.QPushButton("Load Preset")
        load_button.clicked.connect(self.load_preset)
        self.layout.addWidget(load_button)

        # Add source controls
        self.populate_controls()

        self.show()

    def populate_controls(self):
        # First: Global wave controls
        self.add_wave_controls()
    
        # Then: Source controls
        for source in self.engine.sources.values():
            self.add_source_controls(source)
    
    def add_wave_controls(self):
        group = QtWidgets.QGroupBox("Wave Physics")
        group_layout = QtWidgets.QVBoxLayout(group)
    
        # Reset button
        reset_button = QtWidgets.QPushButton("Reset Field")
        reset_button.clicked.connect(self.reset_field)
        group_layout.addWidget(reset_button)
    
        # Damping
        damping_label = QLabel(f"Damping: {self.engine.damping:.3f}")
        damping_slider = QSlider(Qt.Horizontal)
        damping_slider.setMinimum(0)
        damping_slider.setMaximum(100000)
        damping_slider.setValue(int(self.engine.damping * 1000))
        damping_slider.valueChanged.connect(lambda val: self.update_damping(val / 1000, damping_label))
        group_layout.addWidget(QtWidgets.QLabel("Wave Damping"))
        group_layout.addWidget(damping_label)
        group_layout.addWidget(damping_slider)
    
        # Speed
        speed_label = QLabel(f"Speed: {self.engine.speed:.1f}")
        speed_slider = QSlider(Qt.Horizontal)
        speed_slider.setMinimum(1)
        speed_slider.setMaximum(1000)
        speed_slider.setValue(int(self.engine.speed))
        speed_slider.valueChanged.connect(lambda val: self.update_speed(val, speed_label))
        group_layout.addWidget(QtWidgets.QLabel("Wave Speed (m/s)"))
        group_layout.addWidget(speed_label)
        group_layout.addWidget(speed_slider)


        # Engine recording controls
        record_start_btn = QtWidgets.QPushButton("Start Recording")
        record_start_btn.clicked.connect(self.engine.enable_recording)
        group_layout.addWidget(record_start_btn)
        
        record_stop_btn = QtWidgets.QPushButton("Stop & Save Recording")
        record_stop_btn.clicked.connect(lambda: (self.engine.disable_recording(), self.engine.save_recording()))
        group_layout.addWidget(record_stop_btn)

        group.setLayout(group_layout)
        self.layout.addWidget(group)
    
    def reset_field(self):
        self.engine.propagator.reset()
        self.engine.Z[:] = 0
        self.engine.excitation[:] = 0
        self.engine.time = 0.0
        print("✅ Field reset.")
    
    def update_damping(self, val, label):
        self.engine.damping = val
        if hasattr(self.engine.propagator, "damping"):
            self.engine.propagator.damping = val
        label.setText(f"Damping: {val:.3f}")
        print(f"✅ Updated damping to {val:.3f}")
    
    def update_speed(self, val, label):
        self.engine.speed = val
        if hasattr(self.engine.propagator, "c"):
            self.engine.propagator.c = val
            self.engine.propagator.c2_dt2 = (val * self.engine.dt / self.engine.dx)**2
        label.setText(f"Speed: {val:.1f}")
        print(f"✅ Updated speed to {val:.1f}")

    def add_source_controls(self, source: ExcitationSourceBase) -> None:
        group = QtWidgets.QGroupBox(source.name)
        group_layout = QtWidgets.QVBoxLayout(group)

        for key, cfg in source.get_controls():
            ctrl_type = cfg.get("type", "slider")
            tooltip = cfg.get("tooltip", "")

            if ctrl_type == "checkbox":
                widget = QtWidgets.QCheckBox(cfg["label"])
                widget.setChecked(bool(cfg["init"]))
                widget.setToolTip(tooltip)
                widget.stateChanged.connect(cfg["on_change"])
                group_layout.addWidget(widget)

            elif ctrl_type == "slider":
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
                        fmt = "{:.2f}" if pname in ["alpha", "amplitude", "gain", "damping"] else "{}"
                        lbl.setText(fmt.format(v / 100 if pname in ["alpha", "amplitude", "gain"] else v))
                    return inner
                slider.valueChanged.connect(_update())

                group_layout.addWidget(label)
                group_layout.addWidget(val_lbl)
                group_layout.addWidget(slider)

            else:
                raise ValueError(f"Unknown control type '{ctrl_type}' for source '{source.name}'.")

        group.setLayout(group_layout)
        self.layout.addWidget(group)

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name:
            return
        config: Dict = {}

        for src_name, source in self.engine.sources.items():
            config[src_name] = {}
            for key, cfg in source.get_controls():
                config[src_name][key] = cfg.get("init", 0)

        out_path = f"outputs/{name}.json"
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Preset saved to {out_path}")

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Preset", "outputs/", "JSON Files (*.json)")
        if not path:
            return

        with open(path, "r") as f:
            config = json.load(f)

        for src_name, params in config.items():
            source = self.engine.sources.get(src_name)
            if not source:
                continue
            for key, val in params.items():
                for k, cfg in source.get_controls():
                    if k == key:
                        cfg["on_change"](val)
        print(f"✅ Preset loaded from {path}")
