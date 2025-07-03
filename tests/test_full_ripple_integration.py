import sys
import numpy as np
from PyQt5 import QtWidgets
import qdarkstyle
from audioviz.engine import RippleEngine
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer
from audioviz.ripple_ui import ControlPanel
from audioviz.sources.synthetic import SyntheticPointExcitation

def test_full_ripple_integration():
    app = QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # Create engine
    resolution = (200, 200)
    engine = RippleEngine(resolution=resolution, dx=0.01, speed=10.0, damping=0.95, use_gpu=False)

    # Add a synthetic source
    synthetic = SyntheticPointExcitation(
        name="Synthetic Ripple",
        dx=0.01,
        resolution=resolution,
        position=(0.5, 0.5),
        frequency=10,
        speed=10.0,
        backend=np,
    )
    engine.add_source(synthetic)

    # Create visualizer
    visualizer = RippleWaveVisualizer(
        plane_size_m=(2.0, 2.0),
        dx=0.01,
        speed=10.0,
        damping=0.95,
        use_gpu=False,
    )
    visualizer.engine = engine

    visualizer.setWindowTitle("Full Ripple Visualizer")
    visualizer.resize(600, 600)
    visualizer.show()

    # Create control panel
    control_panel = ControlPanel(engine)
    control_panel.resize(300, 600)
    control_panel.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    test_full_ripple_integration()
