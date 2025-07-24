import sys
import numpy as np
from PyQt5 import QtWidgets
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer
from audioviz.engine import RippleEngine
from audioviz.sources.synthetic import SyntheticPointExcitation

def test_visualizer_with_engine():
    app = QtWidgets.QApplication([])

    resolution = (200, 200)
    engine = RippleEngine(resolution=resolution, dx=0.01, speed=10.0, damping=0.95, use_gpu=False)

    synthetic = SyntheticPointExcitation(
        name="Test Synthetic",
        dx=0.01,
        resolution=resolution,
        position=(0.5, 0.5),
        frequency=10,
        speed=10.0,
        backend=np,
    )
    engine.add_source(synthetic)

    visualizer = RippleWaveVisualizer(
        plane_size_m=(2.0, 2.0),
        dx=0.01,
        speed=10.0,
        damping=0.95,
        use_gpu=False,
    )
    visualizer.engine = engine

    visualizer.setWindowTitle("Ripple Visualizer Test")
    visualizer.resize(600, 600)
    visualizer.show()

    # This is manual — will need to close window manually or add a timer
    sys.exit(app.exec())

if __name__ == "__main__":
    test_visualizer_with_engine()
