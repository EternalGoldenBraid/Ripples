import sys
import numpy as np
from PyQt5 import QtWidgets
import qdarkstyle

from audioviz.engine import RippleEngine
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer3D

def test_ripple_visualizer_3d_basic():
    app = QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    resolution = (50, 50)
    engine = RippleEngine(
        resolution=resolution,
        dx=0.01,
        speed=10.0,
        damping=0.95,
        use_gpu=False,
    )

    visualizer = RippleWaveVisualizer3D(
        plane_size_m=(1.0, 1.0),
        dx=0.01,
        speed=10.0,
        damping=0.95,
        use_gpu=False,
    )
    visualizer.engine = engine

    # Manually call update (simulate timer tick)
    try:
        visualizer.update_visualization()
        print("✅ 3D Visualizer update executed without errors.")
    except Exception as e:
        print("❌ Error during 3D visualizer update:", e)
        assert False, "3D visualizer update failed"

    visualizer.resize(600, 600)
    visualizer.show()

    # Comment this out if running headless or want automatic close
    sys.exit(app.exec())

if __name__ == "__main__":
    test_ripple_visualizer_3d_basic()
