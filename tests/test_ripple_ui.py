import sys
import numpy as np
from PyQt5 import QtWidgets
from audioviz.ripple_ui import ControlPanel
from audioviz.engine import RippleEngine
from audioviz.sources.synthetic import SyntheticPointExcitation

def test_control_panel_with_source():
    app = QtWidgets.QApplication([])

    resolution = (200, 200)
    engine = RippleEngine(resolution=resolution, dx=0.01, speed=10.0, damping=0.95, use_gpu=False)

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

    panel = ControlPanel(engine)
    panel.setWindowTitle("Control Panel Test")
    panel.resize(400, 600)
    panel.show()

    sys.exit(app.exec())



if __name__ == "__main__":
    test_control_panel_with_source()
