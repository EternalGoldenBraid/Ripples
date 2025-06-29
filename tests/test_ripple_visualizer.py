import sys
from PyQt5 import QtWidgets
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer
from audioviz.sources.synthetic import SyntheticPointExcitation

class TestRippleVisualizer:
    def test_synthetic_source(self) -> None:
        app = QtWidgets.QApplication([])

        conf = dict(
            plane_size_m=(10., 10.),
            dx=5e-3,
            speed=10.0,
            damping=0.90,
            use_gpu=True,
        )
        ripple = RippleWaveVisualizer(**conf)
        ripple.setWindowTitle("Ripple Field - Synthetic Test")
        ripple.resize(600, 600)

        synthetic_source = SyntheticPointExcitation(
            name="Synthetic Ripple",
            dx=conf["dx"],
            resolution=ripple.resolution,
            position=(0.5, 0.5),
            frequency=400,
            speed=conf["speed"],
            backend=ripple.backend,
        )
        ripple.add_excitation_source(synthetic_source)

        ripple.show()

        sys.exit(app.exec())

if __name__ == "__main__":
    tester = TestRippleVisualizer()
    tester.test_synthetic_source()
