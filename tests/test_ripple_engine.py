
# test_ripple_engine.py
import numpy as np
from audioviz.engine import RippleEngine
from audioviz.sources.synthetic import SyntheticPointExcitation

def test_engine_basic_update():
    resolution = (4, 4)
    engine = RippleEngine(resolution=resolution, dx=0.01, speed=10.0, damping=0.95, use_gpu=False)

    synthetic = SyntheticPointExcitation(
        name="Test Synthetic",
        dx=0.01,
        resolution=resolution,
        position=(0.5, 0.5),
        frequency=10,
        speed=10.0,
        amplitude=1.0,
        backend=np,
    )
    engine.add_source(synthetic)

    t = 0.0
    for i in range(5):
        engine.update(t)
        field = engine.get_field()
        assert field.shape == resolution
        if i > 0: assert np.any(field != 0)
        t += engine.dt

    print("✅ RippleEngine basic update test passed.")

if __name__ == "__main__":
    test_engine_basic_update()
