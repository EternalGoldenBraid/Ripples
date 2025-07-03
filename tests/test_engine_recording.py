import numpy as np
import time
import os

from audioviz.engine import RippleEngine
from utils import MockSource  # Make sure this exists as in previous example

class TestEngineRecording:

    def setup_method(self):
        """Shared setup for each test."""
        resolution = (20, 20)
        self.engine = RippleEngine(
            resolution,
            dx=0.01,
            speed=1.0,
            damping=0.99,
            use_gpu=False,
            ram_budget_gb=0.001,    # small, so we can test quickly
            disk_budget_gb=0.002,
            num_disk_buffers=2,
        )

        self.iy, self.ix = 10, 10
        self.mock_source = MockSource(self.iy, self.ix, resolution)
        self.engine.set_probe_location(self.iy, self.ix)
        self.engine.add_source(self.mock_source)

    def test_basic_recording(self):
        self.engine.enable_recording()

        for i in range(5):
            t = i * 0.1
            self.engine.update(t)

        self.engine.disable_recording()

        assert self.engine._ram_frame_idx == 5
        assert self.engine.Z_ram_online.shape[1:] == (20, 20)

        recorded_vals = self.engine.probe_ram_online[:5]
        assert np.any(recorded_vals != 0)
        assert recorded_vals.shape == (5,)

        print("✅ Basic in-memory recording test passed!")

    def test_fill_and_render(self):
        self.engine.enable_recording()

        # Compute approximate frames to fill at least one buffer
        frames_to_fill = self.engine.disk_buffer_size*4 
        print(f"Filling {frames_to_fill} frames to test disk buffer...")

        for i in range(frames_to_fill):
            t = i * 0.05
            self.engine.update(t)

        self.engine.disable_recording()

        # Wait a moment for threads to finish writing and potentially render
        time.sleep(5)

        # Check that disk buffer files were created
        for path in self.engine._disk_paths:
            assert os.path.exists(path), f"Disk buffer missing: {path}"

        # Check if any rendered video exists
        renders = list(self.engine.render_dir.glob("render_*.mp4"))
        assert len(renders) > 0, "No rendered videos found!"

        print("✅ Disk buffer filling & render test passed!")

if __name__ == "__main__":
    tester = TestEngineRecording()
    # tester.setup_method()
    # tester.test_basic_recording()
    tester.setup_method()  # reset
    tester.test_fill_and_render()
