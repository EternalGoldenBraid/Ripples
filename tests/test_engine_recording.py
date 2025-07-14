import numpy as np
import time
from pathlib import Path
import shutil


from loguru import logger as log


from audioviz.engine import RippleEngine


class TestRippleEngineRecorder:
    def __init__(self):
        # Clean up outputs before starting
        outputs = Path("outputs")
        if outputs.exists():
            shutil.rmtree(outputs/"disk_buffers")
            shutil.rmtree(outputs/"renders")
        outputs.mkdir(exist_ok=True)

        # ────────────── Setup engine ──────────────
        resolution = (64, 64)
        dx = 0.01
        speed = 1.0
        damping = 0.05

        # Use small RAM budget to force quick flush
        self.engine = RippleEngine(
            resolution=resolution,
            dx=dx,
            speed=speed,
            damping=damping,
            use_gpu=False,
            ram_budget_gb=0.01,  # very small buffer
            disk_budget_gb=0.05,
            num_disk_buffers=2,
        )
        # self.audio_chunk = np.random.randn(self.engine.samples_per_frame).astype(np.float32) * 0.1
        # Generate a 100 Hz sine tone (audible)
        sr = self.engine.audio_sr
        t = np.arange(self.engine.samples_per_frame) / sr
        freq = 100.0  # Hz
        sine = 0.05 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        self.audio_chunk = sine

    def test_full_disk_fill(self):
        """Test case where we fill disk buffers fully and trigger rendering."""

        self.engine.enable_recording()

        # Calculate frames to fully fill all disk buffers at least once
        frames_to_fill_all_disks = (self.engine.disk_buffer_size * self.engine.num_disk_buffers)*8

        for i in range(frames_to_fill_all_disks):
            self.engine.feed_audio(self.audio_chunk)
            self.engine.update(i * self.engine.dt)

        self.engine.disable_recording()

        # Check outputs
        self._print_results("Full disk fill test")

    def test_partial_ram_only_fill(self):
        """Test case where we fill only partial RAM buffer, no disk flush yet."""

        # Reinitialize engine to reset state
        self.__init__()

        self.engine.enable_recording()

        # Fill with fewer frames than one full RAM buffer
        frames_to_partial_ram = max(1, self.engine.ram_buffer_size // 4)

        for i in range(frames_to_partial_ram):
            self.engine.feed_audio(self.audio_chunk)
            self.engine.update(i * self.engine.dt)

        log.debug(f"TEST: Waiting for any async operations to complete...")
        time.sleep(2)  # Allow some time for any async operations
        self.engine.disable_recording()

        time.sleep(2)  # Allow some time for any async operations
        # Check outputs
        self._print_results(f"Partial RAM-only fill test")

    def _print_results(self, title: str):
        disk_buffers = list(Path("outputs/disk_buffers").glob("buffer_*.h5"))
        renders = list(Path("outputs/renders").glob("render_*.mp4"))

        print(f"\n========== {title} ==========")
        print(f"✅ Disk buffers: {len(disk_buffers)} found")
        for buf in disk_buffers:
            print(f"  - {buf}")

        print(f"🎥 MP4 renders: {len(renders)} found")
        for mp4 in renders:
            print(f"  - {mp4}")

        session_mp4 = list(Path("outputs/renders").glob("session_*.mp4"))
        print(f"🟢 Session MP4: {session_mp4[0] if session_mp4 else 'Not found'}")

        raw_wav = list(Path("outputs/renders").glob("slice_*.raw.wav"))
        print(f"🎧 Raw WAV slice: {raw_wav[0] if raw_wav else 'Not found'}")

# ────────────── Run tests ──────────────
if __name__ == "__main__":
    tester = TestRippleEngineRecorder()
    # tester.test_full_disk_fill()
    tester.test_partial_ram_only_fill()
