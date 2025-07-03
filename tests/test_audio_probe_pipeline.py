# tests/test_audio_probe_pipeline.py
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from audioviz.engine import RippleEngine


def tmp_engine(tmp_path):
    """RippleEngine with tiny buffers and stubbed render."""
    eng = RippleEngine(
        resolution=(10, 10),
        dx=0.01,
        speed=1.0,
        damping=0.99,
        use_gpu=False,
        ram_budget_gb=0.00001,   # ≈ 7 frames -> flushes fast
        disk_budget_gb=0.00002,
        num_disk_buffers=1,
    )

    # 1️⃣  Short-circuit the heavy FFmpeg path
    def fake_launch(idx: int):
        # pretend render succeeded, touch empty files
        out_mp4 = eng.render_dir / f"render_{idx:02d}.mp4"
        out_wav = eng.render_dir / f"render_{idx:02d}.wav"
        out_mp4.touch(); out_wav.touch()
        eng._disk_rendering[idx] = False
        eng._disk_ready[idx] = False

    eng._launch_ffmpeg = fake_launch  # type: ignore[assignment]

    return eng


def test_audio_modulation_pipeline(engine: RippleEngine, tmp_path: Path):
    eng = engine
    sr           = 44_100
    n_blocks     = 5                # simulate 5 GUI timer ticks
    block_size   = 512
    probe_frames = n_blocks         # 1 probe per update (60 Hz ≈ blocks here)

    eng.enable_recording()

    # -------  feed synthetic audio blocks  -------
    for i in range(n_blocks):
        pcm = np.ones(block_size, dtype=np.float32) * (i + 1)  # 1,2,3,4,5 pattern
        eng.feed_audio(pcm)                                    # ring-buffer test
        eng.update(i / sr)                                     # advance physics

    # Probe ring should contain the newest value
    assert eng._audio_idx == n_blocks * block_size

    # -------  stop recording to write raw WAV  -------
    eng.disable_recording()
    print("Recording stopped, checking audio output...")

    raw_wavs = list(eng.render_dir.glob("slice_*.raw.wav"))
    assert len(raw_wavs) == 1, "raw wav should be created"
    audio_raw, rsr = sf.read(raw_wavs[0])
    assert rsr == sr
    assert np.isclose(audio_raw.mean(), 3.0, atol=0.1), f"Unexpected audio mean: {audio_raw.mean()}"

    print(f"Raw audio shape: {audio_raw.shape}, mean: {audio_raw.mean()}")

    # -------  trigger render → creates modulated wav (stubbed) -------
    eng._disk_ready[0] = True     # force watcher to call fake render
    eng._launch_ffmpeg(0)  # manually trigger render
    mod_wavs = list(eng.render_dir.glob("render_*.wav"))
    assert mod_wavs, "modulated wav should exist"

    # In a real run the mod wav differs from raw (envelope applied)
    # Here we just check the stub touched the file.
    assert mod_wavs[0].stat().st_size == 0

    print("Audio modulation pipeline test passed!")

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        engine = tmp_engine(tmp_path)
        test_audio_modulation_pipeline(engine, tmp_path)
        print("✅ Audio modulation pipeline test passed!")
