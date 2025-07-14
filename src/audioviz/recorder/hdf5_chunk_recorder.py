from typing import Tuple, Optional
import uuid
import time
import subprocess
from pathlib import Path
from threading import Thread, Lock


import numpy as np
import h5py
import soundfile as sf
from loguru import logger as log
import matplotlib.cm as cm


from .base import RecorderBase


class Hdf5ChunkRecorder(RecorderBase):
    """Exactly the old behaviour, now boxed in a class."""

    # ‑‑ construction -------------------------------------------------------
    def __init__(
        self,
        resolution: Tuple[int, int],
        samples_per_frame: int,
        *,
        ram_budget_gb: float = 1.5,
        disk_budget_gb: float = 10,
        num_disk_buffers: int = 4,
        ffmpeg_bin: str = "ffmpeg",
        ffmpeg_codec: str = "libx264",
        ffmpeg_crf: int = 18,
        render_dir: Path | None = None,
    ) -> None:
        self.resolution = resolution
        self.samples_per_frame = samples_per_frame
        self.ram_budget_gb = ram_budget_gb
        self.disk_budget_gb = disk_budget_gb
        self.num_disk_buffers = num_disk_buffers

        self.ffmpeg_bin = ffmpeg_bin
        self.ffmpeg_codec = ffmpeg_codec
        self.ffmpeg_crf = ffmpeg_crf

        self.render_dir = Path(render_dir or "outputs/renders")
        self.render_dir.mkdir(parents=True, exist_ok=True)

        # runtime state
        self._fps: int | None = None
        self._session_id: str | None = None
        self._all_renders: list[Path] = []

        # HDF5 buffer bookkeeping
        self._init_buffers()

        # watcher that launches FFmpeg when a file is ready
        self._watcher = Thread(target=self._watch_disk_buffers, daemon=True)
        self._watcher.start()

    # ‑‑ public API ---------------------------------------------------------
    def start(self, fps: int) -> None:
        if self._session_id is not None:
            log.warning("Recorder already running; ignoring duplicate start()")
            return
        self._fps = fps
        self._session_id = uuid.uuid4().hex[:8]
        self._session_renders: list[Path] = []
        self._ram_frame_idx = 0
        log.info("🎙️  HDF5 recorder started (session %s)", self._session_id)

    def stop(self) -> None:
        if self._session_id is None:
            return
        log.info("🛑 Stopping HDF5 recorder…")

        # flush partial RAM buffer
        if self._ram_frame_idx:
            self._flush_n_frames(self._ram_frame_idx)
            self._ram_frame_idx = 0

        # mark all non‑empty disk buffers as ready
        for i in range(self.num_disk_buffers):
            if self._disk_offsets[i] > 0:
                self._mark_buffer_ready(i)

        self._wait_for_renders()
        self._concat_session_mp4()
        self._session_id = None

    # called from GUI thread
    def feed_video(self, rgb_frame: np.ndarray) -> None:  # noqa: D401
        H, W, _ = rgb_frame.shape
        assert (H, W) == self.resolution, "resolution mismatch"
        if self._ram_frame_idx >= self.ram_buffer_size:
            self._flush_full_ram_buffer()
        self.Z_ram_on[self._ram_frame_idx] = rgb_frame.astype(np.float32) / 255.0
        self._ram_frame_idx += 1

    # called from audio callback
    def feed_audio(self, pcm_block: np.ndarray) -> None:  # noqa: D401
        if self._ram_frame_idx == 0:  # no video yet – skip block to keep sync
            return
        idx = self._ram_frame_idx - 1
        self.audio_ram_on[idx] = pcm_block.astype(np.float32)

    # ‑‑ internal: buffer initialisation -----------------------------------
    def _init_buffers(self):
        Ny, Nx = self.resolution
        bytes_per_frame = Ny * Nx * 4
        frames_ram = int(self.ram_budget_gb * 1024 ** 3 // bytes_per_frame)
        frames_ram = max(1, frames_ram)
        frames_disk_raw = self.disk_budget_gb * 1024 ** 3 / bytes_per_frame
        frames_disk = frames_ram * max(1, int(frames_disk_raw // frames_ram))

        self.ram_buffer_size = frames_ram
        self.disk_buffer_size = frames_disk

        # RAM double buffers
        self.Z_ram_on = np.zeros((frames_ram, Ny, Nx), np.float32)
        self.audio_ram_on = np.zeros((frames_ram, self.samples_per_frame), np.float32)
        self.Z_ram_off = np.zeros_like(self.Z_ram_on)
        self.audio_ram_off = np.zeros_like(self.audio_ram_on)

        self._ram_frame_idx = 0

        # disk buffers
        disk_dir = Path("outputs/disk_buffers")
        disk_dir.mkdir(parents=True, exist_ok=True)
        self._disk_paths = [disk_dir / f"buffer_{i:02d}.h5" for i in range(self.num_disk_buffers)]
        self._disk_locks: list[Lock] = [Lock() for _ in range(self.num_disk_buffers)]
        self._disk_offsets = [0] * self.num_disk_buffers
        self._active_disk_idx = 0
        self._disk_ready = [False] * self.num_disk_buffers
        self._disk_rendering = [False] * self.num_disk_buffers

        Ny, Nx = self.resolution
        for p in self._disk_paths:
            with h5py.File(p, "w") as f:
                f.create_dataset("fields", (frames_disk, Ny, Nx), np.float32)
                f.create_dataset("audio", (frames_disk, self.samples_per_frame), np.float32)

    # ‑‑ internal: flushing helpers ----------------------------------------
    def _flush_full_ram_buffer(self):
        """Write the *whole* on‑buffer to disk in a background thread."""
        if self.ram_buffer_size == 0:
            return
        self.Z_ram_off[:] = self.Z_ram_on
        self.audio_ram_off[:] = self.audio_ram_on
        t = Thread(target=self._write_buffer, daemon=True)
        t.start()

    def _flush_n_frames(self, n: int):
        if n == 0:
            return
        self.Z_ram_off[:n] = self.Z_ram_on[:n]
        self.audio_ram_off[:n] = self.audio_ram_on[:n]
        self._write_buffer(n_frames=n)

    def _write_buffer(self, n_frames: Optional[int] = None):
        i = self._active_disk_idx
        lock = self._disk_locks[i]
        off = self._disk_offsets[i]
        n_frames = n_frames or self.ram_buffer_size
        with lock, h5py.File(self._disk_paths[i], "r+") as f:
            sl = slice(off, off + n_frames)
            f["fields"][sl] = self.Z_ram_off[:n_frames]
            f["audio"][sl] = self.audio_ram_off[:n_frames]
            self._disk_offsets[i] += n_frames
            if self._disk_offsets[i] >= self.disk_buffer_size:
                self._disk_offsets[i] = 0
                self._active_disk_idx = (i + 1) % self.num_disk_buffers
                self._disk_ready[i] = True  # watcher will render
                log.info("♻️  Disk buffer %d full → queued", i)

    # ‑‑ internal: render watcher -----------------------------------------
    def _watch_disk_buffers(self):
        while True:
            for i in range(self.num_disk_buffers):
                if self._disk_ready[i] and not self._disk_rendering[i]:
                    self._render_buffer(i)
            time.sleep(0.2)

    def _render_buffer(self, idx: int):
        def _render():
            in_path = self._disk_paths[idx]
            out_mp4 = self.render_dir / f"render_{idx:02d}.mp4"
            H, W = self.resolution
            with h5py.File(in_path, "r") as f:
                fields = f["fields"]
                audio = f["audio"][:].reshape(-1)
                wav_path = out_mp4.with_suffix(".wav")
                sf.write(wav_path, audio, 44100)

                cmd = [
                    self.ffmpeg_bin, "-y",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{W}x{H}", "-r", str(self._fps), "-i", "-",
                    "-i", str(wav_path),
                    "-c:v", self.ffmpeg_codec, "-crf", str(self.ffmpeg_crf),
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    str(out_mp4),
                ]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                for z in fields:
                    norm = (z - z.min()) / (np.ptp(z) + 1e-6)
                    rgb = (cm.inferno(norm)[..., :3] * 255).astype(np.uint8)
                    proc.stdin.write(rgb.tobytes())
                proc.stdin.close()
                proc.wait()
                wav_path.unlink(missing_ok=True)
            self._disk_rendering[idx] = False
            self._disk_ready[idx] = False
            self._all_renders.append(out_mp4)
            self._session_renders.append(out_mp4)

        self._disk_rendering[idx] = True
        Thread(target=_render, daemon=True).start()

    # ‑‑ internal: util -----------------------------------------------------
    def _mark_buffer_ready(self, idx: int):
        with self._disk_locks[idx]:
            self._disk_ready[idx] = True

    def _wait_for_renders(self):
        while any(self._disk_rendering):
            time.sleep(0.2)

    def _concat_session_mp4(self):
        if not self._session_renders:
            return
        out = self.render_dir / f"session_{self._session_id}.mp4"
        if len(self._session_renders) == 1:
            self._session_renders[0].rename(out)
            return
        lst = out.with_suffix(".txt")
        lst.write_text("".join(f"file '{p.resolve()}'\n" for p in self._session_renders))
        subprocess.run([
            self.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
            "-i", str(lst), "-c:v", self.ffmpeg_codec, "-crf", str(self.ffmpeg_crf),
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(out)
        ], check=True)
        lst.unlink()
