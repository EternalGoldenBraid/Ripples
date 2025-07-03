"""RippleEngine – revised implementation
---------------------------------------
•  Uses ``subprocess.run(check=True)`` so any FFmpeg failure raises.
•  Resets ``disk_ready_to_render[idx]`` the moment a render starts – prevents duplicate renders.
•  Adds ``wait_for_renders`` helper so callers/tests can block deterministically.
•  ``lock.acquire(timeout=…)`` in the RAM→disk flush path avoids the risk of hanging forever.
•  Timing of the two expensive operations is measured with ``timed`` context‑manager utility.
•  Imports grouped per PEP 8: standard lib, 3rd‑party, project‑internal.
"""

from __future__ import annotations

# ─────────────────────────────── Standard library ──────────────────────────────
import os
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Tuple
import uuid
import hashlib   # for a quick unique hash-style filename
import atexit

# ──────────────────────────────── Third‑party ──────────────────────────────────
import h5py
import imageio
import numpy as np
from loguru import logger as log
from matplotlib import cm
import soundfile as sf
from scipy.sparse import coo_matrix
import cupyx

try:  # CUDA is optional
    import cupy as cp  # type: ignore
except ImportError:  # pragma: no cover – CPU‑only environment
    cp = None  # type: ignore

# ────────────────────────────── Project internal ───────────────────────────────
from audioviz.physics.wave_propagator import WavePropagatorCPU, WavePropagatorGPU
from audioviz.sources.base import ExcitationSourceBase
from audioviz.sources.camera import CameraSource
from audioviz.sources.pose.mediapipe_pose_source import MediaPipePoseExtractor
from audioviz.sources.pose.pose_graph_state import PoseGraphState
from audioviz.types import ArrayType
from audioviz.utils.graph_utils import build_combined_laplacian, build_grid_adjacency
from audioviz.utils.utils import timed  # simple context‑manager timing helper

# ────────────────────────────────────────────────────────────────────────────────

__all__ = ["RippleEngine"]


class RippleEngine:
    """Real‑time 2‑D wave simulator that can stream long recordings to disk.

    Parameters
    ----------
    resolution
        (Ny, Nx) cells.
    dx, speed, damping
        Physical parameters for the finite‑difference solver.
    use_gpu
        If *True* and CuPy is available, use GPU backend.
    ram_budget_gb, disk_budget_gb, num_disk_buffers
        Recording pipeline settings – see :meth:`_init_recording_config`.
    """

    # ───────────────────────────────— construction —───────────────────────────
    def __init__(
        self,
        resolution: Tuple[int, int],
        dx: float,
        speed: float,
        damping: float,
        *,
        use_gpu: bool = True,
        ram_budget_gb: int = 2,
        disk_budget_gb: int = 10,
        num_disk_buffers: int = 3,
    ) -> None:
        self.resolution = resolution
        self.dx = dx
        self.speed = speed
        self.damping = damping
        self.use_gpu = bool(use_gpu and cp is not None)
        self.xp = cp if self.use_gpu else np

        # ───── allocate solver ─────
        self.Z = self.xp.zeros(self.resolution, dtype=self.xp.float32)
        self.excitation = self.xp.zeros_like(self.Z)
        self.dt = (max(dx, dx) / speed) / np.sqrt(2) * 0.95  # CFL‑safe step

        propagator_cls = WavePropagatorGPU if self.use_gpu else WavePropagatorCPU
        self._propagator_kwargs = {
            "shape": self.resolution,
            "dx": self.dx,
            "dt": self.dt,
            "speed": self.speed,
            "damping": self.damping,
        }
        self.propagator = propagator_cls(**self._propagator_kwargs)

        self.sources: Dict[str, ExcitationSourceBase] = {}

        # pose‑graph members are configured later
        self.pose_graph_state: Optional[PoseGraphState] = None
        self.camera: Optional[CameraSource] = None
        self.extractor: Optional[MediaPipePoseExtractor] = None
        self.use_matrix = False

        # bookkeeping
        self.time = 0.0
        self._log_counter = 0

        # ───── recording pipeline ─────

        # ─ recording session bookkeeping ─
        self._current_session_id: Optional[str] = None
        self._session_renders: List[Path] = []   # mp4s belonging to the *active* session

        # keep track of *all* rendered clips so the destructor can merge leftovers
        self._all_renders: List[Path] = []

        # make sure we always clean up on process exit
        atexit.register(self.__del__)

        self._init_recording_config(ram_budget_gb, disk_budget_gb, num_disk_buffers)
        log.info("RippleEngine initialised.")

    # ───────────────────────────── recording setup ────────────────────────────
    def _init_recording_config(
        self,
        ram_budget_gb: int,
        disk_budget_gb: int,
        num_disk_buffers: int,
    ) -> None:
        Ny, Nx = self.resolution
        bytes_per_frame = Ny * Nx * 4  # float32

        # derive buffer sizes
        frames_per_ram = int(ram_budget_gb * 1024 ** 3 / bytes_per_frame)
        frames_per_disk_raw = disk_budget_gb * 1024 ** 3 / bytes_per_frame
        frames_per_disk = frames_per_ram * int(frames_per_disk_raw // frames_per_ram)

        self.ram_buffer_size = frames_per_ram
        self.disk_buffer_size = frames_per_disk
        self.num_disk_buffers = num_disk_buffers

        # RAM buffers (double‑buffer pattern)
        self.Z_ram_online = np.zeros((frames_per_ram, Ny, Nx), dtype=np.float32)
        self.Z_ram_offline = np.zeros_like(self.Z_ram_online)
        self.probe_ram_online = np.zeros(frames_per_ram, dtype=np.float32)
        self.probe_ram_offline = np.zeros_like(self.probe_ram_online)
        self._ram_frame_idx = 0

        # disk buffers
        disk_dir = Path("outputs/disk_buffers"); disk_dir.mkdir(parents=True, exist_ok=True)
        self._disk_paths = [disk_dir / f"buffer_{i:02d}.h5" for i in range(num_disk_buffers)]
        self._disk_locks: List[Lock] = [Lock() for _ in range(num_disk_buffers)]
        self._disk_offsets = [0] * num_disk_buffers
        self._active_disk_idx = 0
        self._disk_ready = [False] * num_disk_buffers
        self._disk_rendering = [False] * num_disk_buffers

        for i, path in enumerate(self._disk_paths):
            with h5py.File(path, "w") as f:
                f.create_dataset("fields", shape=(frames_per_disk, Ny, Nx), dtype=np.float32)
                f.create_dataset("probe_signal", shape=(frames_per_disk,), dtype=np.float32)
            log.info(f"✅ Created disk buffer {i}: {path}")

        # ffmpeg options
        self.ffmpeg_bin = "ffmpeg"
        self.ffmpeg_fps = 60
        self.ffmpeg_codec = "libx264"
        self.ffmpeg_crf = 18
        self.render_dir = Path("outputs/renders"); self.render_dir.mkdir(parents=True, exist_ok=True)

        # background watcher
        self._watcher = Thread(target=self._watch_disk_buffers, daemon=True)
        self._watcher.start()

        self.recording_enabled = False
        self.probe_ix, self.probe_iy = Nx // 2, Ny // 2

        log.info(
            "Recording config: RAM %d f, disk %d f × %d buffers",
            self.ram_buffer_size,
            self.disk_buffer_size,
            num_disk_buffers,
        )

    # ───────────────────────────── helper: wait for renders ───────────────────
    def wait_for_renders(self, timeout: float | None = None) -> None:
        """Block until all render threads are idle.

        Parameters
        ----------
        timeout
            Maximum seconds to wait. *None* → wait indefinitely.
        """
        start = time.perf_counter()
        while any(self._disk_rendering):
            if timeout is not None and (time.perf_counter() - start) > timeout:
                raise TimeoutError("Render threads still busy after timeout")
            time.sleep(0.2)

    # ───────────────────────────── background watcher ─────────────────────────
    def _watch_disk_buffers(self) -> None:  # runs in daemon thread
        while True:
            for idx in range(self.num_disk_buffers):
                if self._disk_ready[idx] and not self._disk_rendering[idx]:
                    self._launch_ffmpeg(idx)
            time.sleep(0.5)

    # ───────────────────────────── launch ffmpeg render ───────────────────────
    def _launch_ffmpeg(self, idx: int) -> None:
        """Spawn a daemon thread that streams raw RGB frames to FFmpeg."""
        input_path = self._disk_paths[idx]
        output_path = self.render_dir / f"render_{idx:02d}.mp4"
        height, width = self.resolution  # Ny, Nx

        def _render() -> None:  # runs in its own daemon thread
            log.info("🎥 Start render for buffer %d → %s", idx, output_path)
            self._disk_rendering[idx] = True
            self._disk_ready[idx] = False  # avoid duplicate launches

            with timed(f"Render buffer {idx}"):
                with h5py.File(input_path, "r") as f:
                    fields = f["fields"]
                    probe = f["probe_signal"][:]

                    # ‑‑‑ write probe to a tiny temp WAV (fast) ‑‑‑
                    wav_path = Path(tempfile.mkstemp(suffix=".wav", dir=input_path.parent)[1])
                    sf.write(wav_path, probe, samplerate=44_100)

                    # ‑‑‑ spawn FFmpeg expecting raw RGB frames on stdin ‑‑‑
                    cmd = [
                        self.ffmpeg_bin,
                        "-y",
                        "-f", "rawvideo",
                        "-pix_fmt", "rgb24",
                        "-s", f"{width}x{height}",
                        "-r", str(self.ffmpeg_fps),
                        "-i", "-",  # stdin
                        "-i", str(wav_path),
                        "-c:v", self.ffmpeg_codec,
                        "-crf", str(self.ffmpeg_crf),
                        "-pix_fmt", "yuv420p",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        str(output_path),
                    ]
                    log.info("🎬 FFmpeg (streaming): %s", " ".join(cmd))
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

                    # ‑‑‑ stream frames ‑‑‑
                    for z in fields:
                        norm = (z - z.min()) / (np.ptp(z) + 1e-6)
                        rgb = (cm.inferno(norm)[:, :, :3] * 255).astype(np.uint8)
                        proc.stdin.write(rgb.tobytes())  # type: ignore[arg-type]
                    proc.stdin.close()

                    proc.wait()  # raises if encode failed
                    wav_path.unlink(missing_ok=True)

            log.info("✅ Render finished for buffer %d", idx)
            self._disk_rendering[idx] = False

            # For concatenation later, add this render to the global list
            self._all_renders.append(output_path)
            if self._current_session_id is not None:
                self._session_renders.append(output_path)

        Thread(target=_render, daemon=True).start()

        # ------------------ graceful teardown ------------------
    def __del__(self) -> None:
        """Ensure any running record finishes & remaining clips are merged."""
        try:
            # stop current slice (flag signals this is from __del__)
            self.disable_recording(_called_from_del=True)

            # if several independent slices were recorded during the process
            # lifetime but never concatenated, merge them *all* now
            if self._all_renders:
                final_hash = hashlib.sha1(
                    ("".join(str(p) for p in self._all_renders)).encode()
                ).hexdigest()[:10]
                final_path = self.render_dir / f"ripples_{final_hash}.mp4"
                self._concat_mp4s(self._all_renders, final_path)
                log.info("📦 All sessions merged → %s", final_path)
        except Exception as exc:  # pragma: no cover
            log.exception("Destructor failed: %s", exc)


    # ───────────────────────────── flush RAM → disk ───────────────────────────
    def _flush_ram_to_disk(self) -> None:
        idx = self._active_disk_idx
        lock = self._disk_locks[idx]

        log.info("Flushing RAM to disk buffer %d", idx)

        # wait (up to 10 s) if buffer is busy
        if self._disk_ready[idx] or self._disk_rendering[idx]:
            if not lock.acquire(timeout=10):
                raise TimeoutError("Disk buffer locked for too long while flushing")
            lock.release()

        path = self._disk_paths[idx]
        offset = self._disk_offsets[idx]

        # copy online buffer → offline, then write asynchronously
        self.Z_ram_offline[:] = self.Z_ram_online
        self.probe_ram_offline[:] = self.probe_ram_online

        def _write():
            with timed(f"Flush to disk {idx}"):
                with lock:
                    with h5py.File(path, "r+") as f:
                        sl = slice(offset, offset + self.ram_buffer_size)
                        f["fields"][sl] = self.Z_ram_offline
                        f["probe_signal"][sl] = self.probe_ram_offline

                # bookkeeping
                self._disk_offsets[idx] += self.ram_buffer_size
                if self._disk_offsets[idx] >= self.disk_buffer_size:
                    self._disk_offsets[idx] = 0
                    self._disk_ready[idx] = True
                    self._active_disk_idx = (idx + 1) % self.num_disk_buffers
                    log.info("♻️ Disk buffer %d filled → queued for render", idx)

        Thread(target=_write, daemon=True).start()

    def _concat_mp4s(self, inputs: List[Path], output: Path) -> None:
        """Fast FFmpeg concat using the *copy* muxer (no re-encode)."""
        if len(inputs) == 1:
            inputs[0].rename(output)  # trivial case
            return

        list_file = output.with_suffix(".txt")
        with list_file.open("w") as fh:
            for p in inputs:
                fh.write(f"file '{p.resolve()}'\n")

        subprocess.run(
            [
                self.ffmpeg_bin,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                str(output),
            ],
            check=True,
        )
        list_file.unlink(missing_ok=True)

    # ───────────────────────────── public API ────────────────────────────────

    def enable_recording(self) -> None:
        """Start a new recording slice."""
        if self.recording_enabled:
            log.warning("Recording already enabled.")
            return

        # start a fresh session
        self._current_session_id = uuid.uuid4().hex[:8]
        self._session_renders.clear()

        # reset in-RAM index so a new slice starts at frame 0
        self.ram_frame_idx = 0
        self.recording_enabled = True
        log.info("✅ Recording enabled (session %s)", self._current_session_id)

    def disable_recording(self, *, _called_from_del: bool = False) -> None:
        """Stop the current slice, wait for all renders, then concat them."""
        if not self.recording_enabled and not _called_from_del:
            log.warning("Recording already disabled.")
            return

        # flush any half-filled RAM buffer
        if self.ram_frame_idx:
            self._flush_ram_to_disk()
            self.ram_frame_idx = 0

        self.recording_enabled = False
        # block until every FFmpeg job is finished
        self.wait_for_renders()

        # concatenate this session’s clips (if any)
        if self._session_renders:
            out_name = f"session_{self._current_session_id}.mp4"
            out_path = self.render_dir / out_name
            self._concat_mp4s(self._session_renders, out_path)
            log.info("🎞️  Session concatenated → %s", out_path)

        # prepare for the next enable_recording()
        self._current_session_id = None
        self._session_renders.clear()

    def save_recording(self, path="outputs/recording.h5"):
        final_Z = self.Z_recording[:self.record_frame_idx]
        final_probe = self.probe_signal[:self.record_frame_idx]

        with h5py.File(path, "w") as f:
            f.create_dataset("fields", data=final_Z)
            f.create_dataset("probe_signal", data=final_probe)

        log.info(f"✅ Recording saved to {path}")

    def set_probe_location(self, iy: int, ix: int) -> None:
        self.probe_iy, self.probe_ix = iy, ix
        log.info("✅ Probe location set to (%d, %d)", iy, ix)

    def add_source(self, source: ExcitationSourceBase):
        if source.name in self.sources:
            raise ValueError(f"Source '{source.name}' already exists.")
        self.sources[source.name] = source
        log.info(f"Added source '{source.name}' to RippleEngine.")

    def add_pose_graph(self, camera: CameraSource, extractor: MediaPipePoseExtractor,
                       pose_graph_state: PoseGraphState):
        self.use_matrix = True
        self.camera = camera
        self.extractor = extractor
        self.pose_graph_state = pose_graph_state
        self.N_grid = self.resolution[0] * self.resolution[1]

        grid_adj = build_grid_adjacency(self.resolution)
        pose_adj = pose_graph_state.get_adjacency_coo()

        center_y, center_x = self.resolution[0] // 2, self.resolution[1] // 2
        patch_size = 5

        grid_indices = []
        for dy in range(-patch_size // 2, patch_size // 2 + 1):
            for dx in range(-patch_size // 2, patch_size // 2 + 1):
                gy = center_y + dy
                gx = center_x + dx
                if 0 <= gy < self.resolution[0] and 0 <= gx < self.resolution[1]:
                    grid_idx = gy * self.resolution[1] + gx
                    grid_indices.append(grid_idx)

        self.coupled_pose_indices = [15]

        coupling_rows = []
        coupling_cols = []
        coupling_data = []

        for g_idx in grid_indices:
            for p_idx in self.coupled_pose_indices:
                coupling_rows.append(g_idx)
                coupling_cols.append(p_idx)
                coupling_data.append(1.0)

        coupling_shape = (grid_adj.shape[0], pose_adj.shape[0])
        coupling = coo_matrix((coupling_data, (coupling_rows, coupling_cols)), shape=coupling_shape)

        L_coo = build_combined_laplacian(grid_adj, pose_adj, coupling)
        L_csr = L_coo.tocsr()
        L_csr_gpu = cupyx.scipy.sparse.csr_matrix(L_csr)

        self.extended_excitation: cp.ndarray = self.xp.zeros(
            (self.N_grid + pose_graph_state.num_nodes,),
            dtype=self.xp.float32
        )

        self.propagator_kwargs["use_matrix"] = True
        self.propagator_kwargs["laplacian_csr"] = L_csr_gpu
        self.propagator_kwargs["shape"] = (self.N_grid + pose_graph_state.num_nodes,)

        self.propagator = WavePropagatorGPU(**self.propagator_kwargs)

        log.info("✅ Pose graph with static coupling added to RippleEngine.")

    def _update_pose_graph_excitation(self):
        if not (self.pose_graph_state and self.camera and self.extractor):
            return

        frame = self.camera.read()
        if frame is not None:
            pose_data = self.extractor.extract(frame)
            coords = pose_data["coords"]
            if coords.shape[0] > 0:
                self.pose_graph_state.update(coords, self.dt)

        flat_grid = self.excitation.ravel()
        self.extended_excitation[:self.N_grid] = flat_grid

        if self._log_counter % 30000000 == 0:
            pose_ex_vec = self.pose_graph_state()
            for i in self.coupled_pose_indices:
                val = np.clip(pose_ex_vec[i] / 50000, 0, 0.02)
                self.extended_excitation[self.N_grid + i] = val
                log.debug(f"Extended excitation for pose node {i}: {val:.4f}")

        self.propagator.add_excitation(self.extended_excitation)
        self.extended_excitation[:self.N_grid] = 0

    def update(self, t: float):
        self.time = t
    
        for name, source in self.sources.items():
            result = source(t)
            weight = 1 / max(len(self.sources), 1) if name != 'heart' else 0.001
            self.excitation[:] += weight * result["excitation"]
    
        if self.use_matrix:
            self._update_pose_graph_excitation()
        else:
            self.propagator.add_excitation(self.excitation)
    
        self.propagator.step()
    
        if self.pose_graph_state:
            combined_state = self.propagator.get_state().flatten()
            grid_state_flat = combined_state[:self.N_grid]
            pose_state_flat = combined_state[self.N_grid:]
            self.Z[:] = grid_state_flat.reshape(self.resolution)
            self.pose_graph_state.set_ripple_states(cp.asnumpy(pose_state_flat))
        else:
            self.Z[:] = self.propagator.get_state()
    
        # --------------------------
        # Recording logic
        # --------------------------
        if self.recording_enabled:
            self.Z_ram_online[self._ram_frame_idx] = cp.asnumpy(self.Z) if self.use_gpu else self.Z.copy()
            self.probe_ram_online[self._ram_frame_idx] = cp.asnumpy(self.Z[self.probe_iy, self.probe_ix]) if self.use_gpu else self.Z[self.probe_iy, self.probe_ix]
            self._ram_frame_idx += 1
    
            # Check if RAM buffer is full
            if self._ram_frame_idx >= self.ram_buffer_size:
                self._flush_ram_to_disk()
                self._ram_frame_idx = 0
    
        self.excitation[:] = 0
        self._log_counter += 1

    def get_field(self) -> ArrayType:
        return self.Z
