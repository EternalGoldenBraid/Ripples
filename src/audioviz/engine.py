"""
RippleEngine – wave-field + audio recorder
──────────────────────────────────────────
•  Writes one row per *visual* frame to *all* datasets
   ──> perfect sync between grid, probe-sample and 44.1 kHz PCM.
•  Constant 735 samples per frame  (44 100 / 60).
•  Adds ``audio`` dataset to every HDF5 buffer:  shape (N_frames, 735).
•  RAM double-buffer now stores Z, probe and audio together.
•  Renderer rebuilds the raw track from the HDF5 slice – no more
   “raw audio missing”.
•  Otherwise identical public API / logging behaviour.
"""

from __future__ import annotations

# ──────────────────────────────── std-lib ────────────────────────────────────
import atexit
import hashlib
import subprocess
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Tuple
import uuid

# ────────────────────────────── third-party ──────────────────────────────────
import h5py
import numpy as np
from loguru import logger as log
from matplotlib import cm
from scipy.signal import resample_poly
import soundfile as sf

try:
    import cupy as cp  # type: ignore
except ImportError:                           # CPU-only
    cp = None  # type: ignore

# ───────────────────────────── project internal ──────────────────────────────
from audioviz.physics.wave_propagator import WavePropagatorCPU, WavePropagatorGPU
from audioviz.sources.base import ExcitationSourceBase
from audioviz.sources.camera import CameraSource
from audioviz.sources.pose.mediapipe_pose_source import MediaPipePoseExtractor
from audioviz.sources.pose.pose_graph_state import PoseGraphState
from audioviz.types import ArrayType
from audioviz.utils.graph_utils import build_combined_laplacian, build_grid_adjacency
from audioviz.utils.utils import timed
from audioviz.recorder import RecorderBase, Hdf5ChunkRecorder

__all__ = ["RippleEngine"]


class RippleEngine_old:
    # ─────────────────────────────── construction ───────────────────────────
    def __init__(
        self,
        resolution: Tuple[int, int],
        dx: float,
        speed: float,
        damping: float,
        *,
        use_gpu: bool = True,
        ram_budget_gb: float = 1.0,
        disk_budget_gb: float = 4,
        num_disk_buffers: int = 4,
    ) -> None:
        # ───── physics ─────
        self.resolution = resolution
        self.dx = dx
        self.speed = speed
        self.damping = damping
        self.use_gpu = bool(use_gpu and cp is not None)
        self.xp = cp if self.use_gpu else np
        self.time: float = 0.0
        self._log_counter: int = 0
        self._excitation_results: Dict[str, Dict[str, ArrayType]] = {}

        self.Z = self.xp.zeros(resolution, dtype=self.xp.float32)
        self.excitation = self.xp.zeros_like(self.Z)
        self.dt = (max(dx, dx) / speed) / np.sqrt(2) * 0.95

        Prop = WavePropagatorGPU if self.use_gpu else WavePropagatorCPU
        self._propagator_kwargs = dict(
            shape=resolution, dx=dx, dt=self.dt, speed=speed, damping=damping
        )
        self.propagator = Prop(**self._propagator_kwargs)
        self.max_frequency = self.speed / (2 * max(self.dx, self.dx))

        self.sources: Dict[str, ExcitationSourceBase] = {}
        self.pose_graph_state: Optional[PoseGraphState] = None
        self.camera: Optional[CameraSource] = None
        self.extractor: Optional[MediaPipePoseExtractor] = None
        self.use_matrix = False

        # ───── audio constants ─────
        self.audio_sr = 44_100
        self.vis_fps = 60
        self.samples_per_frame = self.audio_sr // self.vis_fps  # 735

        # ring-buffer for incoming PCM (one minute)
        self._audio_ring = np.zeros(self.audio_sr * 60, dtype=np.float32)
        self._audio_wptr = 0      # write index
        self._audio_rptr = 0      # read  index (only used during recording)

        # ───── recording bookkeeping ─────
        self._current_session_id: Optional[str] = None
        self._session_renders: List[Path] = []
        self._session_raw_wav: Optional[Path] = None
        self._all_renders: List[Path] = []

        atexit.register(self.__del__)

        self.recording_enabled = False
        self._init_recording_config(ram_budget_gb, disk_budget_gb, num_disk_buffers)
        log.info("RippleEngine initialised.")

    # ───────────────────────────── recording config ──────────────────────────
    def _init_recording_config(
        self, ram_budget_gb: int, disk_budget_gb: int, num_disk_buffers: int
    ) -> None:
        Ny, Nx = self.resolution
        bytes_per_frame = Ny * Nx * 4
        frames_ram = int(ram_budget_gb * 1024**3 / bytes_per_frame)
        frames_disk_raw = disk_budget_gb * 1024**3 / bytes_per_frame
        frames_disk = frames_ram * int(frames_disk_raw // frames_ram)
        assert frames_disk > 0, "disk_buffer_size ended up zero – check RAM/DISK budgets"

        self.ram_buffer_size = frames_ram
        self.disk_buffer_size = frames_disk
        self.num_disk_buffers = num_disk_buffers

        # ─ RAM double buffers ─
        self.Z_ram_on = np.zeros((frames_ram, Ny, Nx), np.float32)
        self.Z_ram_off = np.zeros_like(self.Z_ram_on)

        self.probe_ram_on = np.zeros(frames_ram, np.float32)
        self.probe_ram_off = np.zeros_like(self.probe_ram_on)

        self.audio_ram_on = np.zeros((frames_ram, self.samples_per_frame), np.float32)
        self.audio_ram_off = np.zeros_like(self.audio_ram_on)

        self._ram_frame_idx = 0

        # ─ disk buffers ─
        disk_dir = Path("outputs/disk_buffers"); disk_dir.mkdir(parents=True, exist_ok=True)
        self._disk_paths = [disk_dir / f"buffer_{i:02d}.h5" for i in range(num_disk_buffers)]
        self._disk_locks: List[Lock] = [Lock() for _ in range(num_disk_buffers)]
        self._disk_offsets = [0] * num_disk_buffers
        self._active_disk_idx = 0
        self._disk_ready = [False] * num_disk_buffers
        self._disk_rendering = [False] * num_disk_buffers

        for i, p in enumerate(self._disk_paths):
            with h5py.File(p, "w") as f:
                f.create_dataset("fields", (frames_disk, Ny, Nx), np.float32)
                f.create_dataset("probe_signal", (frames_disk,), np.float32)
                f.create_dataset("audio", (frames_disk, self.samples_per_frame), np.float32)
            log.info(f"✅ Created disk buffer {i}: {p}")

        # ffmpeg / render
        self.ffmpeg_bin = "ffmpeg"
        self.ffmpeg_codec = "libx264"
        self.ffmpeg_crf = 18
        self.render_dir = Path("outputs/renders"); self.render_dir.mkdir(parents=True, exist_ok=True)

        self.recording_enabled = False
        self.probe_ix, self.probe_iy = Nx // 2, Ny // 2

        self._watcher = Thread(target=self._watch_disk_buffers, daemon=True)
        self._watcher.start()

        log.info(
            "Recording config: RAM %d f, disk %d f × %d buffers",
            self.ram_buffer_size,
            self.disk_buffer_size,
            self.num_disk_buffers,
        )

    # ───────────────────────────── audio I/O ─────────────────────────────────
    def feed_audio(self, pcm: np.ndarray) -> None:
        """Append raw mono *pcm* (float32 -1..1) to the ring buffer."""
        n = len(pcm)
        ring = self._audio_ring
        idx = self._audio_wptr % ring.size
        end = idx + n
        if end <= ring.size:
            ring[idx:end] = pcm
        else:                       # wrap-around
            first = ring.size - idx
            ring[idx:] = pcm[:first]
            ring[: n - first] = pcm[first:]
        self._audio_wptr += n

    def _pop_audio_frame(self) -> np.ndarray:
        """Return exactly *samples_per_frame* fresh samples (zero-pad if under-flow)."""
        need = self.samples_per_frame
        ring = self._audio_ring
        out = np.zeros(need, dtype=np.float32)

        available = self._audio_wptr - self._audio_rptr
        take = min(need, available)
        if take:
            r_idx = self._audio_rptr % ring.size
            r_end = r_idx + take
            if r_end <= ring.size:
                out[:take] = ring[r_idx:r_end]
            else:  # wrap-around
                first = ring.size - r_idx
                out[:first] = ring[r_idx:]
                out[first:take] = ring[: r_end - ring.size]
            self._audio_rptr += take
        # remaining samples (if any) stay zeros
        return out

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
            log.debug(f"Disk buffers busy: {self._disk_rendering}")
            if timeout is not None and (time.perf_counter() - start) > timeout:
                raise TimeoutError("Render threads still busy after timeout")
            time.sleep(0.2)

    # ───────────────────────────── helper threads ────────────────────────────

    def _mark_buffer_ready(self, idx: int):
        lock = self._disk_locks[idx]
        with lock:
            self._disk_ready[idx] = True

    def _watch_disk_buffers(self) -> None:
        while True:
            for i in range(self.num_disk_buffers):
                lock = self._disk_locks[i]
                if self._disk_ready[i] and not self._disk_rendering[i]:
                    self._launch_ffmpeg(i)
            time.sleep(0.5)

    def _launch_ffmpeg(self, idx: int) -> None:
        """Render one filled disk buffer → mp4 (+ wav)."""
        log.debug("Launching FFmpeg for buffer %d", idx)

        log.debug(f"Updating flag _disk_rendering[{idx}] to True")
        log.debug(f"Updating flag _disk_ready[{idx}] to False")
        lock = self._disk_locks[idx]
        with lock:
            self._disk_rendering[idx] = True
            self._disk_ready[idx] = False

        in_path = self._disk_paths[idx]
        out_mp4 = self.render_dir / f"render_{idx:02d}.mp4"
        H, W = self.resolution

        log.debug("Flags updated, starting render thread for buffer %d", idx)

        def _render() -> None:
            """
            Render without reading the disk buffer into RAM.
            """
            log.info(f"🎥 Render buffer {idx} → {out_mp4}")

            with timed(f"Render buffer {idx}"):
                with h5py.File(in_path, "r") as f:
                    # fields = f["fields"][...]
                    audio = f["audio"][...].reshape(-1)  # 1-D PCM
                    fields = f["fields"]

                    wav_path = out_mp4.with_suffix(".wav")
                    sf.write(wav_path, audio, samplerate=self.audio_sr)
                    cmd = [
                        self.ffmpeg_bin, "-y",
                        "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{W}x{H}", "-r", str(self.vis_fps),
                        "-i", "-", "-i", str(wav_path),
                        "-c:v", self.ffmpeg_codec, "-crf", str(self.ffmpeg_crf),
                        "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "192k",
                        str(out_mp4),
                    ]
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

                    # for z in fields:
                    for i in range(fields.shape[0]):
                        z = fields[i, ...]  # get one frame
                        norm = (z - z.min()) / (np.ptp(z) + 1e-6)
                        rgb = (cm.inferno(norm)[:, :, :3] * 255).astype(np.uint8)
                        proc.stdin.write(rgb.tobytes())
                    proc.stdin.close()
                    proc.wait()
                    proc.wait()
                    if proc.returncode:
                        log.error(f"FFmpeg failed {proc.returncode} on buffer {idx}")
                    wav_path.unlink(missing_ok=True)

            self._disk_rendering[idx] = False
            self._all_renders.append(out_mp4)
            if self._current_session_id:
                self._session_renders.append(out_mp4)

        log.info(f"Starting render thread for buffer {idx} → {out_mp4}")
        Thread(target=_render, daemon=True).start()
        log.info("Render thread started for buffer %d", idx)

    # ───────────────────────────── RAM → disk ────────────────────────────────
    def _flush_ram_to_disk(self) -> Thread:
        idx = self._active_disk_idx
        lock = self._disk_locks[idx]
        log.info(f"Flushing RAM → disk buffer {idx}")

        path = self._disk_paths[idx]
        off = self._disk_offsets[idx]

        self.Z_ram_off[:] = self.Z_ram_on
        self.probe_ram_off[:] = self.probe_ram_on
        self.audio_ram_off[:] = self.audio_ram_on

        def _write():
            with timed(f"Flush disk {idx}"):
                with lock, h5py.File(path, "r+") as f:
                    sl = slice(off, off + self.ram_buffer_size)
                    f["fields"][sl] = self.Z_ram_off
                    f["probe_signal"][sl] = self.probe_ram_off
                    f["audio"][sl] = self.audio_ram_off

                    self._disk_offsets[idx] += self.ram_buffer_size
                    if self._disk_offsets[idx] >= self.disk_buffer_size:
                        self._disk_offsets[idx] = 0
                        self._active_disk_idx = (idx + 1) % self.num_disk_buffers
                        self._disk_ready[idx] = True
                        log.info(f"♻️ Disk buffer {idx} full → queued")

        thread: Thread = Thread(target=_write, daemon=True)
        thread.start()
        return thread

    def _flush_n_frames(self, n_frames: int) -> None:
        """Write exactly *n_frames* from the on-buffer to disk (no thread)."""

        if n_frames == 0:
            return                                   # nothing to do

        idx  = self._active_disk_idx
        lock = self._disk_locks[idx]
        off  = self._disk_offsets[idx]
        path = self._disk_paths[idx]

        with timed(f"Flush {n_frames} f → disk {idx}"):
            with lock, h5py.File(path, "r+") as f:
                sl = slice(off, off + n_frames)      # never overruns
                f["fields"][sl]        = self.Z_ram_on[:n_frames]
                f["probe_signal"][sl]  = self.probe_ram_on[:n_frames]
                f["audio"][sl]         = self.audio_ram_on[:n_frames]

        self._disk_offsets[idx] += n_frames
        if self._disk_offsets[idx] >= self.disk_buffer_size:
            self._disk_offsets[idx] = 0
            self._active_disk_idx   = (idx + 1) % self.num_disk_buffers
            self._disk_ready[idx]   = True

    # ───────────────────────────── public API ────────────────────────────────
    def enable_recording(self) -> None:
        if self.recording_enabled:
            log.warning("Recording already enabled.")
            return
        self._current_session_id = uuid.uuid4().hex[:8]
        self._session_renders.clear()
        self._ram_frame_idx = 0
        self._audio_rptr = self._audio_wptr     # align reader to “now”
        self.recording_enabled = True
        log.info("✅ Recording enabled – session %s", self._current_session_id)

    def disable_recording(self, *, _called_from_del: bool = False) -> None:

        if not self.recording_enabled and not _called_from_del:
            return

        self.recording_enabled = False

        log.info("Stopping recording...")

        # if self._ram_frame_idx:
        #     log.info("Flushing remaining RAM frames to disk...")
        #     last_flush_thread = self._flush_ram_to_disk()
        #     last_flush_thread.join()  # wait for last flush to finish
        #     log.info("RAM frames flushed to disk.")

        if self._ram_frame_idx:
            log.info("Flushing remaining %d frames to disk...", self._ram_frame_idx)
            self._flush_n_frames(self._ram_frame_idx)   # <── new
            self._ram_frame_idx = 0

        log.info("Marking all non-empty disk buffers as ready...")
        for i in range(self.num_disk_buffers):
            if self._disk_offsets[i] > 0:
                self._mark_buffer_ready(i)
        
        log.info("Disk buffers marked as ready.")
        time.sleep(2.5)  # give some time for the watcher to pick up changes

        log.info("Waiting for all render threads to finish...")
        self.wait_for_renders()
        log.info("All render threads finished.")

        #
        # # write any remaining audio in the ring to a raw wav (diagnostics)
        # samples = self._audio_wptr - self._audio_rptr
        # audio_left = self._pop_audio_frame()  # might be < one frame, that's OK
        # raw = np.concatenate([self._audio_ring[-samples:], audio_left])
        # raw_path = self.render_dir / f"slice_{self._current_session_id}.raw.wav"
        # sf.write(raw_path, raw, samplerate=self.audio_sr)
        # self._session_raw_wav = raw_path
        #
        # log.info("Waiting for render threads to finish...")
        # self.wait_for_renders()
        # log.info("Rendering session %s complete.", self._current_session_id)
        # 
        # if self._session_renders:
        #     out = self.render_dir / f"session_{self._current_session_id}.mp4"
        #     self._concat_mp4s(self._session_renders, out)

        # self._current_session_id = None
        # self._session_renders.clear()

    # ───────────────────────────── misc helpers ──────────────────────────────
    def feed_audio_block(self, pcm: np.ndarray) -> None:  # alias
        self.feed_audio(pcm)

    def wait(self, timeout: float | None = None) -> None:
        self.wait_for_renders(timeout)

    def _concat_mp4s(self, ins: List[Path], out: Path) -> None:
        if len(ins) == 1:
            ins[0].rename(out); return
        lst = out.with_suffix(".txt")
        lst.write_text("".join(f"file '{p.resolve()}'\n" for p in ins))
        cmd = [
            self.ffmpeg_bin, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(lst),           # listfile contains  ‘file path/to/render_00.mp4’ …
            "-c:v", self.ffmpeg_codec,     # libx264
            "-crf", str(self.ffmpeg_crf),  # keep same quality number you already use
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            str(out),                  # ripples_<hash>.mp4
        ]
        subprocess.run(cmd, check=True)

        lst.unlink(missing_ok=True)

    # rest of pose-graph helpers unchanged …

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
        self.max_frequency = self.speed / (2 * max(self.dx, self.dx))

    # ───────────────────────────── destructor ────────────────────────────────
    def __del__(self) -> None:

        if self.recording_enabled:
            try:
                self.disable_recording(_called_from_del=True)
                if self._all_renders:
                    h = hashlib.sha1("".join(map(str, self._all_renders)).encode()).hexdigest()[:10]
                    self._concat_mp4s(self._all_renders, self.render_dir / f"ripples_{h}.mp4")
            except Exception as exc:  # pragma: no cover
                log.exception("Destructor failed: %s", exc)


    # ───────────────────────────── frame update ──────────────────────────────
    def update(self, t: float) -> None:
        self.time = t

        for name, src in self.sources.items():
            res = src(t)
            self._excitation_results[name] = res
            # w = 1 / max(len(self.sources), 1) if name != "heart" else 0.001
            if name == "heart":
                w = 0.0001
            else:
                w = 1 / max(len(self.sources), 1)

            self.excitation += w * res["excitation"]

        if self.use_matrix:
            self._update_pose_graph_excitation()
        else:
            self.propagator.add_excitation(self.excitation)

        self.propagator.step()
        self.Z[:] = self.propagator.get_state()

        # ───── recording per-frame ─────
        if self.recording_enabled:
            i = self._ram_frame_idx
            self.Z_ram_on[i] = cp.asnumpy(self.Z) if self.use_gpu else self.Z
            self.probe_ram_on[i] = self.Z[self.probe_iy, self.probe_ix]
            self.audio_ram_on[i] = self._pop_audio_frame()
            self._ram_frame_idx += 1
            if self._ram_frame_idx >= self.ram_buffer_size:
                self._flush_ram_to_disk()
                self._ram_frame_idx = 0

        self.excitation[:] = 0
        self._log_counter += 1

        if 'heart' in self._excitation_results:
            return self._excitation_results['heart']['overlay']
        else:
            return None


    # expose current field
    def get_field(self) -> ArrayType:
        return self.Z
