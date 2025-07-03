from typing import Optional, Dict, Tuple, Union
import numpy as np
import cupy as cp
import cupyx
from scipy.sparse import coo_matrix
from loguru import logger as log

from audioviz.physics.wave_propagator import WavePropagatorGPU, WavePropagatorCPU
from audioviz.sources.base import ExcitationSourceBase
from audioviz.sources.camera import CameraSource
from audioviz.sources.pose.mediapipe_pose_source import MediaPipePoseExtractor
from audioviz.sources.pose.pose_graph_state import PoseGraphState
from audioviz.utils.graph_utils import build_grid_adjacency, build_combined_laplacian


class RippleEngine:
    def __init__(self,
                 resolution: Tuple[int, int],
                 dx: float,
                 speed: float,
                 damping: float,
                 use_gpu: bool = True):

        self.resolution = resolution
        self.dx = dx
        self.speed = speed
        self.damping = damping
        self.use_gpu = use_gpu

        self.max_frequency = self.speed / (2 * max(self.dx, self.dx))

        self.xp = cp if use_gpu else np

        self.Z = self.xp.zeros(self.resolution, dtype=self.xp.float32)
        self.excitation = self.xp.zeros(self.resolution, dtype=self.xp.float32)

        self.safe_factor = 0.95
        self.dt = (max(dx, dx) / speed) * 1 / np.sqrt(2) * self.safe_factor

        self.propagator_kwargs = {
            "shape": self.resolution,
            "dx": self.dx,
            "dt": self.dt,
            "speed": self.speed,
            "damping": self.damping,
        }
        self.propagator = WavePropagatorGPU(**self.propagator_kwargs) if use_gpu else \
                          WavePropagatorCPU(**self.propagator_kwargs)

        self.sources: Dict[str, ExcitationSourceBase] = {}

        self.pose_graph_state: Optional[PoseGraphState] = None
        self.camera: Optional[CameraSource] = None
        self.extractor: Optional[MediaPipePoseExtractor] = None
        self.use_matrix = False

        self.time = 0.0
        self.log_counter_ = 0

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

    def update(self, t: float):
        self.time = t

        for name, source in self.sources.items():
            result = source(t)

            if name != 'heart':
                weight = 1 / max(len(self.sources), 1)
            else:
                weight = 0.001

            self.excitation[:] += weight * result["excitation"]


        if self.pose_graph_state and self.camera and self.extractor:
            frame = self.camera.read()
            if frame is not None:
                pose_data = self.extractor.extract(frame)
                coords = pose_data["coords"]
                if coords.shape[0] > 0:
                    self.pose_graph_state.update(coords, self.dt)

            flat_grid = self.excitation.ravel()
            self.extended_excitation[:self.N_grid] = flat_grid

            if self.log_counter_ % 30000000 == 0:
                pose_ex_vec = self.pose_graph_state()
                for i in self.coupled_pose_indices:
                    val = np.clip(pose_ex_vec[i] / 50000, 0, 0.02)
                    self.extended_excitation[self.N_grid + i] = val
                    log.debug(f"Extended excitation for pose node {i}: {val:.4f}")

            self.propagator.add_excitation(self.extended_excitation)
            self.extended_excitation[:self.N_grid] = 0
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

        self.excitation[:] = 0
        self.log_counter_ += 1

    def get_field(self) -> Union[np.ndarray, cp.ndarray]:
        return self.Z
