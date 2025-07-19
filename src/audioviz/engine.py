from typing import Optional, Dict, Tuple, Union, Any
import numpy as np
from scipy.sparse import coo_matrix
from loguru import logger as log

from audioviz.sources.base import ExcitationSourceBase
#from audioviz.sources.camera import CameraSource
#from audioviz.sources.pose.mediapipe_pose_source import MediaPipePoseExtractor
#from audioviz.sources.pose.pose_graph_state import PoseGraphState

from audioviz.types import ArrayType

class RippleEngine:
    def __init__(self,
                 resolution: Tuple[int, int],
                 dx: float,
                 speed: float,
                 damping: float,
                 use_gpu: bool = True,
                 use_matrix: bool = True,
                 pose_graph_state: Optional[Any] = None):

        self.resolution = resolution
        self.dx = dx
        self.speed = speed
        self.damping = damping
        self.use_gpu = use_gpu
        self.use_matrix = use_matrix
        self.pose_graph_state = pose_graph_state

        self.max_frequency = self.speed / (2 * max(self.dx, self.dx))
        if use_gpu:
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np

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
            "use_matrix": self.use_matrix,
            "pose_graph_state": self.pose_graph_state,
        }
       
        if use_gpu:
            from audioviz.physics.wave_propagator import WavePropagatorGPU
            import pdb; pdb.set_trace() 
            self.propagator = WavePropagatorGPU(**self.propagator_kwargs)
        else:
            from audioviz.physics.wave_propagator import WavePropagatorCPU
            self.propagator = WavePropagatorCPU(**self.propagator_kwargs)

        self.sources: Dict[str, ExcitationSourceBase] = {}
        self.time = 0.0

        self.mask_set = False

    def add_source(self, source: ExcitationSourceBase):
        if source.name in self.sources:
            raise ValueError(f"Source '{source.name}' already exists.")

        self.sources[source.name] = source
        log.info(f"Added source '{source.name}' to RippleEngine.")


        # NEW: check for boundary mask
        if hasattr(source, "get_boundary_mask"):
            mask: ArrayType = source.get_boundary_mask()
            if mask is not None:
                log.info(f"Source '{source.name}' provides boundary mask. Updating propagator...")
                self.propagator.update_boundary_mask(mask)
        else:
            log.warning(f"Source '{source.name}' does not provide a boundary mask.")


    def update(self, t: float):
        self.time = t

        for name, source in self.sources.items():
            result = source(t)
            weight = 1 / max(len(self.sources), 1)
            self.excitation[:] += weight * result["excitation"]

            if name == 'heart':
                boundary_mask = result['boundary']['region']

        if not self.mask_set:
            self.propagator.update_boundary_mask(
                mask=boundary_mask
            )
            self.mask_set = True

        self.propagator.add_excitation(self.excitation)
        self.propagator.step()
        self.Z[:] = self.propagator.get_state().reshape(self.resolution)
        self.excitation[:] = 0

    def get_field(self) -> ArrayType:
        return self.Z
