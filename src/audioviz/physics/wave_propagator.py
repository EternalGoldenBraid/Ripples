from typing import Optional, Tuple, Dict, Any


import numpy as np
from loguru import logger as log
from scipy.sparse import coo_matrix, diags, csr_matrix

from audioviz.utils.graph_utils import (
        build_grid_adjacency, apply_boundary_conditions
)
from audioviz.types import ArrayType

class WavePropagatorCPU:
    def __init__(self,
                 shape,
                 dx,
                 dt,
                 speed,
                 damping,
                 use_matrix=True,
                 pose_graph_state=None):
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = speed
        self.damping = damping
        self.use_matrix = use_matrix
        self.pose_graph_state = pose_graph_state

        self.c2_dt2 = (self.c * self.dt / self.dx)**2

        self.Z = np.zeros(self.shape, dtype=np.float32)
        self.Z_old = np.zeros_like(self.Z)
        self.Z_new = np.zeros_like(self.Z)

        if self.use_matrix:
            self._init_matrix(shape=self.shape, pose_graph_state=pose_graph_state)
            self.step = self._step_matrix
            self.add_excitation = self._add_excitation_matrix
        else:
            self.step = self._step_stencil
            self.add_excitation = self._add_excitation_stencil

    def _init_matrix(self, shape, mask=None, pose_graph_state=None):
        log.info("Building internal Laplacian matrix (CPU)...")

        N = shape[0] * shape[1] if len(shape) == 2 else shape[0]
        grid_adj = build_grid_adjacency(shape)

        if pose_graph_state:
            raise NotImplementedError("Pose graph state not supported yet.")

        if mask is not None:
            L = apply_boundary_conditions(adj=grid_adj, shape=shape,
                                          mask=mask, bc_type="dirichlet",
                                          reflection_R=0.5)
            log.info("✅ Grid Laplacian with boundary mask applied.")
        else:
            degrees = diags(np.array(grid_adj.sum(axis=1)).flatten(), 0)
            A = coo_matrix((grid_adj.data, (grid_adj.row, grid_adj.col)), shape=(N, N))
            L = (degrees - A).tocoo()
            log.info("✅ Grid Laplacian without boundary mask.")

        self.L = L.tocsr()
        self.extended_excitation = np.zeros(self.shape, dtype=np.float32)

    def update_boundary_mask(self, mask: np.ndarray):
        assert mask.shape == self.shape
        self._init_matrix(shape=self.shape, mask=mask)

    def _add_excitation_stencil(self, excitation: ArrayType):
        assert excitation.shape == self.Z.shape
        self.Z += excitation

    def _add_excitation_matrix(self, excitation: ArrayType):
        if excitation.shape == self.Z.shape:
            self.Z += excitation
        elif len(excitation.shape) == 2 and np.prod(excitation.shape) == self.Z.size:
            self.Z += excitation.ravel()
        else:
            raise ValueError(f"Incompatible excitation shape: {excitation.shape}")

    def _step_stencil(self):
        laplacian = (
            -4 * self.Z +
            np.roll(self.Z, 1, axis=0) + np.roll(self.Z, -1, axis=0) +
            np.roll(self.Z, 1, axis=1) + np.roll(self.Z, -1, axis=1)
        )
        self._apply_leapfrog_update(laplacian)

    def _step_matrix(self):
        laplacian = (-1 * self.L @ self.Z.ravel()).reshape(self.shape)
        self._apply_leapfrog_update(laplacian)

    def _apply_leapfrog_update(self, laplacian):
        self.Z_new = (
            2 * self.Z - self.Z_old + self.c2_dt2 * laplacian
            - self.damping * self.dt * (self.Z - self.Z_old)
        )
        self.Z_old = self.Z.copy()
        self.Z = self.Z_new.copy()

    def get_state(self):
        return self.Z

    def reset(self):
        self.Z[:] = 0
        self.Z_old[:] = 0
        self.Z_new[:] = 0

class WavePropagatorGPU:
    """
    2D or graph-based damped wave equation propagator (GPU version).

    Parameters
    ----------
    shape : tuple
        Grid shape or (N_nodes,) if using graph Laplacian.
    dx : float
        Grid spacing.
    dt : float
        Time step.
    speed : float
        Wave speed c.
    damping : float
        Damping factor (1.0 = no damping).
    use_matrix : bool
        If True, build and use graph Laplacian internally.
    pose_graph_state : Optional[Any]
        Optional pose graph state object to combine with grid adjacency.
    """
    def __init__(self,
                 shape: Tuple[int, ...],
                 dx: float,
                 dt: float,
                 speed: float,
                 damping: float,
                 use_matrix: bool = True,
                 pose_graph_state: Optional[Any] = None):

        import cupy as cp
        import cupyx.scipy.sparse as cupyx

        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = speed
        self.damping = damping
        self.use_matrix = use_matrix
        self.pose_graph_state = pose_graph_state
    
        self.c2_dt2: float = (self.c * self.dt / self.dx)**2
    
        if use_matrix:
            self._init_matrix(shape=shape, pose_graph_state=pose_graph_state)
        else:
            if len(shape) != 2:
                raise ValueError("Shape must be 2D (H, W) for stencil mode.")
            self.step = self._step_stencil
            self.add_excitation = self._add_excitation_stencil

        self.Z = cp.zeros(self.shape, dtype=cp.float32)
        self.Z_old = cp.zeros_like(self.Z)
        self.Z_new = cp.zeros_like(self.Z)
        
    def update_boundary_mask(self, mask: np.ndarray):
        """
        Updates the internal Laplacian using a new boundary mask.
        
        Parameters:
        -----------
        mask : np.ndarray
            Boolean array (H, W) where True indicates wall/boundary.
        """
        assert mask.shape == self.shape, f"Mask shape {mask.shape} must match simulation shape {self.shape}."
        self._init_matrix(self.shape, mask=mask)


    def _init_matrix(self,
                     shape: Tuple[int, ...],
                     mask: Optional[np.ndarray] = None,
                     pose_graph_state: Optional[Any] = None):
    
        log.info("Building internal Laplacian matrix...")
    
        N_grid = shape[0] * shape[1] if len(shape) == 2 else shape[0]
    
        grid_adj = build_grid_adjacency(shape)
    
        if pose_graph_state:
            raise NotImplementedError("Pose graph state integration not implemented yet.")
            # --- Future: pose graph logic here ---
    
        else:
    

            if mask is not None:
                L_coo = apply_boundary_conditions(
                        adj=grid_adj, shape=shape,
                        mask=mask, 
                        # bc_type="neumann"
                        bc_type="dirichlet",
                        reflection_R=0.5,  # Example reflection ratio
                )
                log.info("✅ Grid Laplacian with boundary constraints constructed.")
            else:
                from scipy.sparse import diags, coo_matrix
                degrees = diags(np.array(grid_adj.sum(axis=1)).flatten())
                rows, cols = grid_adj.row, grid_adj.col
                data       = grid_adj.data.copy()
                assert len(shape) == 2, "Shape must be 2D (H, W) for grid Laplacian."
                N = shape[0] * shape[1]
                A = coo_matrix((data, (rows, cols)), shape=(N, N))
                D = diags(np.asarray(A.sum(axis=1)).flatten(), dtype=np.float32)
                L_coo = (degrees - A).tocoo()


                log.info("✅ Grid Laplacian without boundary constraints constructed.")
    
        # --- Convert to CSR and move to GPU ---
        L_csr = L_coo.tocsr()
        self.L = cupyx.scipy.sparse.csr_matrix(L_csr)
    
        self.extended_excitation: ArrayType = cp.zeros(self.shape, dtype=cp.float32)
        self.step = self._step_matrix
        self.add_excitation = self._add_excitation_matrix

    def _add_excitation_stencil(self, excitation: ArrayType) -> None:
        assert excitation.shape == self.Z.shape, \
            f"Excitation shape {excitation.shape} does not match state shape {self.Z.shape}."
        self.Z += excitation

    def _add_excitation_matrix(self, excitation: ArrayType) -> None:
        # If excitation shape matches self.shape, fine
        if excitation.shape == self.Z.shape:
            self.Z += excitation
        # If excitation is 2D and matches grid shape, reshape
        elif len(excitation.shape) == 2 and np.prod(excitation.shape).item() == self.Z.size:
            self.Z += excitation.ravel()
        else:
            raise ValueError(
                f"Excitation shape {excitation.shape} is incompatible with internal state shape {self.Z.shape}"
            )

    def _step_matrix(self) -> None:
        self._apply_leapfrog_update((-1 * self.L @ self.Z.ravel()).reshape(self.shape))

    def _step_stencil(self) -> None:
        self._apply_leapfrog_update(
            -4 * self.Z +
            cp.roll(self.Z, 1, axis=0) + cp.roll(self.Z, -1, axis=0) +
            cp.roll(self.Z, 1, axis=1) + cp.roll(self.Z, -1, axis=1)
        )

    def _apply_leapfrog_update(self, laplacian_Z: ArrayType) -> None:
        self.Z_new = (
            2 * self.Z - self.Z_old + self.c2_dt2 * laplacian_Z
            - (self.damping) * self.dt * (self.Z - self.Z_old)
        )
        self.Z_old = self.Z.copy()
        self.Z = self.Z_new.copy()

    def get_state(self) -> ArrayType:
        return self.Z

    def reset(self) -> None:
        self.Z[:] = 0
        self.Z_old[:] = 0
        self.Z_new[:] = 0
