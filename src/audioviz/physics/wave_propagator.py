from typing import Optional, Tuple, Dict, Any



import numpy as np
import cupy as cp
import cupyx
from loguru import logger as log
from scipy.sparse import coo_matrix

from audioviz.utils.graph_utils import (
        build_grid_adjacency, apply_boundary_conditions
)

class WavePropagatorCPU:
    """
    2D Damped Wave Equation Propagator (CPU version).

    Implements explicit leap-frog time stepping for the damped wave equation
    on a uniform 2D grid using a five-point finite-difference stencil.

    Update formula:

        Z_new = 2 * Z - Z_old + c2_dt2 * laplacian(Z)
        Z_new *= damping

    where:
        laplacian(Z) = -4*Z + roll(Z, ±1, x) + roll(Z, ±1, y)
        c2_dt2       = (c * dt / dx)^2
        damping      = user-defined damping coefficient (1 − gamma * dt).

    The Laplacian uses periodic boundary conditions (via np.roll).

    For a complete mathematical derivation and stability analysis,
    see: docs/wave_derivation.pdf

    Parameters
    ----------
    shape : tuple
        Grid shape (height, width).
    dx : float
        Grid spacing.
    dt : float
        Time step.
    speed : float
        Wave speed c.
    damping : float
        Damping factor (1.0 = no damping).
    """
    def __init__(self, shape, dx, dt, speed, damping):
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = speed
        self.damping = damping

        self.Z = np.zeros(shape, dtype=np.float32)
        self.Z_old = np.zeros_like(self.Z)
        self.Z_new = np.zeros_like(self.Z)

        self.c2_dt2 = (self.c * self.dt / self.dx)**2

    def add_excitation(self, excitation: np.ndarray):
        assert excitation.shape == self.Z.shape
        self.Z += excitation

    def step(self):
        Z = self.Z
        laplacian = (
            -4 * Z +
            np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)
        )
        self.Z_new = (2 * Z - self.Z_old + self.c2_dt2 * laplacian)
        self.Z_new *= self.damping
        self.Z_old = Z.copy()
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
    
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.c = speed
        self.damping = damping
        self.use_matrix = use_matrix
        self.pose_graph_state = pose_graph_state
    
        self.c2_dt2: float = (self.c * self.dt / self.dx)**2
    
        if use_matrix:

            # mask = np.zeros(shape, dtype=bool)
            # # Wall max a circle of ones in the center of the grid
            # radius = min(shape) // 4  # Example radius
            # center = (shape[0] // 2, shape[1] // 2)
            # # Set to ones on the perimeter of the circle not on the inside
            # for i in range(shape[0]):
            #     for j in range(shape[1]):
            #         if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
            #             mask[i, j] = True
            self._init_matrix(shape=shape, pose_graph_state=pose_graph_state,
                              # mask=mask
                              )
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
    
        self.extended_excitation: cp.ndarray = cp.zeros(self.shape, dtype=cp.float32)
        self.step = self._step_matrix
        self.add_excitation = self._add_excitation_matrix

    def _add_excitation_stencil(self, excitation: cp.ndarray) -> None:
        assert excitation.shape == self.Z.shape, \
            f"Excitation shape {excitation.shape} does not match state shape {self.Z.shape}."
        self.Z += excitation

    def _add_excitation_matrix(self, excitation: cp.ndarray) -> None:
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

    def _apply_leapfrog_update(self, laplacian_Z: cp.ndarray) -> None:
        self.Z_new = (
            2 * self.Z - self.Z_old + self.c2_dt2 * laplacian_Z
            - (self.damping) * self.dt * (self.Z - self.Z_old)
        )
        self.Z_old = self.Z.copy()
        self.Z = self.Z_new.copy()

    def get_state(self) -> cp.ndarray:
        return self.Z

    def reset(self) -> None:
        self.Z[:] = 0
        self.Z_old[:] = 0
        self.Z_new[:] = 0
