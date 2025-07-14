from typing import Optional, Tuple, Dict, Any

import numpy as np
import cupy as cp
import cupyx
from loguru import logger as log

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

    Supports either:
        - finite-difference stencil on a 2D grid (stencil mode), or
        - explicit sparse Laplacian on arbitrary graphs (matrix mode).

    Parameters
    ----------
    shape : tuple
        Shape of the state vector. (H, W) for stencil, (N_nodes,) for matrix mode.
    dx : float
        Grid spacing.
    dt : float
        Time step.
    speed : float
        Wave speed c.
    damping : float
        Damping factor (1.0 = no damping).
    use_matrix : bool
        If True, use explicit sparse Laplacian.
    laplacian_csr : Optional[cupyx.scipy.sparse.csr_matrix]
        Sparse Laplacian matrix for matrix mode.
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 dx: float,
                 dt: float,
                 speed: float,
                 damping: float,
                 use_matrix: bool = False,
                 laplacian_csr: Optional[cupyx.scipy.sparse.csr_matrix] = None):

        self.shape = shape
        self.dx: float = dx
        self.dt: float = dt
        self.c: float = speed
        self.damping: float = damping
        self.use_matrix: bool = use_matrix

        self.c2_dt2: float = (self.c * self.dt / self.dx)**2

        if use_matrix:
            if laplacian_csr is None:
                raise ValueError("laplacian_csr must be provided when use_matrix=True.")
            if len(shape) != 1:
                raise ValueError("Shape must be 1D (N_nodes,) for matrix mode.")
            self.L: cupyx.scipy.sparse.csr_matrix = laplacian_csr

            self.step = self._step_matrix
            self.add_excitation = self._add_excitation_matrix
        else:
            if len(shape) != 2:
                raise ValueError("Shape must be 2D (H, W) for stencil mode.")

            self.step = self._step_stencil
            self.add_excitation = self._add_excitation_stencil

        self.Z = cp.zeros(shape, dtype=cp.float32)
        self.Z_old = cp.zeros_like(self.Z)
        self.Z_new = cp.zeros_like(self.Z)

    def _add_excitation_stencil(self, excitation: cp.ndarray) -> None:
        assert excitation.shape == self.Z.shape
        self.Z += excitation

    def _add_excitation_matrix(self, excitation: cp.ndarray) -> None:
        assert excitation.shape == self.Z.shape
        self.Z += excitation

    def _step_matrix(self) -> None:
        laplacian_Z = -1*self.L @ self.Z
        self.laplacian_Z = cp.asarray(laplacian_Z, dtype=cp.float32) #NOTE: for debugging

        # self.Z_new = (2 * Z - self.Z_old + self.c2_dt2 * laplacian) # No damping
        # self.Z_new = (2 * self.Z - self.Z_old + self.c2_dt2 * laplacian_Z - (self.damping)*self.dt*(self.Z - self.Z_old)) # True damping
        # self.Z_new = (2 * Z - self.Z_old + self.c2_dt2 * laplacian)*self.damping

        # self.Z_old = self.Z.copy()
        # self.Z = self.Z_new.copy()
        self._apply_leapfrog_update(laplacian_Z)

    def _step_stencil(self) -> None:
        laplacian_Z = (
            -4 * self.Z +
            cp.roll(self.Z, 1, axis=0) + cp.roll(self.Z, -1, axis=0) +
            cp.roll(self.Z, 1, axis=1) + cp.roll(self.Z, -1, axis=1)
        )

        self.laplacian_Z = cp.asarray(laplacian_Z, dtype=cp.float32) #NOTE: for debugging


        # # self.Z_new = (2 * Z - self.Z_old + self.c2_dt2 * laplacian) # No damping
        # self.Z_new = (2 * self.Z - self.Z_old + self.c2_dt2 * laplacian_Z - (self.damping) * self.dt * (self.Z - self.Z_old)) # True damping
        # # self.Z_new = (2 * Z - self.Z_old + self.c2_dt2 * laplacian)*self.damping
        # self.Z_old = self.Z.copy()
        # self.Z = self.Z_new.copy()
        self._apply_leapfrog_update(laplacian_Z)

    def _apply_leapfrog_update(self, laplacian_Z: cp.ndarray) -> None:
        self.Z_new = (
            2 * self.Z - self.Z_old + self.c2_dt2 * laplacian_Z
            - (self.damping) * self.dt * (self.Z - self.Z_old)
        )
        self.Z_new -= self.Z_new.mean()
        # self.Z_new = (2 * Z - self.Z_old + self.c2_dt2 * laplacian_Z)*self.damping
        self.Z_old = self.Z.copy()
        self.Z = self.Z_new.copy()


    def get_state(self) -> cp.ndarray:
        return self.Z

    def reset(self) -> None:
        self.Z[:] = 0
        self.Z_old[:] = 0
        self.Z_new[:] = 0
