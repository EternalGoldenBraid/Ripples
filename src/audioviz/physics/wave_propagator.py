from typing import Optional, Tuple, Dict, Any

import numpy as np
import cupy as cp
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
    2D Damped Wave Equation Propagator (GPU version).

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
    def __init__(self, shape, dx: float, dt: float, speed: float, damping: float):
        self.shape = shape
        self.dx: float = dx
        self.dt: float = dt
        self.c: float = speed
        self.damping: float = damping

        self.Z = cp.zeros(shape, dtype=cp.float32)
        self.Z_old = cp.zeros_like(self.Z)
        self.Z_new = cp.zeros_like(self.Z)

        self.c2_dt2: float = (self.c * self.dt / self.dx)**2

    def add_excitation(self, excitation: cp.ndarray):
        assert excitation.shape == self.Z.shape
        self.Z += excitation

    def step(self):
        Z = self.Z
        laplacian = (
            -4 * Z +
            cp.roll(Z, 1, axis=0) + cp.roll(Z, -1, axis=0) +
            cp.roll(Z, 1, axis=1) + cp.roll(Z, -1, axis=1)
        )
        self.Z_new = (2 * Z - self.Z_old + self.c2_dt2 * laplacian - (1-self.damping)*self.dt*(Z - self.Z_old))
        # self.Z_new = (2 * Z - self.Z_old + self.c2_dt2 * laplacian)
        # self.Z_new *= self.damping
        self.Z_old = Z.copy()
        self.Z = self.Z_new.copy()

    def get_state(self):
        return self.Z

    def reset(self):
        self.Z[:] = 0
        self.Z_old[:] = 0
        self.Z_new[:] = 0
