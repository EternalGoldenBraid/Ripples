from typing import Optional, Union


import numpy as np
from scipy.sparse import coo_matrix

from audioviz.types import ArrayType
class PoseGraphState:
    def __init__(self, num_nodes: int, adjacency: Optional[ArrayType],
                 backend = np, velocity_smoothing_alpha: float = 1.0):
        self.num_nodes = num_nodes
        self.backend = backend
        self.xp = backend
        self.adjacency = adjacency

        # State shape: (num_nodes, 7) -> [x, y, vx, vy, ax, ay, ripple]
        self.state = self.xp.zeros((num_nodes, 7), dtype=self.xp.float32)

        # For finite difference
        self.prev_positions = self.xp.zeros((num_nodes, 2), dtype=self.xp.float32)
        self.prev_velocities = self.xp.zeros((num_nodes, 2), dtype=self.xp.float32)

        # For smoothed velocity
        self.smooth_velocities = self.xp.zeros((num_nodes, 2), dtype=self.xp.float32)
        self.velocity_smoothing_alpha = velocity_smoothing_alpha

    def __call__(self) -> np.ndarray:
        """
        Returns per-node excitation vector, here using velocity norms.
        Shape: (num_nodes,)
        """
        vnorms = np.linalg.norm(self.get_velocities(), axis=1)
        return vnorms

    def get_adjacency_coo(self) -> coo_matrix:
        """
        Return current adjacency matrix as SciPy COO matrix.
        """
        return coo_matrix(self.adjacency)

    def update(self, new_positions: np.ndarray, dt: float) -> None:
        """
        Update all state variables using new positions.
        """
        if self.backend != np:
            new_positions = self.xp.asarray(new_positions, dtype=self.xp.float32)

        velocities = (new_positions - self.prev_positions) / dt

        # Apply exponential moving average smoothing to velocities
        alpha = self.velocity_smoothing_alpha
        self.smooth_velocities = alpha * velocities + (1 - alpha) * self.smooth_velocities

        # Compute acceleration using smoothed velocity
        accelerations = (self.smooth_velocities - self.prev_velocities) / dt

        # Update state
        self.state[:, 0:2] = new_positions
        self.state[:, 2:4] = self.smooth_velocities
        self.state[:, 4:6] = accelerations
        # self.state[:, 6] is ripple state

        # Store for next frame
        self.prev_positions = new_positions.copy()
        self.prev_velocities = self.smooth_velocities.copy()

    def set_ripple_states(self, values: np.ndarray) -> None:
        assert values.shape == (self.num_nodes,)
        self.state[:, 6] = values

    def get_state_array(self) -> np.ndarray:
        return self.state.copy()

    def get_positions(self) -> np.ndarray:
        return self.state[:, 0:2]

    def get_velocities(self) -> np.ndarray:
        return self.state[:, 2:4]

    def get_accelerations(self) -> np.ndarray:
        return self.state[:, 4:6]

    def get_ripple_states(self) -> np.ndarray:
        return self.state[:, 6]
