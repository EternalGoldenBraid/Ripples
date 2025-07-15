from typing import Tuple
import numpy as np
from scipy.sparse import coo_matrix

def build_grid_adjacency(shape: Tuple[int, int]) -> coo_matrix:
    H, W = shape
    N = H * W

    row = []
    col = []
    data = []

    def idx(y, x): return y * W + x

    for y in range(H):
        for x in range(W):
            i = idx(y, x)

            if y > 0:
                j = idx(y - 1, x)
                row.append(i)
                col.append(j)
                data.append(1.0)
            if y < H - 1:
                j = idx(y + 1, x)
                row.append(i)
                col.append(j)
                data.append(1.0)
            if x > 0:
                j = idx(y, x - 1)
                row.append(i)
                col.append(j)
                data.append(1.0)
            if x < W - 1:
                j = idx(y, x + 1)
                row.append(i)
                col.append(j)
                data.append(1.0)

    return coo_matrix((data, (row, col)), shape=(N, N))


def apply_boundary_conditions(adj: coo_matrix, shape: Tuple[int, int], bc_type: str = "periodic") -> coo_matrix:
    H, W = shape

    if bc_type == "periodic":
        raise NotImplementedError("Periodic BC logic to reconnect edges not implemented yet.")
    elif bc_type == "dirichlet":
        raise NotImplementedError("Dirichlet BC logic to remove edge connections on boundaries not implemented yet.")
    elif bc_type == "neumann":
        raise NotImplementedError("Neumann BC logic to adjust edge weights not implemented yet.")
    else:
        raise ValueError(f"Unknown boundary condition: {bc_type}")

    return adj  # In default (future fallback) could return unmodified

def combine_laplacians(L1: coo_matrix, L2: coo_matrix, L_coupling: coo_matrix = None) -> coo_matrix:
    """
    Combine two Laplacians (and optional coupling Laplacian).

    Parameters
    ----------
    L1 : coo_matrix
        Laplacian of first subgraph.
    L2 : coo_matrix
        Laplacian of second subgraph.
    L_coupling : coo_matrix, optional
        Coupling Laplacian to connect subgraphs.

    Returns
    -------
    L_combined : coo_matrix
        Combined Laplacian matrix.
    """
    raise NotImplementedError("Dynamic Laplacian combination logic not implemented yet.")
