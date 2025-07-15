from typing import Tuple
import numpy as np
from scipy.sparse import coo_matrix, diags

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

def apply_boundary_conditions(
        adj: coo_matrix,
        shape: Tuple[int, int],
        mask: np.ndarray,
        reflection_R: float = 1.0,          # 1 = mirror, 0 = transparent
        bc_type: str = "neumann",        # "neumann" or "dirichlet"
):
    """
    Build a Laplacian that realises partial reflection | R | <= 1
    across an internal mask.  Outside the mask the grid is untouched.

    Parameters
    ----------
    reflection_R : float
        Desired reflected *power* ratio (0-1).
    bc_type: str
        "neumann"  -> no phase flip
        "dirichlet"-> phase-inverting (sign flip)
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)

    H, W   = shape
    N      = H * W
    flat   = mask.ravel()

    R = np.clip(reflection_R, 0.0, 1.0)
    w = (1.0 - R) / (1.0 + R)          # see §1

    # ---------- cut or down-weight crossing edges ------------------
    rows, cols = adj.row, adj.col
    data       = adj.data.copy()

    cross = flat[rows] ^ flat[cols]    # True = across the wall
    data[cross] *= w                   # keep edge with weight 0<=w<=1

    A_mod = coo_matrix((data, (rows, cols)), shape=(N, N))

    # ---------- diagonal (degree) ----------------------------------
    deg = np.asarray(A_mod.sum(axis=1)).flatten()

    if bc_type.lower().startswith("dir"):
        # Dirichlet wants the *full* 4 on the diagonal for nodes inside the wall
        need_boost            = flat & (deg < 4.0)
        deg[need_boost] += (4.0 - deg[need_boost])

    D_mod = diags(deg, dtype=np.float32)

    return (D_mod - A_mod).tocoo()

def apply_boundary_conditions_old(adj: coo_matrix, shape: Tuple[int, int], bc_type: str = "periodic") -> coo_matrix:
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
