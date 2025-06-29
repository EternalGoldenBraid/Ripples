from typing import Tuple
import numpy as np
from scipy.sparse import coo_matrix

def build_grid_adjacency(shape: tuple[int, int]) -> coo_matrix:
    H, W = shape
    N = H * W

    row = []
    col = []
    data = []

    def idx(y, x): return y * W + x

    for y in range(H):
        for x in range(W):
            i = idx(y, x)
            neighbors = [((y - 1) % H, x), ((y + 1) % H, x), (y, (x - 1) % W), (y, (x + 1) % W)]
            for ny, nx in neighbors:
                j = idx(ny, nx)
                row.append(i)
                col.append(j)
                data.append(1.0)

    return coo_matrix((data, (row, col)), shape=(N, N))


def build_combined_laplacian(grid_adj: coo_matrix, pose_adj: coo_matrix, coupling: coo_matrix) -> coo_matrix:
    from scipy.sparse import hstack, vstack

    top = hstack([grid_adj, coupling])
    bottom = hstack([coupling.transpose(), pose_adj])
    A = vstack([top, bottom]).tocoo()

    degrees = np.array(A.sum(axis=1)).flatten()
    D = coo_matrix((degrees, (np.arange(A.shape[0]), np.arange(A.shape[0]))), shape=A.shape)

    L = D - A
    return L
