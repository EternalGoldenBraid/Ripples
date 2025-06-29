import numpy as np
from scipy.sparse import coo_matrix

def build_grid_laplacian_coo(shape: tuple[int, int]) -> coo_matrix:
    H, W = shape
    N = H * W

    row_indices = []
    col_indices = []
    data = []

    def idx(y, x): return y * W + x

    for y in range(H):
        for x in range(W):
            i = idx(y, x)
            # Self-degree contribution (diagonal)
            degree = 0

            neighbors = [((y - 1) % H, x), ((y + 1) % H, x), (y, (x - 1) % W), (y, (x + 1) % W)]
            for ny, nx in neighbors:
                j = idx(ny, nx)
                row_indices.append(i)
                col_indices.append(j)
                data.append(-1.0)
                degree += 1

            # Diagonal entry
            row_indices.append(i)
            col_indices.append(i)
            data.append(degree)

    A = coo_matrix((data, (row_indices, col_indices)), shape=(N, N))
    return A
