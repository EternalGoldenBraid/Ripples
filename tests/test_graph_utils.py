import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
plt.style.use('dark_background')


def idx(y, x, W):
    return y * W + x

def build_expected_adjacency(shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    N = H * W
    expected_adj = np.zeros((N, N), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            i = idx(y, x, W)
            neighbors = [
                ((y - 1) % H, x),  # up
                ((y + 1) % H, x),  # down
                (y, (x - 1) % W),  # left
                (y, (x + 1) % W),  # right
            ]
            for ny, nx in neighbors:
                j = idx(ny, nx, W)
                expected_adj[i, j] = 1.0

    return expected_adj


def plot_adjacency_and_graph(expected_adj: np.ndarray, adj_coo: coo_matrix, shape: tuple[int, int]):
    adj_dense = adj_coo.toarray().astype(np.float32)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].imshow(expected_adj, cmap='gray')
    axs[0].set_title("Expected (Manual) Adjacency")

    axs[1].imshow(adj_dense, cmap='gray')
    axs[1].set_title("Built by Function")

    plt.tight_layout()
    plt.show()

    # Optional: plot graph connectivity
    plot_grid_adjacency_as_graph(adj_coo, shape)


def plot_grid_adjacency_as_graph(adj_coo: coo_matrix, shape: tuple[int, int]):
    H, W = shape
    N = H * W

    coords = np.array([[x, -y] for y in range(H) for x in range(W)])

    plt.figure(figsize=(6, 6))

    rows, cols = adj_coo.nonzero()
    for i, j in zip(rows, cols):
        if i < j:
            xi, yi = coords[i]
            xj, yj = coords[j]
            plt.plot([xi, xj], [yi, yj], color="gray", linewidth=1)

    plt.scatter(coords[:, 0], coords[:, 1], s=100, color="red", zorder=3)
    plt.gca().set_aspect('equal')
    plt.title("Grid Graph Connectivity")
    plt.axis("off")
    plt.show()


def test_build_grid_adjacency_combined():
    from audioviz.utils.graph_utils import build_grid_adjacency

    shape = (3, 3)
    expected_adj = build_expected_adjacency(shape)

    adj_coo = build_grid_adjacency(shape)
    adj_dense = adj_coo.toarray().astype(np.float32)

    # --- Assert equality
    assert np.allclose(adj_dense, expected_adj), "Adjacency matrices do not match!"

    print("✅ test_build_grid_adjacency_combined passed.")

    # --- Plot both matrices and graph connectivity
    plot_adjacency_and_graph(expected_adj, adj_coo, shape)


if __name__ == "__main__":
    test_build_grid_adjacency_combined()
