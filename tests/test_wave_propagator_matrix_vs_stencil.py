import numpy as np
import cupy as cp
import cupyx
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from audioviz.physics.wave_propagator import WavePropagatorGPU
from audioviz.utils.graph_utils import build_grid_adjacency

def test_plot_wave_propagator_matrix_vs_stencil():
    shape = (5, 5)
    N_grid = shape[0] * shape[1]

    dx = 5e-3
    dt = 5e-6
    speed = 10.0
    damping = 0.99

    cfl = speed * dt / dx
    assert cfl < 1/np.sqrt(2), "CFL condition not satisfied! Adjust dt or dx."

    # ---------------- Stencil propagator
    prop_stencil = WavePropagatorGPU(
        shape=shape,
        dx=dx,
        dt=dt,
        speed=speed,
        damping=damping,
        use_matrix=False,
    )

    # ---------------- Matrix propagator
    grid_adj = build_grid_adjacency(shape)
    degrees = np.array(grid_adj.sum(axis=1)).flatten()
    D = coo_matrix((degrees, (np.arange(grid_adj.shape[0]), np.arange(grid_adj.shape[0]))), shape=grid_adj.shape)
    L_coo = (D - grid_adj)
    L_csr = L_coo.tocsr()
    L_csr_gpu = cupyx.scipy.sparse.csr_matrix(L_csr)

    prop_matrix = WavePropagatorGPU(
        shape=(N_grid,),
        dx=dx,
        dt=dt,
        speed=speed,
        damping=damping,
        use_matrix=True,
        laplacian_csr=L_csr_gpu,
    )

    # ---------------- Same initial state
    init_exc = cp.zeros(shape, dtype=cp.float32)
    init_exc[shape[0]//2, shape[1]//2] = 1.0  # small impulse

    prop_stencil.add_excitation(init_exc)

    flat_exc = init_exc.ravel()
    prop_matrix.add_excitation(flat_exc)

    diff_norms = []

    steps = 50
    for step in range(steps):
        prop_stencil.step()
        prop_matrix.step()

        # --- Capture states
        Z_stencil = cp.asnumpy(prop_stencil.get_state())
        Z_matrix = cp.asnumpy(prop_matrix.get_state()).reshape(shape)

        lap_stencil = cp.asnumpy(prop_stencil.laplacian_Z)
        lap_matrix = cp.asnumpy(prop_matrix.laplacian_Z).reshape(shape)

        diff_norm = np.linalg.norm(Z_stencil - Z_matrix)
        diff_norms.append(diff_norm)

        print(f"Step {step:02d} | Diff norm: {diff_norm:.3e}")

    # --- Final Laplacians
    fig3, axs3 = plt.subplots(1, 3, figsize=(15, 5))

    vlim = max(np.abs(lap_stencil).max(), np.abs(lap_matrix).max(), 1e-6)

    im0 = axs3[0].imshow(lap_stencil, cmap="seismic", vmin=-vlim, vmax=vlim)
    axs3[0].set_title("Stencil Laplacian")
    plt.colorbar(im0, ax=axs3[0])

    im1 = axs3[1].imshow(lap_matrix, cmap="seismic", vmin=-vlim, vmax=vlim)
    axs3[1].set_title("Matrix Laplacian")
    plt.colorbar(im1, ax=axs3[1])

    im2 = axs3[2].imshow(lap_stencil - lap_matrix, cmap="seismic")
    axs3[2].set_title("Laplacian Difference")
    plt.colorbar(im2, ax=axs3[2])

    plt.tight_layout()
    plt.show()

    # --- Final state fields
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axs2[0].imshow(Z_stencil, cmap="seismic", vmin=-1, vmax=1)
    axs2[0].set_title("Stencil Final")
    plt.colorbar(im0, ax=axs2[0])

    im1 = axs2[1].imshow(Z_matrix, cmap="seismic", vmin=-1, vmax=1)
    axs2[1].set_title("Matrix Final")
    plt.colorbar(im1, ax=axs2[1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_plot_wave_propagator_matrix_vs_stencil()
