import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from audioviz.physics.wave_propagator import WavePropagatorGPU

class WavePropagatorTestSuite:
    def __init__(self, shape=(128, 128), steps=50, dx=5e-3, dt=5e-6, speed=10.0, damping=0.99):
        self.shape = shape
        self.N_grid = shape[0] * shape[1]
        self.steps = steps
        self.dx = dx
        self.dt = dt
        self.speed = speed
        self.damping = damping

        # Sanity check
        cfl = speed * dt / dx
        assert cfl < 1 / np.sqrt(2), f"CFL condition not satisfied (cfl={cfl:.3f})"

    def _create_propagators(self):
        # Stencil
        prop_stencil = WavePropagatorGPU(
            shape=self.shape,
            dx=self.dx,
            dt=self.dt,
            speed=self.speed,
            damping=self.damping,
            use_matrix=False,
        )

        # Matrix
        prop_matrix = WavePropagatorGPU(
            shape=self.shape,
            dx=self.dx,
            dt=self.dt,
            speed=self.speed,
            damping=self.damping,
            use_matrix=True,
        )

        return prop_stencil, prop_matrix

    def test_correctness(self, plot=False):
        prop_stencil, prop_matrix = self._create_propagators()

        init_exc = cp.zeros(self.shape, dtype=cp.float32)
        init_exc[self.shape[0] // 2, self.shape[1] // 2] = 1.0

        prop_stencil.add_excitation(init_exc)
        prop_matrix.add_excitation(init_exc)

        diff_norms = []

        for step in range(self.steps):
            prop_stencil.step()
            prop_matrix.step()

            Z_s = cp.asnumpy(prop_stencil.get_state())
            Z_m = cp.asnumpy(prop_matrix.get_state())

            diff_norm = np.linalg.norm(Z_s - Z_m)
            diff_norms.append(diff_norm)

            print(f"Step {step:02d} | Diff norm: {diff_norm:.3e}")

        if plot:
            plt.plot(diff_norms, label="Stencil vs Matrix Diff Norm")
            plt.xlabel("Step")
            plt.ylabel("Norm")
            plt.yscale("log")
            plt.legend()
            plt.show()

    def benchmark(self):
        prop_stencil, prop_matrix = self._create_propagators()

        init_exc = cp.zeros(self.shape, dtype=cp.float32)
        init_exc[self.shape[0] // 2, self.shape[1] // 2] = 1.0

        prop_stencil.add_excitation(init_exc)
        prop_matrix.add_excitation(init_exc)

        # --- Stencil timing
        start_s = time.time()
        for _ in range(self.steps):
            prop_stencil.step()
        duration_s = time.time() - start_s

        # --- Matrix timing
        start_m = time.time()
        for _ in range(self.steps):
            prop_matrix.step()
        duration_m = time.time() - start_m

        print(f"Stencil total time: {duration_s:.4f} s ({duration_s / self.steps:.6f} s/step)")
        print(f"Matrix total time:  {duration_m:.4f} s ({duration_m / self.steps:.6f} s/step)")

if __name__ == "__main__":
    suite = WavePropagatorTestSuite(shape=(128, 128), steps=50, dx=5e-3, dt=5e-6, speed=10.0, damping=0.99)
    suite.test_correctness(plot=True)

    suite = WavePropagatorTestSuite(
        shape=(1024, 1024),
        steps=50,
        dx=5e-3, dt=5e-6,
        speed=10.0, damping=0.99
    )
    suite.benchmark()
