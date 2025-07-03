import numpy as np


class MockSource:
    def __init__(self, iy, ix, resolution):
        self.iy = iy
        self.ix = ix
        self.resolution = resolution
        self.name = "mock"

    def __call__(self, t: float):
        exc = np.zeros(self.resolution, dtype=np.float32)
        exc[self.iy, self.ix] = np.sin(t)
        return {"excitation": exc}
    
    def get_controls(self):
        return []
