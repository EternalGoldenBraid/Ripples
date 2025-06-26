from typing import List
import numpy as np
from .base import ExcitationSourceBase

class ExcitationManager:
    def __init__(self, sources: List[ExcitationSourceBase]):
        self.sources = sources

    def get_combined_excitation(self, t: float) -> np.ndarray:
        return np.sum([source(t) for source in self.sources])
