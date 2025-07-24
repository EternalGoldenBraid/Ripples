from abc import ABC, abstractmethod
import numpy as np
from typing import Dict

class PoseGraphExtractor(ABC):
    @abstractmethod
    def extract(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Takes an RGB frame, returns a dict with:
            - 'coords': np.ndarray of shape (num_nodes, 2)
            - 'adjacency': np.ndarray (num_nodes, num_nodes), optional for connectivity
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Free any resources if necessary (e.g., for GPU sessions).
        """
        pass
