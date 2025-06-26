from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class ExcitationSourceBase(ABC):
    """
    Abstract base class for all excitation sources that produce a 2D wave excitation field.
    
    Any subclass must implement the `__call__` method, which returns a (H, W) excitation 
    array at a given time `t`.
    
    Attributes:
        name (str): Human-readable name for the source.
    
    Notes:
        - All excitations are defined in pixel coordinates (resolution space).
        - Future refactor may shift to world (meter) coordinates.
    """

    def __init__(self, *, name: Optional[str] = None):
        if name is None:
            raise ValueError("All excitation sources must specify a `name`.")
        self.name = name

    @abstractmethod
    def __call__(self, t: float) -> np.ndarray:
        """
        Return a 2D excitation field of shape (H, W) at time t.
        """
        pass

    def get_controls(self):
        """Return a list of controls for the excitation source."""
        return []
