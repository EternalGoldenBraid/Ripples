from abc import ABC, abstractmethod
from typing import Optional, Callable, final, Dict, Any
import numpy as np

from loguru import logger as log

class ExcitationSourceBase(ABC):
    """
    Base class for anything that injects energy into the ripple field.
    Do not override `_set_gain` directly; use `_set_gain` method instead.

    Parameters
    ----------
    nominal_peak : float
        An upper-bound estimate of the raw amplitude this source can emit
        **before** applying any user gain.  The visualiser uses the largest
        nominal_peak × gain across all sources to pick a global display
        range so that every ripple can be shown without re-scaling each
        frame.
    name : str
        Human-readable label (also shown in the UI group-box).
    """

    def __init__(self, *, 
                 nominal_peak: Optional[float],
                 name: Optional[str] = None,
                 ):
        if name is None:
            raise ValueError(
                "All excitation sources must specify a `name`.")

        if nominal_peak is None:
            raise ValueError(
                "All excitation sources must specify a `nominal_peak`.")

        self.nominal_peak = nominal_peak
        self.name = name
        self._subscribers: dict[str, list[Callable]] = {}
        self.gain: float = 1.0
        self._log_debug = False

    @abstractmethod
    def __call__(self, t: float) -> Dict[str, Any]:
        """
        Return a dictionary that contains:
            'excitation': 2D array of the excitation field at time `t`.
            'overlay': Optional 2D array for visual overlays (e.g. masks).
        """
        pass

    def subscribe(self, event: str, fn):
        """
        Register **fn** as a listener for *event*.

        Parameters
        ----------
        event : str
            Name of the event (e.g. ``"gain_changed"``).
        fn : Callable
            Callback invoked as ``fn(*args, **kwargs)`` whenever the event is
            emitted.  Callbacks are stored as weak-references so they do **not**
            keep GUI widgets alive.
        """
        self._subscribers.setdefault(event, []).append(fn)

    def _emit(self, event: str, *args, **kw):
        """Internal: invoke all callbacks previously registered for *event*."""
        for fn in self._subscribers.get(event, []):
            fn(*args, **kw)

    def get_controls(self):
        """Return a list of controls for the excitation source."""
        return []

    @final
    def _set_gain(self, value: float):
            self.gain = value
            self._emit("gain_changed", self.gain)
