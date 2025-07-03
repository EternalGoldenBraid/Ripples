from typing import Union, TYPE_CHECKING, Any, List, Dict

if TYPE_CHECKING:
    import numpy
    try:
        import cupy
        ArrayType = Union[numpy.ndarray, cupy.ndarray]
    except ImportError:
        ArrayType = numpy.ndarray
else:
    ArrayType = Any
