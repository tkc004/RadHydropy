"""NumPy array subclass that allows solver-side scratch attributes."""

import numpy as np


class NamedArray(np.ndarray):
    """A plain NumPy array that can carry extra attributes."""

    def __new__(cls, input_array, unit=None, dtype=float, copy=False, **kwargs):
        obj = np.array(input_array, dtype=dtype, copy=copy).view(cls)
        return obj

    def __array_finalize__(self, obj):
        # NumPy calls this when new views are created; we intentionally keep
        # any existing scratch attributes attached to the view.
        if obj is None:
            return

    def __deepcopy__(self, memo):
        """Return a clean copy without recursive solver scratch metadata."""
        copied = np.array(self, copy=True).view(type(self))
        memo[id(self)] = copied
        return copied


def as_named_array(value, dtype=float):
    """Return ``value`` as a mutable NumPy subclass with attribute support."""
    if isinstance(value, NamedArray):
        return value
    return NamedArray(value, dtype=dtype, copy=False)
