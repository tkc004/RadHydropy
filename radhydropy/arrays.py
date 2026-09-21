# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""NumPy array subclass that allows solver-side scratch attributes."""

from typing import Any, cast

import numpy as np


class NamedArray(np.ndarray[Any, np.dtype[Any]]):
    """A plain NumPy array that can carry extra attributes."""

    def __new__(  # noqa: PYI034
        cls,
        input_array: Any,
        unit: Any = None,
        dtype: Any = float,
        *,
        copy: bool = False,
        **kwargs: Any,
    ) -> "NamedArray":
        del unit, kwargs
        if copy:
            arr = np.array(input_array, dtype=dtype, copy=True)
        else:
            arr = np.asarray(input_array, dtype=dtype)
        return cast("NamedArray", arr.view(cls))

    def __array_finalize__(self, obj: Any) -> None:
        # NumPy calls this when new views are created; we intentionally keep
        # any existing scratch attributes attached to the view.
        if obj is None:
            return

    def __deepcopy__(self, memo: dict[int, Any] | None) -> "NamedArray":
        """Return a clean copy without recursive solver scratch metadata."""
        copied = np.array(self, copy=True).view(type(self))
        if memo is None:
            memo = {}
        memo[id(self)] = copied
        return cast("NamedArray", copied)


def as_named_array(value: Any, dtype: Any = float) -> NamedArray:
    """Return ``value`` as a mutable NumPy subclass with attribute support."""
    if hasattr(value, "units") or hasattr(value, "to_value"):
        raise TypeError(
            "runtime arrays must be unitless code-unit values; "
            "convert physical quantities at the input boundary",
        )
    if isinstance(value, NamedArray):
        return value
    return NamedArray(value, dtype=dtype, copy=False)
