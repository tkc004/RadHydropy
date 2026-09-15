from types import SimpleNamespace

import pytest
import numpy as np
import unyt

from radhydropy.fluid import Fluid
from radhydropy.mesh import Mesh
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
    SUPERCOMOVING_RUNTIME_FIELDS,
    runtime_fields,
)
from radhydropy.arrays import as_named_array
from radhydropy.arrays import NamedArray


def _cosmological_par():
    return SimpleNamespace(
        cosmological_expansion=True,
        coordinate_frame="comoving",
        time_coordinate="supercomoving",
        velocity_representation="supercomoving_peculiar",
    )


def test_runtime_fields_selects_explicit_cosmological_names():
    fields = runtime_fields(_cosmological_par())
    assert fields is SUPERCOMOVING_RUNTIME_FIELDS
    assert fields.density == "rho_comoving_code"
    assert fields.velocity == "vel_supercomoving_code"
    assert fields.time == "tau_supercomoving_code"


def test_runtime_fields_selects_explicit_proper_names():
    fields = runtime_fields(SimpleNamespace(cosmological_expansion=False))
    assert fields is PROPER_RUNTIME_FIELDS
    assert fields.density == "rho_proper_code"
    assert fields.velocity == "vel_proper_code"
    assert fields.time == "time_proper_code"


def test_cosmological_runtime_rejects_incomplete_representation():
    with pytest.raises(ValueError, match="coordinate_frame"):
        runtime_fields(
            SimpleNamespace(
                cosmological_expansion=True,
                coordinate_frame="physical",
                time_coordinate="cosmic",
                velocity_representation="physical",
            )
        )


def test_containers_start_without_legacy_runtime_contract():
    assert Fluid().runtime_fields is None
    assert Mesh().runtime_fields is None


def test_runtime_arrays_reject_unitful_values():
    with pytest.raises(TypeError, match="unitless"):
        as_named_array(np.ones(2) * unyt.cm)
    with pytest.raises(TypeError, match="unitless"):
        MeshGeometryState(x_proper_code=np.ones(2) * unyt.cm)
    with pytest.raises(TypeError, match="unitless"):
        FluidRuntimeState(rho_proper_code=np.ones(2) * unyt.g / unyt.cm**3)


def test_named_array_copy_false_allows_required_dtype_copy():
    source = np.array([1, 2], dtype=np.int64)
    result = NamedArray(source, dtype=float, copy=False)

    assert isinstance(result, NamedArray)
    assert result.dtype == np.dtype(float)
    np.testing.assert_array_equal(result, source)


def test_named_array_copy_true_is_independent():
    source = np.array([1.0, 2.0])
    result = NamedArray(source, copy=True)

    result[0] = 9.0
    assert source[0] == 1.0
