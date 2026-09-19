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
    select_fluid_primitive_arrays,
    select_mesh_geometry_arrays,
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


def test_runtime_selectors_choose_matching_proper_and_supercomoving_arrays():
    proper_par = SimpleNamespace(cosmological_expansion=False)
    proper_fluid = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        rho_proper_code=np.array([1.0]),
        vel_proper_code=np.array([2.0]),
        pre_proper_code=np.array([3.0]),
        temp_proper_code=np.array([4.0]),
        time_proper_code=5.0,
    )
    proper_geometry = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=np.array([1.0]),
        boundary_proper_code=np.array([0.0, 2.0]),
        width_proper_code=np.array([2.0]),
        area_proper_code=np.array([3.0]),
        volume_proper_code=np.array([4.0]),
    )
    proper_arrays = select_fluid_primitive_arrays(proper_fluid, proper_par)
    proper_geometry_arrays = select_mesh_geometry_arrays(proper_geometry, proper_par)
    assert [array[0] for array in proper_arrays[:4]] == [1.0, 2.0, 3.0, 4.0]
    assert [array[0] for array in proper_geometry_arrays] == [1.0, 0.0, 2.0, 3.0, 4.0]
    assert proper_arrays[4] == 5.0

    supercomoving_par = _cosmological_par()
    supercomoving_fluid = FluidRuntimeState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS,
        rho_comoving_code=np.array([6.0]),
        vel_supercomoving_code=np.array([7.0]),
        pre_supercomoving_code=np.array([8.0]),
        temp_supercomoving_code=np.array([9.0]),
        tau_supercomoving_code=10.0,
    )
    supercomoving_geometry = MeshGeometryState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS,
        x_comoving_code=np.array([6.0]),
        boundary_comoving_code=np.array([5.0, 7.0]),
        width_comoving_code=np.array([2.0]),
        area_comoving_code=np.array([8.0]),
        volume_comoving_code=np.array([9.0]),
    )
    supercomoving_arrays = select_fluid_primitive_arrays(
        supercomoving_fluid, supercomoving_par
    )
    supercomoving_geometry_arrays = select_mesh_geometry_arrays(
        supercomoving_geometry, supercomoving_par
    )
    assert [array[0] for array in supercomoving_arrays[:4]] == [6.0, 7.0, 8.0, 9.0]
    assert [array[0] for array in supercomoving_geometry_arrays] == [6.0, 5.0, 2.0, 8.0, 9.0]
    assert supercomoving_arrays[4] == 10.0


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
