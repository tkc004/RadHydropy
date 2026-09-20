# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
from types import SimpleNamespace  # noqa: CPY001

import numpy as np
import pytest
import unyt

from radhydropy.arrays import NamedArray, as_named_array
from radhydropy.fluid import Fluid
from radhydropy.mesh import Mesh
from radhydropy.runtime_fields import (
    PROPER_RUNTIME_FIELDS,
    SUPERCOMOVING_RUNTIME_FIELDS,
    FluidRuntimeState,
    MeshGeometryState,
    require_runtime_fields,
    runtime_fields,
    select_fluid_primitive_arrays,
    select_mesh_geometry_arrays,
    validate_runtime_state_shapes,
)


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
        supercomoving_fluid,
        supercomoving_par,
    )
    supercomoving_geometry_arrays = select_mesh_geometry_arrays(
        supercomoving_geometry,
        supercomoving_par,
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
            ),
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


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("rho_proper_code", np.ones((2, 1)), "one-dimensional"),
        ("vel_proper_code", np.array([0.0, np.inf]), "finite"),
        ("pre_proper_code", np.array([0.0, np.nan]), "finite"),
    ],
)
def test_runtime_state_rejects_malformed_arrays(field, value, match):
    arrays = {
        "rho_proper_code": np.ones(2),
        "vel_proper_code": np.ones(2),
        "pre_proper_code": np.ones(2),
        "temp_proper_code": np.ones(2),
        "time_proper_code": 0.0,
    }
    arrays[field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        FluidRuntimeState.from_arrays(PROPER_RUNTIME_FIELDS, **arrays)


def test_runtime_state_rejects_non_scalar_clock():
    with pytest.raises(ValueError, match="must be scalar"):
        FluidRuntimeState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            rho_proper_code=np.ones(2),
            vel_proper_code=np.ones(2),
            pre_proper_code=np.ones(2),
            temp_proper_code=np.ones(2),
            time_proper_code=np.zeros(2),
        )


def test_runtime_state_rejects_mismatched_fluid_array_lengths():
    with pytest.raises(ValueError, match="matching lengths"):
        FluidRuntimeState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            rho_proper_code=np.ones(2),
            vel_proper_code=np.ones(3),
            pre_proper_code=np.ones(2),
            temp_proper_code=np.ones(2),
            time_proper_code=0.0,
        )


def test_mesh_state_rejects_boundary_length_mismatch():
    with pytest.raises(ValueError, match="one more entry"):
        MeshGeometryState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            x_proper_code=np.ones(2),
            boundary_proper_code=np.ones(2),
            width_proper_code=np.ones(2),
            area_proper_code=np.ones(2),
            volume_proper_code=np.ones(2),
        )


def test_require_runtime_fields_rejects_present_but_uninitialized_values():
    owner = SimpleNamespace(
        x_proper_code=np.ones(1),
        boundary_proper_code=np.ones(2),
        width_proper_code=np.ones(1),
        area_proper_code=np.ones(1),
        volume_proper_code=np.ones(1),
        rho_proper_code=np.ones(1),
        vel_proper_code=np.ones(1),
        pre_proper_code=None,
        temp_proper_code=np.ones(1),
        time_proper_code=0.0,
    )
    with pytest.raises(ValueError, match="pre_proper_code"):
        require_runtime_fields(owner, PROPER_RUNTIME_FIELDS, "runtime")


def test_mesh_and_fluid_runtime_states_require_matching_cell_counts():
    geometry = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=np.ones(2),
        boundary_proper_code=np.arange(3),
        width_proper_code=np.ones(2),
        area_proper_code=np.ones(2),
        volume_proper_code=np.ones(2),
    )
    fluid = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        rho_proper_code=np.ones(3),
        vel_proper_code=np.ones(3),
        pre_proper_code=np.ones(3),
        temp_proper_code=np.ones(3),
        time_proper_code=0.0,
    )
    with pytest.raises(ValueError, match="cell counts disagree"):
        validate_runtime_state_shapes(geometry, fluid, PROPER_RUNTIME_FIELDS)


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
