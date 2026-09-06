from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import unyt

from radhydropy.state_boundaries import (
    CgsSourceState,
    ProperCodeState,
    UnitBoundaryError,
    cgs_source_state_from_code,
    cgs_source_state_to_code,
    proper_code_state_from_physical,
)
from radhydropy.fluid import Fluid
from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS
from radhydropy.thermo_networks.hydrogen import source_state as hydrogen_source_state
from radhydropy.units import CodeUnits
from radhydropy.units import code_unit_scales
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
)


@pytest.fixture
def code_units():
    return CodeUnits.from_mapping(
        {
            "name": "boundary-test",
            "UnitMass_in_cgs": 2.0,
            "UnitLength_in_cgs": 3.0,
            "UnitVelocity_in_cgs": 4.0,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 5.0,
        }
    )


def test_physical_to_cgs_to_code_round_trip(code_units):
    physical = proper_code_state_from_physical(
        code_units=code_units,
        rho_unyt=np.array([2.0, 4.0]) * code_units.density_unit,
        vel_unyt=np.array([1.0, -2.0]) * code_units.velocity_unit,
        temp_unyt=np.array([3.0, 6.0]) * code_units.temperature_unit,
        specific_energy_unyt=np.array([7.0, 8.0]) * code_units.specific_energy_unit,
        pre_unyt=np.array([9.0, 10.0]) * code_units.pressure_unit,
        ngamma_unyt=np.array([11.0, 12.0]) * code_units.number_density_unit,
        xHI_dimensionless=np.array([0.2, 0.8]),
        time_unyt=13.0 * code_units.time_unit,
    )
    assert isinstance(physical, ProperCodeState)
    assert not hasattr(physical.rho_proper_code, "units")

    source = cgs_source_state_from_code(
        code_units=code_units,
        fluid=physical,
        boundary_code=np.array([0.0, 1.0, 2.0]),
        volume_code=np.array([1.0, 1.5]),
    )
    assert isinstance(source, CgsSourceState)
    assert source.rho_cgs_g_cm3[0] == pytest.approx(
        2.0 * code_unit_scales(code_units)["density_cgs_g_cm3"]
    )
    assert source.time_cgs_s == pytest.approx(13.0 * code_units.time_in_cgs)

    restored = cgs_source_state_to_code(code_units=code_units, source=source)
    np.testing.assert_allclose(restored.rho_proper_code, physical.rho_proper_code)
    np.testing.assert_allclose(restored.vel_proper_code, physical.vel_proper_code)
    np.testing.assert_allclose(restored.temp_proper_code, physical.temp_proper_code)
    np.testing.assert_allclose(
        restored.specific_energy_proper_code, physical.specific_energy_proper_code
    )
    np.testing.assert_allclose(restored.ngamma_code, physical.ngamma_code)


def test_code_state_rejects_unitful_values(code_units):
    with pytest.raises(UnitBoundaryError, match="unitless numeric code-unit"):
        ProperCodeState(
            rho_proper_code=np.ones(2) * unyt.g / unyt.cm**3,
            vel_proper_code=np.ones(2),
            temp_proper_code=np.ones(2),
        )


def test_physical_boundary_rejects_unitless_values(code_units):
    with pytest.raises(UnitBoundaryError, match="requires a physical unyt quantity"):
        proper_code_state_from_physical(
            code_units=code_units,
            rho_unyt=np.ones(2),
            vel_unyt=np.ones(2) * code_units.velocity_unit,
            temp_unyt=np.ones(2) * code_units.temperature_unit,
        )


def test_cgs_state_rejects_unyt_values():
    with pytest.raises(UnitBoundaryError, match="unitless numeric cgs"):
        CgsSourceState(
            boundary_cgs_cm=np.array([0.0, 1.0]) * unyt.cm,
            volume_cgs_cm3=np.ones(1),
            rho_cgs_g_cm3=np.ones(1),
            velocity_cgs_cm_s=np.ones(1),
            temperature_cgs_K=np.ones(1),
            specific_energy_cgs_erg_g=np.ones(1),
        )


def test_energy_and_mass_form_specific_energy(code_units):
    fluid = ProperCodeState(
        rho_proper_code=np.ones(1),
        vel_proper_code=np.zeros(1),
        temp_proper_code=np.ones(1),
        Mass_code=np.array([2.0]),
        Energy_code=np.array([8.0]),
    )
    source = cgs_source_state_from_code(
        code_units=code_units,
        fluid=fluid,
        boundary_code=np.array([0.0, 1.0]),
        volume_code=np.ones(1),
    )
    expected = 4.0 * code_units.unit_conversion["specific_energy_cgs_erg_g"]
    assert source.specific_energy_cgs_erg_g[0] == pytest.approx(expected)


def test_fluid_exposes_validated_runtime_state():
    fluid = Fluid()
    fluid.rho_proper_code = np.array([2.0])
    fluid.vel_proper_code = np.array([3.0])
    fluid.temp_proper_code = np.array([4.0])
    fluid.pre_proper_code = np.array([5.0])
    fluid.eth_code = np.array([6.0])
    fluid.mu = np.array([0.6])
    fluid.xHI = np.array([0.2])
    fluid.time_proper_code = 7.0
    fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        density=fluid.rho_proper_code,
        velocity=fluid.vel_proper_code,
        pressure=fluid.pre_proper_code,
        temperature=fluid.temp_proper_code,
        time=fluid.time_proper_code,
        mu=fluid.mu,
        xHI=fluid.xHI,
    )

    state = fluid.code_state

    assert isinstance(state, ProperCodeState)
    np.testing.assert_allclose(state.specific_energy_proper_code, [3.0])
    np.testing.assert_allclose(state.mu_dimensionless, [0.6])
    assert state.time_proper_code == pytest.approx(7.0)


def test_fluid_runtime_state_rejects_unitful_arrays():
    fluid = Fluid()
    fluid.rho_proper_code = np.array([2.0]) * unyt.g / unyt.cm**3
    fluid.vel_proper_code = np.array([3.0])
    fluid.temp_proper_code = np.array([4.0])
    fluid.pre_proper_code = np.array([1.0])
    fluid.mu = np.array([1.0])
    fluid.time_proper_code = 0.0
    fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    with pytest.raises(TypeError, match="unitless numeric fluid code value"):
        fluid.runtime_state = FluidRuntimeState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            density=fluid.rho_proper_code,
            velocity=fluid.vel_proper_code,
            pressure=fluid.pre_proper_code,
            temperature=fluid.temp_proper_code,
            time=fluid.time_proper_code,
            mu=fluid.mu,
        )


def test_hydrogen_source_state_uses_typed_cgs_boundary(code_units):
    fluid = Fluid()
    fluid.rho_proper_code = np.array([2.0])
    fluid.vel_proper_code = np.array([0.0])
    fluid.temp_proper_code = np.array([100.0])
    fluid.xHI = np.array([1.0])
    fluid.eos = SimpleNamespace(gamma=5.0 / 3.0)
    fluid.pre_proper_code = np.ones(1)
    fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        density=fluid.rho_proper_code,
        velocity=fluid.vel_proper_code,
        pressure=fluid.pre_proper_code,
        temperature=fluid.temp_proper_code,
        time=0.0,
        mu=np.ones(1),
        xHI=fluid.xHI,
    )

    mesh = SimpleNamespace(
        boundary=np.array([0.0, 1.0]),
        vol=np.array([1.0]),
        coordinate=np.array([0.5]),
    )
    mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        coordinate=mesh.coordinate,
        boundary=mesh.boundary,
        width=np.ones(1), area=np.ones(1), volume=mesh.vol,
    )
    par = SimpleNamespace(
        units=SimpleNamespace(CodeUnits=code_units),
        mesh=SimpleNamespace(ghost_cells=0, grid_cells=1),
        hydrogen_mass_fraction=1.0,
        supercomoving_coordinates=False,
    )

    with patch(
        "radhydropy.thermo_networks.hydrogen.cgs_source_state_from_code",
        wraps=cgs_source_state_from_code,
    ) as conversion:
        state = hydrogen_source_state(mesh, fluid, par)

    conversion.assert_called_once()
    np.testing.assert_allclose(
        state["rho_cgs_g_cm3"],
        [2.0 * code_unit_scales(code_units)["density_cgs_g_cm3"]],
    )
    np.testing.assert_allclose(
        state["temperature_cgs_K"],
        [100.0 * code_unit_scales(code_units)["temperature_cgs_K"]],
    )
    expected_specific_energy = (
        BOLTZMANN_CONSTANT_CGS * state["temperature_cgs_K"]
        / ((5.0 / 3.0 - 1.0) * PROTON_MASS_CGS)
    )
    np.testing.assert_allclose(
        state["specific_energy_cgs_erg_g"], expected_specific_energy
    )
