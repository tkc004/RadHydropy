from types import SimpleNamespace

import numpy as np
import unyt

from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
from radhydropy.units import CodeUnits

from radhydropy.runtime_fields import (
    FluidRuntimeState,
    SUPERCOMOVING_RUNTIME_FIELDS,
    runtime_fields,
)


def test_supercomoving_fluid_runtime_state_has_explicit_fields():
    state = FluidRuntimeState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS,
        density=np.array([1.0, 2.0]),
        velocity=np.array([0.1, 0.2]),
        pressure=np.array([0.3, 0.4]),
        temperature=np.array([100.0, 200.0]),
        time=0.5,
    )
    assert state.rho_comoving_code is not None
    assert state.vel_supercomoving_code is not None
    assert state.pre_supercomoving_code is not None
    assert state.temp_supercomoving_code is not None
    assert state.tau_supercomoving_code == 0.5
    assert state.rho_proper_code is None


def test_fluid_setup_records_selected_runtime_state():
    par = SimpleNamespace(
        cosmological_expansion=False,
        coordinate_frame="physical",
        time_coordinate="proper",
        velocity_representation="proper",
    )
    assert runtime_fields(par).density == "rho_proper_code"


def test_fluid_setup_converts_physical_builder_arrays_before_runtime_state():
    code_units = CodeUnits.from_mapping(
        {
            "name": "runtime-builder-test",
            "InternalUnitSystem": {
                "UnitMass_in_cgs": 1.0,
                "UnitLength_in_cgs": 1.0,
                "UnitVelocity_in_cgs": 1.0,
                "UnitCurrent_in_cgs": 1.0,
                "UnitTemp_in_cgs": 1.0,
            },
        }
    )
    par = SimpleNamespace(
        CodeUnits=code_units,
        units=SimpleNamespace(CodeUnits=code_units),
        cosmological_expansion=False,
        coordinate_frame="physical",
        time_coordinate="proper",
        velocity_representation="proper",
        noghost=1,
        mesh=SimpleNamespace(ghost_cells=1, grid_cells=2),
        hydrogen_chemistry=False,
        gas_angular_momentum=False,
    )
    fluid = Fluid()
    fluid.eos = EOS("polytropic", gamma=5.0 / 3.0, code_units=code_units)
    fluid.rho_proper_code = np.ones(2) * unyt.g / unyt.cm**3
    fluid.vel_proper_code = np.zeros(2) * unyt.cm / unyt.s
    fluid.temp_proper_code = np.ones(2) * unyt.K
    fluid.mu = np.ones(2)

    fluid.SetUpFluid(par)

    for name in (
        "rho_proper_code", "vel_proper_code", "pre_proper_code",
        "temp_proper_code", "time_proper_code",
    ):
        value = getattr(fluid.runtime_state, name)
        assert not hasattr(value, "units")

    for legacy_name in ("rho_code", "vel_code", "pre_code", "temp_code", "time_code"):
        assert not hasattr(fluid, legacy_name)
