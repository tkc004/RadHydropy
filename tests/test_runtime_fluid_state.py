# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
from types import SimpleNamespace

import numpy as np
import pytest
import unyt

from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
from radhydropy.runtime_fields import (
    SUPERCOMOVING_RUNTIME_FIELDS,
    FluidRuntimeState,
    runtime_fields,
)
from radhydropy.state_boundaries import UnitBoundaryError
from radhydropy.units import CodeUnits


def test_supercomoving_fluid_runtime_state_has_explicit_fields():
    state = FluidRuntimeState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS,
        rho_comoving_code=np.array([1.0, 2.0]),
        vel_supercomoving_code=np.array([0.1, 0.2]),
        pre_supercomoving_code=np.array([0.3, 0.4]),
        temp_supercomoving_code=np.array([100.0, 200.0]),
        tau_supercomoving_code=0.5,
    )
    assert state.rho_comoving_code is not None
    assert state.vel_supercomoving_code is not None
    assert state.pre_supercomoving_code is not None
    assert state.temp_supercomoving_code is not None
    assert state.tau_supercomoving_code == 0.5  # noqa: PLR2004
    assert state.rho_proper_code is None


def test_fluid_setup_records_selected_runtime_state():
    par = SimpleNamespace(
        cosmological_expansion=False,
        coordinate_frame="physical",
        time_coordinate="proper",
        velocity_representation="proper",
    )
    assert runtime_fields(par).density == "rho_proper_code"


def test_set_fluid_time_accepts_one_element_clock_arrays():
    fluid = Fluid()

    fluid.SetFluidTime(np.array([1.25]))

    assert fluid.time_proper_code == 1.25  # noqa: PLR2004


def test_fluid_primitive_updates_reject_unconfigured_runtime_representation():
    fluid = Fluid()
    fluid.eos = EOS("polytropic", gamma=5.0 / 3.0)
    fluid.rho_code = np.ones(2)
    fluid.temp_code = np.ones(2)
    fluid.mu = np.ones(2)

    for method_name in ("SetPressure", "SetEnergyDensity", "SetSoundSpeed"):
        with pytest.raises(UnitBoundaryError, match="representation-specific"):
            getattr(fluid, method_name)()


def test_hydrogen_helium_mu_rejects_unconfigured_runtime_representation():
    fluid = Fluid()
    fluid.rho_code = np.ones(2)
    fluid.xHI = np.ones(2)
    fluid.xHeI = np.ones(2)
    fluid.xHeII = np.zeros(2)
    fluid.xHeIII = np.zeros(2)

    with pytest.raises(UnitBoundaryError, match="representation-specific"):
        fluid.SetHydrogenHeliumMu()


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
        },
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
        "rho_proper_code",
        "vel_proper_code",
        "pre_proper_code",
        "temp_proper_code",
        "time_proper_code",
    ):
        value = getattr(fluid.runtime_state, name)
        assert not hasattr(value, "units")

    for legacy_name in ("rho_code", "vel_code", "pre_code", "temp_code", "time_code"):
        assert not hasattr(fluid, legacy_name)
