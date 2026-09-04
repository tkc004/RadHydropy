"""Tests for momentum deposition from absorbed radiation."""

from types import SimpleNamespace
from tests.parameter_fixtures import parameter_namespace

import numpy as np

from radhydropy.constants import SPEED_OF_LIGHT_CGS
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits


CODE_UNITS = CodeUnits.from_mapping(
    {
        "name": "radiation_pressure_test_units",
        "InternalUnitSystem": {
            "UnitMass_in_cgs": 1.0,
            "UnitLength_in_cgs": 1.0,
            "UnitVelocity_in_cgs": 1.0,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        },
    }
)


def make_state(rho):
    rho = np.asarray(rho, dtype=float)
    ncell = rho.size
    mesh = SimpleNamespace(
        coordsys="cartesian",
        boundary=np.arange(ncell + 1, dtype=float),
        vol=np.ones(ncell, dtype=float),
        coordinate=np.arange(ncell, dtype=float) + 0.5,
    )
    fluid = SimpleNamespace(
        rho_code=rho.copy(),
        vel_code=np.zeros(ncell, dtype=float),
        Mom_code=np.zeros(ncell, dtype=float),
        Energy_code=np.zeros(ncell, dtype=float),
    )
    par = parameter_namespace(
        CodeUnits=CODE_UNITS,
        noghost=0,
        nogrid=ncell,
        radiation_pressure=True,
        radiation_pressure_efficiency=1.0,
    )
    return mesh, fluid, par


def source_result(absorbed, energies, direction=1):
    return {
        "source_steps": 1,
        "absorbed_photon_rate": np.asarray(absorbed, dtype=float),
        "photon_energy_cgs_erg": np.asarray(energies, dtype=float),
        "direction": direction,
    }


def test_one_cell_radiation_pressure_matches_absorbed_momentum():
    mesh, fluid, par = make_state([2.0])
    absorbed = 3.0
    energy = 5.0
    dt = 7.0

    Solver().ApplyRadiationPressure(
        dt,
        mesh,
        fluid,
        par,
        source_result([absorbed], [energy]),
    )

    expected_force = absorbed * energy / SPEED_OF_LIGHT_CGS
    np.testing.assert_allclose(fluid.Mom_code, [expected_force * dt])
    np.testing.assert_allclose(fluid.Energy_code, [0.0])


def test_zero_density_cell_is_skipped():
    mesh, fluid, par = make_state([2.0, 0.0])
    fluid.Mom_code[:] = [1.0, 2.0]
    fluid.Energy_code[:] = [3.0, 4.0]
    before_mom = fluid.Mom_code.copy()
    before_energy = fluid.Energy_code.copy()

    Solver().ApplyRadiationPressure(
        1.0,
        mesh,
        fluid,
        par,
        source_result([1.0, 100.0], [5.0]),
    )

    np.testing.assert_allclose(fluid.Mom_code[1], before_mom[1])
    np.testing.assert_allclose(fluid.Energy_code[1], before_energy[1])


def test_radiation_pressure_direction_reverses_momentum():
    mesh, positive, par = make_state([1.0])
    _, negative, _ = make_state([1.0])
    result = source_result([2.0], [4.0])

    Solver().ApplyRadiationPressure(1.0, mesh, positive, par, result)
    Solver().ApplyRadiationPressure(
        1.0,
        mesh,
        negative,
        par,
        source_result([2.0], [4.0], direction=-1),
    )

    np.testing.assert_allclose(negative.Mom_code, -positive.Mom_code)


def test_multigroup_pressure_is_energy_weighted():
    mesh, fluid, par = make_state([1.0])
    absorbed = np.array([[2.0], [3.0]])
    energies = np.array([5.0, 7.0])

    Solver().ApplyRadiationPressure(
        1.0,
        mesh,
        fluid,
        par,
        source_result(absorbed, energies),
    )

    expected = (2.0 * 5.0 + 3.0 * 7.0) / SPEED_OF_LIGHT_CGS
    np.testing.assert_allclose(fluid.Mom_code, [expected])


def test_radiation_pressure_can_be_disabled():
    mesh, fluid, par = make_state([1.0])
    par.radiation_pressure = False
    fluid.Mom_code[:] = 2.0
    fluid.Energy_code[:] = 3.0

    Solver().ApplyRadiationPressure(
        1.0,
        mesh,
        fluid,
        par,
        source_result([10.0], [20.0]),
    )

    np.testing.assert_allclose(fluid.Mom_code, [2.0])
    np.testing.assert_allclose(fluid.Energy_code, [3.0])
