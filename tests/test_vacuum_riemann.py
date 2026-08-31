import numpy as np
from types import SimpleNamespace
from tests.parameter_fixtures import parameter_namespace

from radhydropy.eos import EOS
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits
from radhydropy.utils import CalFluxFromLR


CODE_UNITS = CodeUnits.from_mapping(
    {
        "name": "vacuum_test_units",
        "InternalUnitSystem": {
            "UnitMass_in_cgs": 1.0,
            "UnitLength_in_cgs": 1.0,
            "UnitVelocity_in_cgs": 1.0,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        },
    }
)


def test_vacuum_face_state_is_zeroed_without_flooring_density():
    rho, vel, pre = Solver._vacuum_safe_primitive_state(
        np.array([1.0, 0.0, -1.0, np.nan]),
        np.array([2.0, 3.0, np.nan, 4.0]),
        np.array([5.0, -2.0, 3.0, np.nan]),
    )

    np.testing.assert_array_equal(rho, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(vel, [2.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(pre, [5.0, 0.0, 0.0, 0.0])


def test_rusanov_flux_between_gas_and_vacuum_is_finite_and_positive():
    flux = CalFluxFromLR(
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.0]),
        gamma=5.0 / 3.0,
        cmax=np.array([2.0]),
    )

    mass_flux, momentum_flux, energy_flux = flux
    assert np.all(np.isfinite(mass_flux))
    assert np.all(np.isfinite(momentum_flux))
    assert np.all(np.isfinite(energy_flux))
    assert mass_flux[0] >= 0.0
    assert energy_flux[0] >= 0.0


def test_primitive_reconstruction_stores_active_mask_for_vacuum_cells():
    mesh = SimpleNamespace(vol=np.ones(3), coordsys="cartesian")
    fluid = SimpleNamespace(
        Mass=np.array([1.0, 0.0, 2.0]),
        Mom=np.array([1.0, 5.0, 0.0]),
        Energy=np.array([2.0, 7.0, 3.0]),
        eos=EOS("polytropic", gamma=5.0 / 3.0, code_units=CODE_UNITS),
    )

    Solver().SetPrimitive(mesh, fluid)

    np.testing.assert_array_equal(fluid.active, [True, False, True])
    np.testing.assert_array_equal(fluid.rho, [1.0, 0.0, 2.0])
    np.testing.assert_array_equal(fluid.vel, [1.0, 0.0, 0.0])
    assert fluid.pre[1] == 0.0


def test_low_density_active_cell_blocks_both_interface_fluxes():
    par = parameter_namespace(noghost=1, nogrid=3, cfl_density_floor=1.0e-9)
    fluid = SimpleNamespace(
        rho=np.array([1.0, 1.0, 1.0e-12, 1.0e-12, 1.0]),
        Mass=SimpleNamespace(flux=np.ones(5)),
        Mom=SimpleNamespace(flux=np.ones(5) * 2.0),
        Energy=SimpleNamespace(flux=np.ones(5) * 3.0),
    )

    Solver()._apply_low_density_flux_mask(fluid, par)

    np.testing.assert_array_equal(fluid.Mass.flux, [1.0, 1.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(fluid.Mom.flux, [2.0, 2.0, 2.0, 0.0, 2.0])
    np.testing.assert_array_equal(fluid.Energy.flux, [3.0, 3.0, 3.0, 0.0, 3.0])
