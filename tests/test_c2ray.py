from pathlib import Path
from types import SimpleNamespace
from tests.parameter_fixtures import parameter_namespace

import numpy as np

from radhydropy.thermo_networks import c2ray
from radhydropy.thermo_networks.pie import MetalPIETable
from radhydropy.units import CodeUnits


CODE_UNITS = CodeUnits.from_mapping(
    {
        "name": "c2ray_test_units",
        "InternalUnitSystem": {
            "UnitMass_in_cgs": 1.0,
            "UnitLength_in_cgs": 1.0,
            "UnitVelocity_in_cgs": 1.0,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        },
    }
)
METAL_PIE_TABLE = (
    Path(__file__).resolve().parents[2]
    / "metal_pie_table"
    / "metal_pie_table_Z1_metals.h5"
)


def make_state(ncell=4):
    return {
        "boundary_cgs_cm": np.arange(ncell + 1, dtype=float),
        "volume_cgs_cm3": np.ones(ncell, dtype=float),
        "rho_cgs_g_cm3": np.ones(ncell, dtype=float) * 1.0e-24,
        "temperature_cgs_K": np.ones(ncell, dtype=float) * 1.0e4,
        "xHI": np.ones(ncell, dtype=float),
        "hydrogen_mass_fraction": 1.0,
        "recombination": False,
        "collisional_ionization": False,
        "thermal_coupling": False,
    }


def make_par():
    return SimpleNamespace(
        CodeUnits=CODE_UNITS,
        coordsys="cartesian",
        thermochemistry_network="hydrogen",
        hydrogen_mass_fraction=1.0,
        hydrogen_sigma_gamma=1.0e-18,
        hydrogen_epsilon_gamma=0.0,
        radiative_transfer_boundary_flux=1.0e18,
        radiative_transfer_source_photon_rate=0.0,
        radiative_transfer_direction=1,
        radiative_transfer_c2ray_max_iterations=32,
        radiative_transfer_c2ray_tolerance=1.0e-8,
        radiative_transfer_c2ray_relaxation=1.0,
        radiative_transfer_c2ray_nonconvergence="raise",
    )


def test_c2ray_is_causal_and_photon_conserving():
    state = make_state()
    par = make_par()
    result = c2ray.advance_state(state, par, 1.0e8)

    assert np.all(np.isfinite(result.photon_density))
    assert np.all(result.iterations >= 1)
    assert result.photon_density.shape == (1, 4)
    assert np.all(result.photon_density[0, :-1] >= result.photon_density[0, 1:])
    assert state["xHI"][0] <= state["xHI"][-1]

    source_rate = par.radiative_transfer_boundary_flux
    absorbed_rate = np.sum(result.absorbed_photon_rate * state["volume_cgs_cm3"])
    incoming_rate = source_rate
    outgoing_rate = np.sum(result.outgoing_photon_rate)
    assert absorbed_rate > 0.0
    np.testing.assert_allclose(incoming_rate, absorbed_rate + outgoing_rate)


def test_c2ray_initial_trace_does_not_change_chemistry():
    state = make_state()
    par = make_par()
    initial = state["xHI"].copy()
    result = c2ray.trace_initial_state(state, par)

    np.testing.assert_array_equal(state["xHI"], initial)
    np.testing.assert_allclose(state["ngamma_cgs_cm3"], result.photon_density[0])


def test_c2ray_hydrogen_helium_uses_coupled_local_solver():
    state = make_state(ncell=3)
    state.update(
        {
            "thermochemistry_network": "hydrogen_helium",
            "hydrogen_mass_fraction": 0.75,
            "helium_mass_fraction": 0.25,
            "xHI": np.ones(3),
            "xHeI": np.ones(3),
            "xHeIII": np.zeros(3),
            "specific_energy_cgs_erg_g": np.ones(3) * 2.0e12,
            "gamma": 5.0 / 3.0,
            "mu": np.ones(3),
            "sigma_gamma_cgs_cm2": {
                "HI": np.full(5, 1.0e-18),
                "HeI": np.full(5, 2.0e-18),
                "HeII": np.array([0.0, 0.0, 0.0, 1.0e-18, 1.0e-18]),
            },
            "epsilon_gamma_cgs_erg": {
                "HI": np.full(5, 1.0e-11),
                "HeI": np.full(5, 1.0e-11),
                "HeII": np.array([0.0, 0.0, 0.0, 1.0e-11, 1.0e-11]),
            },
            "thermal_coupling": True,
        }
    )
    par = make_par()
    par.thermochemistry_network = "hydrogen_helium"
    par.hydrogen_mass_fraction = 0.75
    par.radiation_group_sigma_gamma = np.full(5, 1.0e-18)
    par.radiation_group_epsilon_gamma = np.full(5, 1.0e-11)
    par.radiation_group_sigma_gamma_HeI = np.full(5, 2.0e-18)
    par.radiation_group_sigma_gamma_HeII = np.array([0.0, 0.0, 0.0, 1.0e-18, 1.0e-18])
    par.radiation_group_epsilon_gamma_HeI = np.full(5, 1.0e-11)
    par.radiation_group_epsilon_gamma_HeII = np.zeros(5)
    par.radiative_transfer_c2ray_nonconvergence = "ignore"
    par.radiative_transfer_c2ray_max_iterations = 128

    result = c2ray.advance_state(state, par, 1.0e6)

    assert np.all(np.isfinite(result.photon_density))
    assert np.all(result.iterations >= 1)
    assert np.all(np.isfinite(state["temperature_cgs_K"]))
    assert np.any(state["xHI"] < 1.0)
    assert np.any(state["xHeI"] < 1.0)
    assert np.any(state["xHeIII"] > 0.0)
    incoming_rate = par.radiative_transfer_boundary_flux * 5
    absorbed_rate = np.sum(result.absorbed_photon_rate * state["volume_cgs_cm3"])
    outgoing_rate = np.sum(result.outgoing_photon_rate)
    np.testing.assert_allclose(incoming_rate, absorbed_rate + outgoing_rate)


def test_c2ray_hydrogen_helium_pie_enters_implicit_thermal_rate():
    table = MetalPIETable(METAL_PIE_TABLE)
    state = make_state(ncell=1)
    state.update(
        {
            "hydrogen_mass_fraction": 0.75,
            "helium_mass_fraction": 0.25,
            "xHI": np.array([0.5]),
            "xHeI": np.array([0.25]),
            "xHeIII": np.array([0.25]),
            "specific_energy_cgs_erg_g": np.array([2.0e12]),
            "gamma": 5.0 / 3.0,
            "mu": np.ones(1),
            "sigma_gamma_cgs_cm2": {
                "HI": np.array([1.0e-18, 1.0e-18]),
                "HeI": np.array([2.0e-18, 2.0e-18]),
                "HeII": np.array([0.0, 1.0e-18]),
            },
            "epsilon_gamma_cgs_erg": {
                "HI": np.array([1.0e-11, 1.0e-11]),
                "HeI": np.array([1.0e-11, 1.0e-11]),
                "HeII": np.array([0.0, 1.0e-11]),
            },
            "metal_pie_table": table,
            "metallicity": 1.0,
        }
    )
    photon_density = np.array([[1.0e-4], [2.0e-4]])
    with_pie = c2ray._hhe_cell_state(state, 0)
    without_pie = c2ray._hhe_cell_state(state, 0)
    without_pie["metal_pie_table"] = None
    pie_derivative = c2ray._hhe_derivative(with_pie, photon_density)
    hhe_derivative = c2ray._hhe_derivative(without_pie, photon_density)

    # The ionization derivatives are unchanged by the metal-only table, while
    # the implicit thermal derivative contains its net heating/cooling rate.
    np.testing.assert_allclose(pie_derivative[:3], hhe_derivative[:3])
    assert not np.isclose(pie_derivative[3], hhe_derivative[3])
