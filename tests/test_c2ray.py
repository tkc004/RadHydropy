from types import SimpleNamespace

import numpy as np

from radhydropy.thermo_networks import c2ray
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


def make_state(ncell=4):
    return {
        "boundary_cm": np.arange(ncell + 1, dtype=float),
        "volume_cm3": np.ones(ncell, dtype=float),
        "rho_g_cm3": np.ones(ncell, dtype=float) * 1.0e-24,
        "temperature_K": np.ones(ncell, dtype=float) * 1.0e4,
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

    assert np.all(result.converged)
    assert np.all(result.iterations >= 1)
    assert result.photon_density.shape == (1, 4)
    assert np.all(result.photon_density[0, :-1] >= result.photon_density[0, 1:])
    assert state["xHI"][0] <= state["xHI"][-1]

    source_rate = par.radiative_transfer_boundary_flux
    absorbed_rate = np.sum(result.absorbed_photon_rate * state["volume_cm3"])
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
    np.testing.assert_allclose(state["ngamma_cm3"], result.photon_density[0])
