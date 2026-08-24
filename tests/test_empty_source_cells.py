import numpy as np

from radhydropy.thermo_networks.hydrogen import (
    _apply_compton_only_source,
    _fast_apply_thermal_source,
)


def test_explicit_thermal_source_skips_empty_cells():
    state = {
        "active": np.array([True, False]),
        "rho_g_cm3": np.array([2.0, 0.0]),
        "specific_total_energy_erg_g": np.array([10.0, 20.0]),
        "specific_kinetic_energy_erg_g": np.zeros(2),
        "gamma": 5.0 / 3.0,
        "mu": np.ones(2),
    }

    _fast_apply_thermal_source(state, np.array([4.0, 9.0]), 0.5)

    np.testing.assert_allclose(state["specific_total_energy_erg_g"], [11.0, 20.0])


def test_compton_source_skips_empty_cells():
    state = {
        "active": np.array([True, False]),
        "rho_g_cm3": np.array([1.0, 0.0]),
        "temperature_K": np.array([1.0, 42.0]),
        "xHI": np.array([0.0, 0.0]),
        "gamma": 5.0 / 3.0,
        "mu": np.ones(2),
        "hydrogen_mass_fraction": 1.0,
        "cmb_temperature_0_K": 2.7255,
        "compton_cmb_redshift": 0.0,
        "specific_kinetic_energy_erg_g": np.zeros(2),
    }

    _apply_compton_only_source(state, 1.0e12)

    assert np.isclose(state["temperature_K"][1], 42.0)
    assert np.isclose(state["specific_total_energy_erg_g"][1], 42.0 / (
        (5.0 / 3.0 - 1.0) * 1.0 * 1.67262192369e-24
    ) * 1.380649e-16)
