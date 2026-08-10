import numpy as np

from radhydropy.thermo_networks.compton import cmb_compton_rate
from radhydropy.thermo_networks.hydrogen import thermal_rate
from radhydropy.thermo_networks.hydrogen_helium import _rates
from radhydropy.constants import PROTON_MASS_CGS


def test_cmb_compton_source_is_opt_in_and_has_expected_sign():
    temperature = np.array([1.0, 1.0e4])
    electrons = np.ones(2)

    disabled = cmb_compton_rate(temperature, electrons, redshift=10.0)
    enabled = cmb_compton_rate(
        temperature,
        electrons,
        enabled=True,
        redshift=10.0,
    )

    assert np.all(disabled == 0.0)
    assert enabled[0] > 0.0
    assert enabled[1] < 0.0


def test_hydrogen_thermal_rate_includes_optional_compton_source():
    state = {
        "rho_g_cm3": np.array([PROTON_MASS_CGS]),
        "temperature_K": np.array([1.0]),
        "xHI": np.array([0.0]),
        "hydrogen_mass_fraction": 1.0,
        "recombination": False,
        "collisional_ionization": False,
        "sigma_gamma_cm2": 0.0,
        "epsilon_gamma_erg": 0.0,
        "compton_cmb_enabled": True,
        "compton_cmb_redshift": 10.0,
        "cmb_temperature_0_K": 2.7255,
    }
    rate = thermal_rate(state, None)
    state["compton_cmb_enabled"] = False
    background_rate = thermal_rate(state, None)
    expected = cmb_compton_rate(
        state["temperature_K"],
        np.array([1.0]),
        enabled=True,
        redshift=10.0,
    )
    np.testing.assert_allclose(rate - background_rate, expected)


def test_hydrogen_helium_thermal_rate_uses_electron_density():
    state = {
        "rho_g_cm3": np.array([PROTON_MASS_CGS]),
        "hydrogen_mass_fraction": 0.7,
        "helium_mass_fraction": 0.28,
        "temperature_K": np.array([1.0]),
        "xHI": np.array([0.0]),
        "xHeI": np.array([0.0]),
        "xHeIII": np.array([1.0]),
        "sigma_gamma_cm2": {
            "HI": np.zeros(1), "HeI": np.zeros(1), "HeII": np.zeros(1)
        },
        "epsilon_gamma_erg": {
            "HI": np.zeros(1), "HeI": np.zeros(1), "HeII": np.zeros(1)
        },
        "compton_cmb_enabled": True,
        "compton_cmb_redshift": 10.0,
        "cmb_temperature_0_K": 2.7255,
    }
    ngamma = np.zeros((1, 1))
    _, _, _, rate = _rates(state, ngamma)
    expected_ne = state["ne_cm3"].copy()
    expected = cmb_compton_rate(
        state["temperature_K"],
        expected_ne,
        enabled=True,
        redshift=10.0,
    )
    state["compton_cmb_enabled"] = False
    _, _, _, background_rate = _rates(state, ngamma)
    np.testing.assert_allclose(rate - background_rate, expected)
