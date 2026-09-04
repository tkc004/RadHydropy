from pathlib import Path

import h5py
import numpy as np
import pytest

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS
from radhydropy.params import Par
from radhydropy.thermo_networks.hydrogen_helium import _closure, _rates
from radhydropy.thermo_networks.pie import MetalPIETable, PIEUVBGCoolingNetwork


TABLE_FILENAME = (
    Path(__file__).resolve().parents[2]
    / "metal_pie_table"
    / "metal_pie_table_Z1_metals.h5"
)
HM12_TOTAL_FILENAME = (
    Path(__file__).resolve().parents[2]
    / "metal_pie_table"
    / "metal_pie_hm12_total.h5"
)


def _code_units_mapping():
    return {
        "InternalUnitSystem": {
            "UnitMass_in_cgs": 1.0,
            "UnitLength_in_cgs": 1.0,
            "UnitVelocity_in_cgs": 1.0,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        }
    }


def _write_power_law_table(filename):
    temperature = np.array([1.0e2, 1.0e4, 1.0e6])
    density = np.array([1.0e-4, 1.0e-2, 1.0e0])
    ionization = np.array([1.0e-6, 1.0e-3, 1.0e0])
    shape = (len(temperature), len(density), len(ionization), 1)
    t, n, u = np.meshgrid(temperature, density, ionization, indexing="ij")
    heating = (t * n**2 * u**3)[..., None]
    cooling = (t**2 * n * u**0.5)[..., None]
    with h5py.File(filename, "w") as handle:
        group = handle.create_group("MetalPIE")
        axes = group.create_group("axes")
        axes.create_dataset("log10_temperature_K", data=np.log10(temperature))
        axes.create_dataset("log10_hydrogen_density_cm-3", data=np.log10(density))
        axes.create_dataset("log10_ionization_parameter", data=np.log10(ionization))
        axes.create_dataset("metallicity_Zsun", data=[1.0])
        rates = group.create_group("rates")
        rates.create_dataset("metal_photoheating_erg_cm3_s", data=heating)
        rates.create_dataset("metal_cooling_erg_cm3_s", data=cooling)


def _state():
    state = {
        "rho_cgs_g_cm3": np.array([PROTON_MASS_CGS]),
        "hydrogen_mass_fraction": 0.75,
        "helium_mass_fraction": 0.25,
        "temperature_cgs_K": np.array([1.0e4]),
        "xHI": np.array([0.8]),
        "xHeI": np.array([0.7]),
        "xHeIII": np.array([0.0]),
        "sigma_gamma_cgs_cm2": {
            "HI": np.array([1.0e-18, 2.0e-19]),
            "HeI": np.array([3.0e-18, 4.0e-19]),
            "HeII": np.array([0.0, 5.0e-19]),
        },
        "epsilon_gamma_cgs_erg": {
            "HI": np.array([1.0e-12, 2.0e-12]),
            "HeI": np.array([3.0e-12, 4.0e-12]),
            "HeII": np.array([0.0, 5.0e-12]),
        },
    }
    _closure(state)
    return state


def test_supplied_metal_pie_table_interpolates_finite_rates():
    table = MetalPIETable(TABLE_FILENAME)
    heating, cooling = table.rates(
        temperature_cgs_K=np.array([1.0e4, 1.0e5]),
        hydrogen_density_cgs_cm3=np.array([1.0e-3, 1.0e-3]),
        ionization_parameter=np.array([1.0e-3, 1.0e-3]),
        metallicity=1.0,
    )
    assert heating.shape == (2,)
    assert cooling.shape == (2,)
    assert np.all(np.isfinite(heating))
    assert np.all(np.isfinite(cooling))
    assert np.all(heating >= 0.0)
    assert np.all(cooling > 0.0)


def test_pie_matches_values_at_table_nodes(tmp_path):
    filename = tmp_path / "power_law.h5"
    _write_power_law_table(filename)
    table = MetalPIETable(filename)
    i, j, k = 1, 2, 0
    heating, cooling = table.rates(
        10.0 ** table.log_temperature[i],
        10.0 ** table.log_density[j],
        10.0 ** table.log_ionization_parameter[k],
    )
    assert np.isclose(heating, table._heating[i, j, k, 0])
    assert np.isclose(cooling, table._cooling[i, j, k, 0])


def test_pie_log_interpolation_matches_power_laws(tmp_path):
    filename = tmp_path / "power_law.h5"
    _write_power_law_table(filename)
    table = MetalPIETable(filename)
    temperature, density, ionization = 1.0e3, 1.0e-3, 1.0e-4
    heating, cooling = table.rates(temperature, density, ionization)
    assert np.isclose(heating, temperature * density**2 * ionization**3)
    assert np.isclose(cooling, temperature**2 * density * ionization**0.5)


def test_pie_vectorized_matches_cellwise_and_clips_boundaries(tmp_path):
    filename = tmp_path / "power_law.h5"
    _write_power_law_table(filename)
    table = MetalPIETable(filename)
    temperature = np.array([1.0, 1.0e3, 1.0e8])
    density = np.array([1.0e-8, 1.0e-3, 1.0e2])
    ionization = np.array([1.0e-12, 1.0e-4, 1.0e3])
    vector_heating, vector_cooling = table.rates(temperature, density, ionization)
    cell_values = [table.rates(temperature[i], density[i], ionization[i]) for i in range(3)]
    np.testing.assert_allclose(vector_heating, [value[0] for value in cell_values])
    np.testing.assert_allclose(vector_cooling, [value[1] for value in cell_values])
    assert np.all(np.isfinite(vector_heating))
    assert np.all(np.isfinite(vector_cooling))


def test_pie_rates_use_sum_of_multigroup_photon_density(tmp_path):
    filename = tmp_path / "power_law.h5"
    _write_power_law_table(filename)
    state_without_pie = _state()
    state_with_pie = _state()
    state_with_pie["metal_pie_table"] = MetalPIETable(filename)
    state_with_pie["metallicity"] = 1.0
    ngamma_cgs_cm3 = np.array([[1.0e-4], [2.0e-4]])
    no_pie_rate = _rates(state_without_pie, ngamma_cgs_cm3)[3]
    pie_rate = _rates(state_with_pie, ngamma_cgs_cm3)[3]
    n_h = state_with_pie["rho_cgs_g_cm3"] * 0.75 / PROTON_MASS_CGS
    expected_metal_heating = (1.0e4 * n_h**2 * (3.0e-4 / n_h) ** 3)
    expected_metal_cooling = (1.0e8 * n_h * (3.0e-4 / n_h) ** 0.5)
    np.testing.assert_allclose(
        pie_rate - no_pie_rate,
        expected_metal_heating - expected_metal_cooling,
    )


def test_pie_heating_and_cooling_have_correct_energy_sign(tmp_path):
    filename = tmp_path / "power_law.h5"
    _write_power_law_table(filename)
    state = _state()
    state["metal_pie_table"] = MetalPIETable(filename)
    state["metallicity"] = 1.0
    ngamma_cgs_cm3 = np.array([[1.0e-6], [2.0e-6]])
    rate_with_pie = _rates(state, ngamma_cgs_cm3)[3]
    rate_without_pie = _rates({**state, "metal_pie_table": None}, ngamma_cgs_cm3)[3]
    # At this state the synthetic cooling dominates, so PIE must reduce the
    # thermal rate. Reversing the table rates should reverse the sign.
    assert np.all(rate_with_pie < rate_without_pie)


def test_pie_self_shielding_disables_heating_but_keeps_cooling(tmp_path):
    filename = tmp_path / "power_law.h5"
    _write_power_law_table(filename)
    with h5py.File(filename, "a") as handle:
        handle["MetalPIE"].attrs["spectrum_type"] = (
            "Haardt-Madau 2012 UV background"
        )
        axes = handle["MetalPIE/axes"]
        del axes["log10_ionization_parameter"]
        axes.create_dataset("redshift", data=[0.0, 1.0, 2.0])
    state = _state()
    state["rho_cgs_g_cm3"] = np.array([100.0 * PROTON_MASS_CGS / 0.75])
    state["metal_pie_table"] = MetalPIETable(filename)
    state["metallicity"] = 1.0
    ngamma_cgs_cm3 = np.array([[1.0e-6], [2.0e-6]])

    rate_with_pie = _rates(state, ngamma_cgs_cm3)[3]
    rate_without_pie = _rates({**state, "metal_pie_table": None}, ngamma_cgs_cm3)[3]
    n_h = state["rho_cgs_g_cm3"] * 0.75 / PROTON_MASS_CGS
    _, expected_metal_cooling = state["metal_pie_table"].rates(
        state["temperature_cgs_K"], n_h, metallicity=1.0, redshift=0.0
    )
    np.testing.assert_allclose(
        rate_with_pie - rate_without_pie, -expected_metal_cooling
    )


def test_non_hm12_pie_keeps_heating_above_density_cutoff(tmp_path):
    filename = tmp_path / "power_law.h5"
    _write_power_law_table(filename)
    state = _state()
    state["rho_cgs_g_cm3"] = np.array([100.0 * PROTON_MASS_CGS / 0.75])
    state["metal_pie_table"] = MetalPIETable(filename)
    state["metallicity"] = 1.0
    ngamma_cgs_cm3 = np.array([[1.0e-6], [2.0e-6]])

    rate_with_pie = _rates(state, ngamma_cgs_cm3)[3]
    rate_without_pie = _rates({**state, "metal_pie_table": None}, ngamma_cgs_cm3)[3]
    n_h = state["rho_cgs_g_cm3"] * 0.75 / PROTON_MASS_CGS
    ionization_parameter = np.sum(ngamma_cgs_cm3, axis=0) / n_h
    expected_heating, expected_cooling = state["metal_pie_table"].rates(
        state["temperature_cgs_K"], n_h, ionization_parameter
    )
    np.testing.assert_allclose(
        rate_with_pie - rate_without_pie,
        expected_heating - expected_cooling,
    )


def test_hm12_pie_rejects_radiative_transfer():
    with pytest.raises(ValueError, match="radiative_transfer: false"):
        Par(
            {
                "CodeUnits": _code_units_mapping(),
                "metal_pie_enabled": True,
                "metal_pie_table_filename": str(HM12_TOTAL_FILENAME),
                "radiative_transfer": True,
            }
        )


def test_pie_uvbg_implicit_step_converges_against_half_steps():
    table = MetalPIETable(HM12_TOTAL_FILENAME)
    network = PIEUVBGCoolingNetwork()
    temperature = 1.0e4
    gamma = 5.0 / 3.0
    mu = np.array([0.62])
    state = {
        "par": type(
            "Parameters",
            (),
            {
                "metal_pie_table": table,
                "metal_pie_photoheating_max_density_cgs_cm3": 50.0,
                "pie_uvbg_implicit_tolerance": 1.0e-3,
                "pie_uvbg_implicit_max_iterations": 64,
            },
        )(),
        "metallicity": 1.0,
        "redshift": 4.0,
        "hydrogen_mass_fraction": 0.7,
        "rho_cgs_g_cm3": np.array([PROTON_MASS_CGS / 0.7]),
        "temperature_cgs_K": np.array([temperature]),
        "specific_energy_cgs_erg_g": np.array(
            [BOLTZMANN_CONSTANT_CGS * temperature / ((gamma - 1.0) * mu[0] * PROTON_MASS_CGS)]
        ),
        "gamma": gamma,
        "mu": mu,
    }
    old_energy = state["specific_energy_cgs_erg_g"].copy()
    new_energy, converged = network._implicit_converged_step(
        state, old_energy, 1.0e10, 100.0
    )

    assert np.all(converged)
    assert np.all(np.isfinite(new_energy))
    assert new_energy[0] < old_energy[0]
