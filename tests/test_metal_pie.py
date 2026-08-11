from pathlib import Path

import h5py
import numpy as np

from radhydropy.constants import PROTON_MASS_CGS
from radhydropy.thermo_networks.hydrogen_helium import _closure, _rates
from radhydropy.thermo_networks.pie import MetalPIETable


TABLE_FILENAME = (
    Path(__file__).resolve().parents[2]
    / "metal_pie_table"
    / "metal_pie_table_Z1_metals.h5"
)


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
        "rho_g_cm3": np.array([PROTON_MASS_CGS]),
        "hydrogen_mass_fraction": 0.75,
        "helium_mass_fraction": 0.25,
        "temperature_K": np.array([1.0e4]),
        "xHI": np.array([0.8]),
        "xHeI": np.array([0.7]),
        "xHeIII": np.array([0.0]),
        "sigma_gamma_cm2": {
            "HI": np.array([1.0e-18, 2.0e-19]),
            "HeI": np.array([3.0e-18, 4.0e-19]),
            "HeII": np.array([0.0, 5.0e-19]),
        },
        "epsilon_gamma_erg": {
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
        temperature_K=np.array([1.0e4, 1.0e5]),
        hydrogen_density_cm3=np.array([1.0e-3, 1.0e-3]),
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
    ngamma = np.array([[1.0e-4], [2.0e-4]])
    no_pie_rate = _rates(state_without_pie, ngamma)[3]
    pie_rate = _rates(state_with_pie, ngamma)[3]
    n_h = state_with_pie["rho_g_cm3"] * 0.75 / PROTON_MASS_CGS
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
    ngamma = np.array([[1.0e-6], [2.0e-6]])
    rate_with_pie = _rates(state, ngamma)[3]
    rate_without_pie = _rates({**state, "metal_pie_table": None}, ngamma)[3]
    # At this state the synthetic cooling dominates, so PIE must reduce the
    # thermal rate. Reversing the table rates should reverse the sign.
    assert np.all(rate_with_pie < rate_without_pie)

