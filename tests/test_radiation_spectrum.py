import h5py
import numpy as np
import pytest

from radhydropy.radiation_spectrum import load_radiation_spectrum


def _write_spectrum(filename, sigma_name, epsilon_name):
    with h5py.File(filename, "w") as handle:
        group = handle.create_group("RadiationSpectrum")
        group.create_dataset("group_edges_eV", data=[13.6, 24.6])
        group.create_dataset("ionizing_photon_energy_cgs_erg", data=[3.0e-11])
        group.create_dataset("star_emission_rates", data=[1.0e-32, 1.0e11])
        group.create_dataset(sigma_name, data=[1.0e-18])
        group.create_dataset(epsilon_name, data=[1.0e-12])
        group.attrs["number_of_radiation_groups"] = 1
        group.attrs["stellar_spectrum_type"] = 1
        group.attrs["stellar_spectrum_blackbody_temperature_cgs_K"] = 1.0e5


def test_spectrum_loader_requires_current_cgs_dataset_names(tmp_path):
    filename = tmp_path / "stale.h5"
    _write_spectrum(filename, "group_sigma_gamma_cm2", "group_epsilon_gamma_erg")

    with pytest.raises(ValueError, match="missing required dataset"):
        load_radiation_spectrum(filename)


def test_spectrum_loader_returns_current_cgs_datasets(tmp_path):
    filename = tmp_path / "current.h5"
    _write_spectrum(filename, "group_sigma_gamma_cgs_cm2", "group_epsilon_gamma_cgs_erg")

    result = load_radiation_spectrum(filename)

    np.testing.assert_allclose(result["radiation_group_sigma_gamma"], [1.0e-18])
    np.testing.assert_allclose(result["radiation_group_epsilon_gamma"], [1.0e-12])
