"""Load radiation-spectrum data used by the runtime parameter system."""

from pathlib import Path

import h5py
import numpy as np

SPECTRUM_GROUP = "RadiationSpectrum"
SPECTRUM_DATASET_GROUP_EDGES = "group_edges_eV"
SPECTRUM_DATASET_IONIZING_ENERGY = "ionizing_photon_energy_cgs_erg"
SPECTRUM_DATASET_STAR_RATES = "star_emission_rates"
SPECTRUM_DATASET_SIGMA = "group_sigma_gamma_cgs_cm2"
SPECTRUM_DATASET_EPSILON = "group_epsilon_gamma_cgs_erg"


def _required_dataset(group, name):
    if name not in group:
        raise ValueError(
            f"radiation spectrum is missing required dataset {name!r}; "
            "regenerate the spectrum with the current schema"
        )
    return group[name]


def load_radiation_spectrum(filename):
    """Read and validate a radiation spectrum from HDF5."""
    with h5py.File(filename, "r") as handle:
        if SPECTRUM_GROUP not in handle:
            raise ValueError(f"radiation spectrum is missing group {SPECTRUM_GROUP!r}")
        group = handle[SPECTRUM_GROUP]
        edges = np.asarray(_required_dataset(group, SPECTRUM_DATASET_GROUP_EDGES), dtype=float)
        energies = np.asarray(_required_dataset(group, SPECTRUM_DATASET_IONIZING_ENERGY), dtype=float)
        rates = np.asarray(_required_dataset(group, SPECTRUM_DATASET_STAR_RATES), dtype=float)
        sigma = np.asarray(_required_dataset(group, SPECTRUM_DATASET_SIGMA), dtype=float)
        epsilon = np.asarray(_required_dataset(group, SPECTRUM_DATASET_EPSILON), dtype=float)
        ngroup = int(group.attrs["number_of_radiation_groups"])
        if edges.size != ngroup + 1 or energies.size != ngroup:
            raise ValueError("radiation spectrum group metadata is inconsistent")
        if rates.size != ngroup + 1 or sigma.size != ngroup or epsilon.size != ngroup:
            raise ValueError("star emission rates must include the non-ionizing entry")
        result = {
            "radiation_group_edges_eV": edges,
            "ionizing_photon_energy_cgs_erg": energies,
            "star_emission_rates": rates,
            "number_of_radiation_groups": ngroup,
            "stellar_spectrum_type": int(group.attrs["stellar_spectrum_type"]),
            "stellar_spectrum_blackbody_temperature_cgs_K": float(
                group.attrs["stellar_spectrum_blackbody_temperature_cgs_K"]
            ),
            "radiation_group_sigma_gamma": sigma,
            "radiation_group_epsilon_gamma": epsilon,
        }
        for species in ("HeI", "HeII"):
            sigma_name = f"group_sigma_gamma_{species}_cgs_cm2"
            epsilon_name = f"group_epsilon_gamma_{species}_cgs_erg"
            if sigma_name in group:
                result[f"radiation_group_sigma_gamma_{species}"] = np.asarray(group[sigma_name], dtype=float)
            if epsilon_name in group:
                result[f"radiation_group_epsilon_gamma_{species}"] = np.asarray(group[epsilon_name], dtype=float)
        return result


def resolve_spectrum_filename(filename, base_directory=None):
    path = Path(filename)
    if not path.is_absolute() and base_directory is not None:
        path = Path(base_directory) / path
    return path.resolve()
