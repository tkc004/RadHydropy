"""Load radiation-spectrum data used by the runtime parameter system."""

from pathlib import Path

import h5py
import numpy as np

SPECTRUM_GROUP = "RadiationSpectrum"


def load_radiation_spectrum(filename):
    """Read and validate a radiation spectrum from HDF5."""
    with h5py.File(filename, "r") as handle:
        group = handle[SPECTRUM_GROUP]
        edges = np.asarray(group["group_edges_eV"], dtype=float)
        energies = np.asarray(group["ionizing_photon_energy_erg"], dtype=float)
        rates = np.asarray(group["star_emission_rates"], dtype=float)
        ngroup = int(group.attrs["number_of_radiation_groups"])
        if edges.size != ngroup + 1 or energies.size != ngroup:
            raise ValueError("radiation spectrum group metadata is inconsistent")
        if rates.size != ngroup + 1:
            raise ValueError("star emission rates must include the non-ionizing entry")
        result = {
            "radiation_group_edges_eV": edges,
            "ionizing_photon_energy_erg": energies,
            "star_emission_rates": rates,
            "number_of_radiation_groups": ngroup,
            "stellar_spectrum_type": int(group.attrs["stellar_spectrum_type"]),
            "stellar_spectrum_blackbody_temperature_K": float(
                group.attrs["stellar_spectrum_blackbody_temperature_K"]
            ),
        }
        if "group_sigma_gamma_cm2" in group:
            result["radiation_group_sigma_gamma"] = np.asarray(
                group["group_sigma_gamma_cm2"], dtype=float
            )
        if "group_epsilon_gamma_erg" in group:
            result["radiation_group_epsilon_gamma"] = np.asarray(
                group["group_epsilon_gamma_erg"], dtype=float
            )
        return result


def resolve_spectrum_filename(filename, base_directory=None):
    path = Path(filename)
    if not path.is_absolute() and base_directory is not None:
        path = Path(base_directory) / path
    return path.resolve()
