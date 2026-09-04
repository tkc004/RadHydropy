"""HDF5 storage for the multifrequency radiation spectrum."""

from pathlib import Path

import h5py
import numpy as np

from radhydropy.radiation_spectrum import (
    SPECTRUM_DATASET_EPSILON,
    SPECTRUM_DATASET_GROUP_EDGES,
    SPECTRUM_DATASET_IONIZING_ENERGY,
    SPECTRUM_DATASET_SIGMA,
    SPECTRUM_DATASET_STAR_RATES,
    SPECTRUM_GROUP,
    load_radiation_spectrum,
)


def write_blackbody_spectrum(filename):
    """Write the three-group pure-H BB(1e5 K) spectrum used by the example."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filename, "w") as handle:
        group = handle.create_group(SPECTRUM_GROUP)
        group.create_dataset(
            SPECTRUM_DATASET_GROUP_EDGES,
            data=np.array([13.6, 24.6, 54.4, 10000.0]),
        ).attrs["units"] = "eV"
        group.create_dataset(
            SPECTRUM_DATASET_IONIZING_ENERGY,
            data=np.array([3.0208e-11, 5.61973e-11, 1.05154e-10]),
        ).attrs["units"] = "erg"
        group.create_dataset(
            SPECTRUM_DATASET_STAR_RATES,
            data=np.array([1.0e-32, 1.05e11, 2.16e11, 4.80e10]),
        ).attrs["units"] = "internal_energy/time"
        group.create_dataset(
            SPECTRUM_DATASET_SIGMA,
            data=np.array([2.99e-18, 5.66e-19, 7.84e-20]),
        ).attrs["units"] = "cm**2"
        group.create_dataset(
            SPECTRUM_DATASET_EPSILON,
            data=np.array([6.17e-12, 2.81e-11, 7.77e-11]),
        ).attrs["units"] = "erg"
        group.attrs["number_of_radiation_groups"] = 3
        group.attrs["number_of_group_edges"] = 4
        group.attrs["stellar_spectrum_type"] = 1
        group.attrs["stellar_spectrum_type_name"] = "blackbody"
        group.attrs["stellar_spectrum_blackbody_temperature_cgs_K"] = 1.0e5
        group.attrs["absorber"] = "HI"
        group.attrs["description"] = "Pure-hydrogen BB spectrum for the SPHM1RT comparison"


def load_spectrum(filename):
    """Load and validate a radiation spectrum HDF5 file."""
    return load_radiation_spectrum(filename)
