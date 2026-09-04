"""Cached CHIANTI CIE ion-fraction and cooling-table interpolation."""

from pathlib import Path

import h5py
import numpy as np

from radhydropy.constants import PROTON_MASS_CGS


ELEMENT_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
)


def _read_abundance_file(filename):
    atomic_number = []
    log_abundance = []
    for line in Path(filename).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].isdigit():
            atomic_number.append(int(fields[0]))
            log_abundance.append(float(fields[1]))
    return np.asarray(atomic_number), 10.0 ** (np.asarray(log_abundance) - 12.0)


class CIETable:
    """Load and interpolate CHIANTI CIE fractions and cooling coefficients."""

    def __init__(self, ion_fraction_file, cooling_file, abundance_file):
        self.ion_fraction_file = Path(ion_fraction_file).expanduser().resolve()
        self.cooling_file = Path(cooling_file).expanduser().resolve()
        self.abundance_file = Path(abundance_file).expanduser().resolve()

        with h5py.File(self.ion_fraction_file, "r") as table:
            self.log_temperature_ion = table["log10_temperature_K"][:]
            fractions = table["ion_fraction"][:]
            self.atomic_number = table["atomic_number"][:]

        with h5py.File(self.cooling_file, "r") as table:
            self.log_temperature_cooling = table["log10_temperature_K"][:]
            self.log_density_cooling = table["log10_electron_density_cm-3"][:]
            self.metallicity_cooling = table["metallicity_Zsun"][:]
            cooling = table["cooling_erg_cm3_s"][:]

        abundance_atomic_number, abundance = _read_abundance_file(
            self.abundance_file
        )
        if not np.array_equal(self.atomic_number, abundance_atomic_number):
            raise ValueError("CIE and abundance tables contain different elements.")

        self.abundance = abundance
        self.ion_stage = np.arange(fractions.shape[1], dtype=float)
        mean_charge = np.sum(fractions * self.ion_stage[None, :, None], axis=1)
        self._mean_charge = mean_charge
        self._cooling = cooling
        self._electron_fraction_cache = {}
        self._cooling_log_cache = {}

    def electron_fraction(self, temperature_cgs_K, metallicity):
        """Return ``ne / nH`` for temperature and metallicity arrays."""
        temperature_cgs_K = np.asarray(temperature_cgs_K, dtype=float)
        log_temperature = np.log10(np.maximum(temperature_cgs_K, 1.0))
        key = float(metallicity)
        if key not in self._electron_fraction_cache:
            scale = np.ones_like(self.abundance)
            scale[2:] = key
            grid = np.sum(
                self._mean_charge * (self.abundance * scale)[:, None],
                axis=0,
            )
            self._electron_fraction_cache[key] = grid
        return np.interp(
            log_temperature,
            self.log_temperature_ion,
            self._electron_fraction_cache[key],
        )

    def cooling_coefficient(self, temperature_cgs_K, electron_density, metallicity):
        """Return Lambda in erg cm^3 s^-1 using log-space interpolation."""
        temperature_cgs_K = np.asarray(temperature_cgs_K, dtype=float)
        electron_density = np.asarray(electron_density, dtype=float)
        log_temperature = np.log10(np.maximum(temperature_cgs_K, 1.0))
        log_density = np.log10(np.maximum(electron_density, 1.0e-99))

        key = float(metallicity)
        if key not in self._cooling_log_cache:
            if key < self.metallicity_cooling[0] or key > self.metallicity_cooling[-1]:
                raise ValueError(
                    f"metallicity {key:g} is outside the cooling table range "
                    f"{self.metallicity_cooling[0]:g} to "
                    f"{self.metallicity_cooling[-1]:g}"
                )
            metallicity_index = np.clip(
                np.searchsorted(self.metallicity_cooling, key) - 1,
                0,
                len(self.metallicity_cooling) - 2,
            )
            z0 = self.metallicity_cooling[metallicity_index]
            z1 = self.metallicity_cooling[metallicity_index + 1]
            weight = (key - z0) / (z1 - z0)
            selected = (
                (1.0 - weight) * self._cooling[metallicity_index]
                + weight * self._cooling[metallicity_index + 1]
            )
            self._cooling_log_cache[key] = np.log10(np.maximum(selected, 1.0e-99))

        log_cooling = self._cooling_log_cache[key]
        temperature_index = np.clip(
            np.searchsorted(self.log_temperature_cooling, log_temperature) - 1,
            0,
            len(self.log_temperature_cooling) - 2,
        )
        density_index = np.clip(
            np.searchsorted(self.log_density_cooling, log_density) - 1,
            0,
            len(self.log_density_cooling) - 2,
        )
        t0 = self.log_temperature_cooling[temperature_index]
        t1 = self.log_temperature_cooling[temperature_index + 1]
        n0 = self.log_density_cooling[density_index]
        n1 = self.log_density_cooling[density_index + 1]
        wt = np.divide(log_temperature - t0, t1 - t0)
        wn = np.divide(log_density - n0, n1 - n0)

        c00 = log_cooling[temperature_index, density_index]
        c01 = log_cooling[temperature_index, density_index + 1]
        c10 = log_cooling[temperature_index + 1, density_index]
        c11 = log_cooling[temperature_index + 1, density_index + 1]
        result = (
            (1.0 - wt) * (1.0 - wn) * c00
            + (1.0 - wt) * wn * c01
            + wt * (1.0 - wn) * c10
            + wt * wn * c11
        )
        result = 10.0 ** result
        valid_temperature = (
            (log_temperature >= self.log_temperature_cooling[0])
            & (log_temperature <= self.log_temperature_cooling[-1])
        )
        return np.where(valid_temperature, result, 0.0)
