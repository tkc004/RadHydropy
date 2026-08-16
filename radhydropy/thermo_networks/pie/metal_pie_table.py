"""Interpolation for volumetric metal PIE heating and cooling tables."""

from pathlib import Path

import h5py
import numpy as np


class MetalPIETable:
    """Load and trilinearly interpolate a metal-only PIE HDF5 table."""

    def __init__(self, filename):
        self.filename = Path(filename).expanduser().resolve()
        with h5py.File(self.filename, "r") as handle:
            group = handle["MetalPIE"]
            axes = group["axes"]
            self.log_temperature = np.asarray(axes["log10_temperature_K"], dtype=float)
            self.log_density = np.asarray(axes["log10_hydrogen_density_cm-3"], dtype=float)
            self.log_ionization_parameter = np.asarray(
                axes["log10_ionization_parameter"], dtype=float
            )
            self.metallicity = np.asarray(axes["metallicity_Zsun"], dtype=float)
            rates = group["rates"]
            self._heating = np.asarray(
                rates["metal_photoheating_erg_cm3_s"], dtype=float
            )
            self._cooling = np.asarray(
                rates["metal_cooling_erg_cm3_s"], dtype=float
            )
            self.metadata = dict(group.attrs)
            self.is_hm12_uv_background = (
                self.metadata.get("spectrum_type")
                == "Haardt-Madau 2012 UV background"
                or self.metadata.get("radiation_background") == "table HM12 redshift"
            )

        expected = (
            len(self.log_temperature),
            len(self.log_density),
            len(self.log_ionization_parameter),
            len(self.metallicity),
        )
        if self._heating.shape != expected or self._cooling.shape != expected:
            raise ValueError("metal PIE rate arrays do not match the table axes")
        if len(self.metallicity) != 1:
            raise ValueError("only singleton metallicity tables are currently supported")

        # The implicit H/He solver may query the table many times per source
        # step.  Store logarithmic rates once instead of taking log10 on every
        # interpolation call.
        self._log_heating = np.log10(np.maximum(self._heating[..., 0], 1.0e-99))
        self._log_cooling = np.log10(np.maximum(self._cooling[..., 0], 1.0e-99))

    @staticmethod
    def _bracket(grid, value):
        value = np.clip(np.asarray(value, dtype=float), grid[0], grid[-1])
        index = np.clip(np.searchsorted(grid, value, side="right") - 1, 0, len(grid) - 2)
        weight = (value - grid[index]) / (grid[index + 1] - grid[index])
        return index, weight

    def _coordinates(self, temperature_K, hydrogen_density_cm3, ionization_parameter):
        log_t = np.log10(np.maximum(np.asarray(temperature_K, dtype=float), 1.0))
        log_n = np.log10(np.maximum(np.asarray(hydrogen_density_cm3, dtype=float), 1.0e-99))
        log_u = np.log10(np.maximum(np.asarray(ionization_parameter, dtype=float), 1.0e-99))
        log_t, log_n, log_u = np.broadcast_arrays(log_t, log_n, log_u)
        it, wt = self._bracket(self.log_temperature, log_t)
        inn, wn = self._bracket(self.log_density, log_n)
        iu, wu = self._bracket(self.log_ionization_parameter, log_u)
        return it, wt, inn, wn, iu, wu

    @staticmethod
    def _interpolate_log(log_values, coordinates):
        it, wt, inn, wn, iu, wu = coordinates
        c000 = log_values[it, inn, iu]
        c001 = log_values[it, inn, iu + 1]
        c010 = log_values[it, inn + 1, iu]
        c011 = log_values[it, inn + 1, iu + 1]
        c100 = log_values[it + 1, inn, iu]
        c101 = log_values[it + 1, inn, iu + 1]
        c110 = log_values[it + 1, inn + 1, iu]
        c111 = log_values[it + 1, inn + 1, iu + 1]
        low = (1.0 - wu) * c000 + wu * c001
        low_n = (1.0 - wn) * low + wn * ((1.0 - wu) * c010 + wu * c011)
        high = (1.0 - wu) * c100 + wu * c101
        high_n = (1.0 - wn) * high + wn * ((1.0 - wu) * c110 + wu * c111)
        return 10.0 ** ((1.0 - wt) * low_n + wt * high_n)

    def rates(self, temperature_K, hydrogen_density_cm3, ionization_parameter, metallicity=1.0):
        if not np.isclose(float(metallicity), self.metallicity[0]):
            raise ValueError(
                f"metal PIE table supports metallicity {self.metallicity[0]:g} only"
            )
        coordinates = self._coordinates(
            temperature_K, hydrogen_density_cm3, ionization_parameter
        )
        return (
            self._interpolate_log(self._log_heating, coordinates),
            self._interpolate_log(self._log_cooling, coordinates),
        )
