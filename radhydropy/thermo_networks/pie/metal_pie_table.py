"""Interpolation for photoionization-equilibrium heating and cooling tables."""

from pathlib import Path

import h5py
import numpy as np


class MetalPIETable:
    """Load and trilinearly interpolate a metal-only PIE HDF5 table."""

    def __init__(self, filename):
        self.filename = Path(filename).expanduser().resolve()
        with h5py.File(self.filename, "r") as handle:
            group = handle["MetalPIE"]
            self.metadata = dict(group.attrs)
            axes = group["axes"]
            self.log_temperature = np.asarray(axes["log10_temperature_K"], dtype=float)
            self.log_density = np.asarray(axes["log10_hydrogen_density_cm-3"], dtype=float)
            self.is_hm12_uv_background = (
                self.metadata.get("spectrum_type")
                == "Haardt-Madau 2012 UV background"
                or self.metadata.get("radiation_background") == "table HM12 redshift"
            )
            if self.is_hm12_uv_background:
                self.redshift = np.asarray(axes["redshift"], dtype=float)
                self.log_ionization_parameter = None
            else:
                self.log_ionization_parameter = np.asarray(
                    axes["log10_ionization_parameter"], dtype=float
                )
                self.redshift = None
            self.metallicity = np.asarray(axes["metallicity_Zsun"], dtype=float)
            rates = group["rates"]
            self.component = self.metadata.get("component", "metals")
            if self.component == "hydrogen+helium+metals":
                heating_name = "photoheating_erg_cm3_s"
                cooling_name = "cooling_erg_cm3_s"
            else:
                heating_name = "metal_photoheating_erg_cm3_s"
                cooling_name = "metal_cooling_erg_cm3_s"
            self._heating = np.asarray(rates[heating_name], dtype=float)
            self._cooling = np.asarray(rates[cooling_name], dtype=float)

        third_axis_length = len(
            self.redshift if self.is_hm12_uv_background
            else self.log_ionization_parameter
        )
        expected = (
            len(self.log_temperature),
            len(self.log_density),
            third_axis_length,
            len(self.metallicity),
        )
        if self._heating.shape != expected or self._cooling.shape != expected:
            raise ValueError("metal PIE rate arrays do not match the table axes")
        # The implicit H/He solver may query the table many times per source
        # step.  Store logarithmic rates once instead of taking log10 on every
        # interpolation call.
        self._log_heating = np.log10(np.maximum(self._heating, 1.0e-99))
        self._log_cooling = np.log10(np.maximum(self._cooling, 1.0e-99))

    @staticmethod
    def _bracket(grid, value):
        value = np.clip(np.asarray(value, dtype=float), grid[0], grid[-1])
        index = np.clip(np.searchsorted(grid, value, side="right") - 1, 0, len(grid) - 2)
        weight = (value - grid[index]) / (grid[index + 1] - grid[index])
        return index, weight

    def _coordinates(self, temperature_cgs_K, hydrogen_density_cgs_cm3, third_axis):
        log_t = np.log10(np.maximum(np.asarray(temperature_cgs_K, dtype=float), 1.0))
        log_n = np.log10(np.maximum(np.asarray(hydrogen_density_cgs_cm3, dtype=float), 1.0e-99))
        third = np.asarray(third_axis, dtype=float)
        if not self.is_hm12_uv_background:
            third = np.log10(np.maximum(third, 1.0e-99))
        log_t, log_n, third = np.broadcast_arrays(log_t, log_n, third)
        it, wt = self._bracket(self.log_temperature, log_t)
        inn, wn = self._bracket(self.log_density, log_n)
        third_grid = self.redshift if self.is_hm12_uv_background else self.log_ionization_parameter
        iu, wu = self._bracket(third_grid, third)
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

    def rates(
        self,
        temperature_cgs_K,
        hydrogen_density_cgs_cm3,
        ionization_parameter=None,
        metallicity=1.0,
        redshift=None,
    ):
        if self.is_hm12_uv_background:
            if redshift is None:
                raise ValueError("HM12 PIE tables require a redshift lookup value")
            third_axis = redshift
        else:
            if ionization_parameter is None:
                raise ValueError("PIE tables require an ionization parameter")
            third_axis = ionization_parameter
        coordinates = self._coordinates(
            temperature_cgs_K, hydrogen_density_cgs_cm3, third_axis
        )
        metallicity_index = int(np.argmin(np.abs(self.metallicity - float(metallicity))))
        return (
            self._interpolate_log(
                self._log_heating[..., metallicity_index], coordinates
            ),
            self._interpolate_log(
                self._log_cooling[..., metallicity_index], coordinates
            ),
        )
