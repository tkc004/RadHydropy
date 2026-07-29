"""Equation-of-state definitions."""

import numpy as np
import unyt


class EOS:
    """Represent the equation of state used by a simulation.

    Parameters
    ----------
    EOStype : str
        Equation-of-state type. Supported values are ``"polytropic"`` and
        ``"isothermal"``.
    gamma : float, optional
        Adiabatic index for a polytropic gas. ``gamma`` must not be 1.
    """

    def __init__(self, EOStype: str, gamma=5.0 / 3.0, code_units=None):
        self.EOStype = EOStype
        self.gamma = gamma
        self.code_units = code_units
        if ((self.EOStype != 'polytropic') and (self.EOStype != 'isothermal')):
            raise Exception("EOS not recognized: only polytropic or isothermal")
        if self.is_polytropic and gamma == 1.0:
            raise Exception("gamma cannot be equal to 1 for a polytropic EOS")

    @property
    def is_polytropic(self):
        """Return ``True`` when the EOS evolves thermal energy."""
        return self.EOStype == 'polytropic'

    @property
    def is_isothermal(self):
        """Return ``True`` for an isothermal closure."""
        return self.EOStype == 'isothermal'

    def _quantity_safe_divide(self, numerator, denominator):
        numerator_value, denominator_value = np.broadcast_arrays(
            np.asarray(numerator.value, dtype=float),
            np.asarray(denominator.value, dtype=float),
        )
        quotient = np.zeros_like(denominator_value, dtype=float)
        np.divide(
            numerator_value,
            denominator_value,
            out=quotient,
            where=denominator_value != 0.0,
        )
        return quotient * (numerator.units / denominator.units)

    def _quantity_values(self, quantity):
        return np.asarray(getattr(quantity, "value", quantity), dtype=float)

    def _quantity_code_values(self, quantity, unit):
        if hasattr(quantity, "to_value"):
            return np.asarray(quantity.to_value(unit), dtype=float)
        return np.asarray(quantity, dtype=float)

    def _pressure_unit(self):
        return self.code_units.pressure_unit if self.code_units is not None else None

    def _temperature_unit(self):
        return self.code_units.temperature_unit if self.code_units is not None else None

    def _velocity_unit(self):
        return self.code_units.velocity_unit if self.code_units is not None else None

    def pressure(self, rho, temp, mu):
        """Return pressure from density, temperature, and mean molecular weight."""
        if self.code_units is not None:
            rho_value = self._quantity_code_values(rho, self.code_units.density_unit)
            temp_value = self._quantity_code_values(temp, self._temperature_unit())
            mu_value = self._quantity_values(mu)
            pressure_value = (
                rho_value
                * self.code_units.boltzmann_code
                * temp_value
                / (mu_value * self.code_units.proton_mass_code)
            )
            return pressure_value * self._pressure_unit()
        return rho / (mu * unyt.mp) * unyt.kb * temp

    def temperature(self, rho, pressure, mu):
        """Return temperature from density, pressure, and mean molecular weight."""
        if self.code_units is not None:
            rho_value = self._quantity_code_values(rho, self.code_units.density_unit)
            pressure_value = self._quantity_code_values(pressure, self._pressure_unit())
            mu_value = self._quantity_values(mu)
            temperature_value = (
                (pressure_value / rho_value)
                * (mu_value * self.code_units.proton_mass_code)
                / self.code_units.boltzmann_code
            )
            return temperature_value * self._temperature_unit()
        pressure_over_rho = self._quantity_safe_divide(pressure, rho)
        return (pressure_over_rho * (mu * unyt.mp) / unyt.kb).to(unyt.K)

    def thermal_energy_density(self, pressure):
        """Return thermal energy density for the selected EOS."""
        if self.is_isothermal:
            return np.zeros_like(np.asarray(pressure.value, dtype=float)) * (
                pressure.units
            )
        return pressure / (self.gamma - 1.0)

    def sound_speed(self, rho, pressure, temp=None, mu=None):
        """Return the characteristic sound speed for the selected EOS."""
        if self.code_units is not None:
            gamma_factor = 1.0 if self.is_isothermal else self.gamma
            pressure_over_rho = (
                self._quantity_code_values(pressure, self._pressure_unit())
                / self._quantity_code_values(rho, self.code_units.density_unit)
            )
            return np.sqrt(gamma_factor * pressure_over_rho) * self._velocity_unit()
        pressure_over_rho = self._quantity_safe_divide(pressure, rho)
        gamma_factor = 1.0 if self.is_isothermal else self.gamma
        soundspeed = np.sqrt(gamma_factor * pressure_over_rho).to(unyt.cm / unyt.s)
        soundspeed[np.isnan(soundspeed)] = 0.0 * unyt.cm / unyt.s
        return soundspeed

    def total_energy_density(self, rho, vel, pressure):
        """Return the conserved energy density."""
        kinetic = 0.5 * rho * vel**2
        return kinetic + self.thermal_energy_density(pressure)

    def pressure_from_conserved(self, rho, vel, energy_density, temp=None, mu=None):
        """Recover pressure from conserved variables."""
        if self.is_isothermal:
            if temp is None or mu is None:
                raise ValueError(
                    "isothermal pressure reconstruction requires temperature and mu"
                )
            return self.pressure(rho, temp, mu)
        if self.code_units is not None:
            rho_value = self._quantity_code_values(rho, self.code_units.density_unit)
            vel_value = self._quantity_code_values(vel, self._velocity_unit())
            energy_value = self._quantity_code_values(energy_density, self._pressure_unit())
            pressure_value = (energy_value - 0.5 * rho_value * vel_value**2) * (self.gamma - 1.0)
            return pressure_value * self._pressure_unit()
        return (energy_density - 0.5 * rho * vel**2) * (self.gamma - 1.0)

    def fluxes(self, rho, vel, pressure):
        """Return conserved densities and Euler fluxes for the selected EOS."""
        Fmass = rho * vel
        qmass = rho
        Fmom = rho * vel * vel
        Fmom[np.logical_or(vel == 0.0, np.isnan(vel))] = 0.0 * rho[0] * vel[0] ** 2
        Fmom += pressure
        qmom = rho * vel
        if self.is_isothermal:
            zero_energy_flux = np.zeros_like(np.asarray(Fmass.value, dtype=float)) * (
                unyt.g / unyt.s**3
            )
            zero_energy_density = np.zeros_like(
                np.asarray(qmass.value, dtype=float)
            ) * (unyt.g / (unyt.cm * unyt.s**2))
            return Fmass, qmass, Fmom, qmom, zero_energy_flux, zero_energy_density
        FEn = vel * (self.gamma * pressure / (self.gamma - 1.0) + 0.5 * rho * vel**2)
        qEn = pressure / (self.gamma - 1.0) + rho * vel**2 * 0.5
        return Fmass, qmass, Fmom, qmom, FEn, qEn

    def apply_piecewise_isothermal_state(
        self,
        fluid,
        par,
        neutral_temperature,
        ionized_temperature,
    ):
        """Apply the piecewise-isothermal closure used by HII-region examples."""
        if not self.is_isothermal:
            raise ValueError('piecewise isothermal state requires an isothermal EOS')

        interior = slice(par.noghost, par.noghost + par.nogrid)
        ionized_fraction = 1.0 - np.clip(fluid.xHI[interior], 0.0, 1.0)
        fluid.temp[interior] = (
            neutral_temperature
            + ionized_fraction * (ionized_temperature - neutral_temperature)
        )
        fluid.SetHydrogenMu(
            hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0)
        )
        fluid.SetPressure()
