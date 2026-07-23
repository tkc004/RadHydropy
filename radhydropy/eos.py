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

    def __init__(self, EOStype: str, gamma=5.0 / 3.0):
        self.EOStype = EOStype
        self.gamma = gamma
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

    def _safe_divide(self, numerator, denominator):
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

    def pressure(self, rho, temp, mu):
        """Return pressure from density, temperature, and mean molecular weight."""
        return rho / (mu * unyt.mp) * unyt.kb * temp

    def temperature(self, rho, pressure, mu):
        """Return temperature from density, pressure, and mean molecular weight."""
        pressure_over_rho = self._safe_divide(pressure, rho)
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
        pressure_over_rho = self._safe_divide(pressure, rho)
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

