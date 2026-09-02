"""Equation-of-state definitions."""

import numpy as np
import unyt
import radhydropy.utils as ru
from radhydropy.arrays import as_named_array


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
        self.CodeUnits = code_units
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

    def pressure(self, rho, temp, mu):
        """Return pressure from density, temperature, and mean molecular weight."""
        if self.CodeUnits is not None:
            rho_value = np.asarray(rho, dtype=np.longdouble)
            temp_value = np.asarray(temp, dtype=np.longdouble)
            mu_value = np.asarray(mu, dtype=np.longdouble)
            conversion = self.CodeUnits.unit_conversion
            pressure_factor = np.longdouble(
                conversion["boltzmann_code"] / conversion["proton_mass_code"]
            )
            pressure_value = rho_value * temp_value * pressure_factor
            quotient = np.zeros_like(pressure_value, dtype=np.longdouble)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                np.divide(pressure_value, mu_value, out=quotient, where=mu_value != 0)
            return as_named_array(np.asarray(quotient, dtype=float))
        return rho / (mu * unyt.mp) * unyt.kb * temp

    def temperature(self, rho, pressure, mu):
        """Return temperature from density, pressure, and mean molecular weight."""
        if self.CodeUnits is not None:
            rho_value = np.asarray(rho, dtype=float)
            pressure_value = np.asarray(pressure, dtype=float)
            mu_value = np.asarray(mu, dtype=float)
            pressure_over_rho = np.zeros_like(pressure_value, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                np.divide(
                    pressure_value,
                    rho_value,
                    out=pressure_over_rho,
                    where=rho_value != 0,
                )
            return as_named_array(
                (
                pressure_over_rho
                * (mu_value * self.CodeUnits.proton_mass_code)
                / self.CodeUnits.boltzmann_code
                )
            )
        pressure_over_rho = ru.SafeDivide(pressure, rho)
        return (pressure_over_rho * (mu * unyt.mp) / unyt.kb).to(unyt.K)

    def thermal_energy_density(self, pressure):
        """Return thermal energy density for the selected EOS."""
        if self.is_isothermal:
            return np.zeros_like(np.asarray(pressure, dtype=float))
        return pressure / (self.gamma - 1.0)

    def sound_speed(self, rho, pressure, temp=None, mu=None):
        """Return the characteristic sound speed for the selected EOS."""
        if self.CodeUnits is not None:
            gamma_factor = 1.0 if self.is_isothermal else self.gamma
            pressure_value = np.asarray(pressure, dtype=float)
            rho_value = np.asarray(rho, dtype=float)
            pressure_over_rho = np.zeros_like(pressure_value, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                np.divide(
                    pressure_value,
                    rho_value,
                    out=pressure_over_rho,
                    where=rho_value != 0.0,
                )
            return as_named_array(np.sqrt(gamma_factor * pressure_over_rho))
        pressure_over_rho = ru.SafeDivide(pressure, rho)
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
        if self.CodeUnits is not None:
            rho_value = np.asarray(rho, dtype=float)
            vel_value = np.asarray(vel, dtype=float)
            energy_value = np.asarray(energy_density, dtype=float)
            return as_named_array((energy_value - 0.5 * rho_value * vel_value**2) * (self.gamma - 1.0))
        return (energy_density - 0.5 * rho * vel**2) * (self.gamma - 1.0)

    def fluxes(self, rho, vel, pressure):
        """Return conserved densities and Euler fluxes for the selected EOS."""
        Fmass = rho * vel
        qmass = rho
        Fmom = rho * vel * vel
        Fmom[np.logical_or(vel == 0.0, np.isnan(vel))] = 0.0
        Fmom += pressure
        qmom = rho * vel
        if self.is_isothermal:
            zero_energy_flux = as_named_array(np.zeros_like(np.asarray(Fmass, dtype=float)))
            zero_energy_density = as_named_array(np.zeros_like(np.asarray(qmass, dtype=float)))
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
        ionized_fraction_threshold=None,
    ):
        """Apply the piecewise-isothermal closure used by HII-region examples.

        If ``ionized_fraction_threshold`` is ``None``, temperature varies
        continuously with the ionized fraction.  If it is set, cells with
        ``xHII > ionized_fraction_threshold`` receive the ionized temperature
        and all other cells receive the neutral temperature.
        """
        if not self.is_isothermal:
            raise ValueError('piecewise isothermal state requires an isothermal EOS')

        ghost_cells = int(par.mesh.ghost_cells)
        grid_cells = int(par.mesh.grid_cells)
        interior = slice(ghost_cells, ghost_cells + grid_cells)
        ionized_fraction = 1.0 - np.clip(fluid.xHI[interior], 0.0, 1.0)
        if ionized_fraction_threshold is None:
            fluid.temp_code[interior] = (
                neutral_temperature
                + ionized_fraction * (ionized_temperature - neutral_temperature)
            )
        else:
            if not 0.0 <= ionized_fraction_threshold <= 1.0:
                raise ValueError('ionized_fraction_threshold must be in [0, 1]')
            fluid.temp_code[interior] = neutral_temperature
            ionized = ionized_fraction > ionized_fraction_threshold
            temperature = np.asarray(fluid.temp_code[interior])
            temperature[ionized] = ionized_temperature
            fluid.temp_code[interior] = temperature
        fluid.SetHydrogenMu(
            hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0)
        )
        fluid.SetPressure()
