"""Analytic background cosmologies used by cosmological test problems."""

from dataclasses import dataclass

import numpy as np

from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS


@dataclass(frozen=True)
class EinsteinDeSitter:
    """Einstein--de Sitter background in simulation code units.

    Parameters are dimensionless code-unit values.  The reference scale
    factor is ``a_ref`` at ``t_ref``; time must be strictly positive.
    """

    t_ref: float = 1.0
    a_ref: float = 1.0
    gravitational_constant: float = 1.0

    @classmethod
    def from_code_units(cls, code_units, t_ref=1.0, a_ref=1.0):
        """Construct the background using the code-unit gravitational constant."""
        g_code = (
            GRAVITATIONAL_CONSTANT_CGS
            * code_units.mass_in_cgs
            / (code_units.length_in_cgs * code_units.velocity_in_cgs**2)
        )
        return cls(t_ref=float(t_ref), a_ref=float(a_ref), gravitational_constant=float(g_code))

    def _validate_time(self, time):
        time = np.asarray(time, dtype=float)
        if np.any(time <= 0.0):
            raise ValueError("Einstein-de Sitter cosmic time must be positive")
        return time

    def scale_factor(self, time):
        """Return ``a(t)`` normalized to ``a_ref`` at ``t_ref``."""
        time = self._validate_time(time)
        return self.a_ref * (time / self.t_ref) ** (2.0 / 3.0)

    def hubble(self, time):
        """Return the Hubble parameter ``H(t)`` in inverse code time."""
        time = self._validate_time(time)
        return 2.0 / (3.0 * time)

    def background_density(self, time):
        """Return the homogeneous EdS density in code mass/length cubed."""
        time = self._validate_time(time)
        return 1.0 / (6.0 * np.pi * self.gravitational_constant * time**2)

    def supercomoving_time(self, time):
        """Return ``tau`` defined by ``d tau = d t / a(t)**2``.

        The origin is chosen at ``t_ref``.  This finite offset is convenient
        for simulations, which must always start at positive cosmic time.
        """
        time = self._validate_time(time)
        return 3.0 * self.t_ref / self.a_ref**2 * (
            1.0 - (time / self.t_ref) ** (-1.0 / 3.0)
        )

    def cosmic_time_from_supercomoving(self, tau):
        """Invert :meth:`supercomoving_time`."""
        tau = np.asarray(tau, dtype=float)
        scale = 3.0 * self.t_ref / self.a_ref**2
        if np.any(tau >= scale):
            raise ValueError("supercomoving time is outside the EdS domain")
        return self.t_ref * (1.0 - tau / scale) ** (-3.0)

    def scale_factor_from_supercomoving(self, tau):
        """Return ``a`` directly from supercomoving time."""
        return self.scale_factor(self.cosmic_time_from_supercomoving(tau))

    def hubble_from_supercomoving(self, tau):
        """Return the physical Hubble parameter at supercomoving time ``tau``."""
        return self.hubble(self.cosmic_time_from_supercomoving(tau))

    def physical_radius(self, x, tau):
        """Convert comoving radius ``x`` to proper radius."""
        return self.scale_factor_from_supercomoving(tau) * np.asarray(x, dtype=float)

    def physical_density(self, varrho, tau):
        """Convert comoving density to proper density."""
        a = self.scale_factor_from_supercomoving(tau)
        return np.asarray(varrho, dtype=float) / a**3

    def physical_pressure(self, pressure, tau, gamma):
        """Convert supercomoving pressure to proper pressure."""
        a = self.scale_factor_from_supercomoving(tau)
        return np.asarray(pressure, dtype=float) / a**(3.0 * gamma)

    def physical_velocity(self, x, velocity, tau):
        """Convert supercomoving velocity to proper velocity."""
        a = self.scale_factor_from_supercomoving(tau)
        hubble = self.hubble_from_supercomoving(tau)
        return hubble * a * np.asarray(x, dtype=float) + np.asarray(velocity, dtype=float) / a

    @property
    def type_name(self):
        return "einstein_de_sitter"
