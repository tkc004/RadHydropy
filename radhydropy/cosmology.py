"""Analytic background cosmologies used by cosmological test problems."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS


@lru_cache(maxsize=None)
def _legendre_quadrature(order):
    return np.polynomial.legendre.leggauss(order)


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

    def cosmic_time_from_scale_factor(self, scale_factor):
        """Return cosmic time for a supplied scale factor."""
        scale_factor = np.asarray(scale_factor, dtype=float)
        if np.any(scale_factor <= 0.0):
            raise ValueError("scale factor must be positive")
        return self.t_ref * (scale_factor / self.a_ref) ** 1.5

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

    def background_state_from_supercomoving(self, tau):
        """Return ``(cosmic_time, scale_factor, hubble)`` at ``tau``."""
        cosmic_time = self.cosmic_time_from_supercomoving(tau)
        scale_factor = self.scale_factor(cosmic_time)
        return cosmic_time, scale_factor, 2.0 / (3.0 * cosmic_time)

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


@dataclass(frozen=True)
class LambdaCDM:
    """Flat matter-plus-cosmological-constant background in code units.

    ``t_ref`` is the cosmic time at ``a_ref``.  ``hubble_ref`` is the Hubble
    parameter at that scale factor; if omitted it is chosen so that the age
    of the universe at ``a_ref`` is ``t_ref``.  The density parameters are
    defined at ``a_ref`` and must sum to one.

    The homogeneous density returned by :meth:`background_density` is the
    gravitating matter density.  Dark energy is smooth and is represented by
    the background expansion, not by an enclosed perturbation mass.
    """

    t_ref: float = 1.0
    a_ref: float = 1.0
    omega_m: float = 0.3
    omega_lambda: float = 0.7
    hubble_ref: float | None = None
    gravitational_constant: float = 1.0

    @classmethod
    def from_code_units(
        cls, code_units, t_ref=1.0, a_ref=1.0, omega_m=0.3,
        omega_lambda=0.7, hubble_ref=None,
    ):
        g_code = (
            GRAVITATIONAL_CONSTANT_CGS * code_units.mass_in_cgs
            / (code_units.length_in_cgs * code_units.velocity_in_cgs**2)
        )
        return cls(
            t_ref=float(t_ref), a_ref=float(a_ref),
            omega_m=float(omega_m), omega_lambda=float(omega_lambda),
            hubble_ref=None if hubble_ref is None else float(hubble_ref),
            gravitational_constant=float(g_code),
        )

    def __post_init__(self):
        if self.t_ref <= 0.0 or self.a_ref <= 0.0:
            raise ValueError("LambdaCDM reference time and scale factor must be positive")
        if self.omega_m <= 0.0 or self.omega_lambda < 0.0:
            raise ValueError("LambdaCDM requires omega_m > 0 and omega_lambda >= 0")
        if not np.isclose(self.omega_m + self.omega_lambda, 1.0):
            raise ValueError("LambdaCDM density parameters must sum to one")
        if self.hubble_ref is not None and self.hubble_ref <= 0.0:
            raise ValueError("LambdaCDM hubble_ref must be positive")

    @property
    def _hubble_ref(self):
        if self.hubble_ref is not None:
            return self.hubble_ref
        if self.omega_lambda == 0.0:
            return 2.0 / (3.0 * self.t_ref)
        return 2.0 * np.arcsinh(np.sqrt(self.omega_lambda / self.omega_m)) / (
            3.0 * self.t_ref * np.sqrt(self.omega_lambda)
        )

    def _validate_time(self, time):
        time = np.asarray(time, dtype=float)
        if np.any(time <= self._big_bang_time):
            raise ValueError("LambdaCDM cosmic time must be after the big bang")
        return time

    @property
    def _age_ref(self):
        if self.omega_lambda == 0.0:
            return 2.0 / (3.0 * self._hubble_ref)
        return 2.0 * np.arcsinh(np.sqrt(self.omega_lambda / self.omega_m)) / (
            3.0 * self._hubble_ref * np.sqrt(self.omega_lambda)
        )

    @property
    def _big_bang_time(self):
        return self.t_ref - self._age_ref

    def scale_factor(self, time):
        age = self._validate_time(time) - self._big_bang_time
        if self.omega_lambda == 0.0:
            return self.a_ref * (age / self._age_ref) ** (2.0 / 3.0)
        argument = 1.5 * self._hubble_ref * np.sqrt(self.omega_lambda) * age
        return self.a_ref * (
            np.sinh(argument) / np.sqrt(self.omega_lambda / self.omega_m)
        ) ** (2.0 / 3.0)

    def hubble(self, time):
        a = self.scale_factor(time)
        ratio = self.a_ref / a
        return self._hubble_ref * np.sqrt(self.omega_m * ratio**3 + self.omega_lambda)

    def cosmic_time_from_scale_factor(self, scale_factor):
        """Return cosmic time for a supplied scale factor."""
        scale_factor = np.asarray(scale_factor, dtype=float)
        if np.any(scale_factor <= 0.0):
            raise ValueError("scale factor must be positive")
        u = scale_factor / self.a_ref
        return np.vectorize(self._time_from_u, otypes=[float])(u)

    def background_density(self, time):
        a = self.scale_factor(time)
        return 3.0 * self._hubble_ref**2 * self.omega_m / (
            8.0 * np.pi * self.gravitational_constant
        ) * (self.a_ref / a) ** 3

    def supercomoving_time(self, time):
        """Return ``tau`` with ``d tau = d t / a(t)**2`` and ``tau(t_ref)=0``."""
        a = self.scale_factor(self._validate_time(time))
        u = np.asarray(a / self.a_ref, dtype=float)
        return np.vectorize(self._tau_integral, otypes=[float])(u) / (
            self.a_ref**2 * self._hubble_ref
        )

    def cosmic_time_from_supercomoving(self, tau):
        tau = np.asarray(tau, dtype=float)
        target = tau * self.a_ref**2 * self._hubble_ref
        result = np.vectorize(self._cosmic_time_from_supercomoving_scalar, otypes=[float])(target)
        return result

    @lru_cache(maxsize=4096)
    def _cosmic_time_from_supercomoving_scalar(self, value):
        """Invert one supercomoving time; repeated solver-time queries are cached."""
        value = float(value)
        lo, hi = np.finfo(float).tiny, 1.0
        if value >= 0.0:
            while self._tau_integral(hi) < value:
                hi *= 2.0
            lo = 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self._tau_integral(mid) < value:
                lo = mid
            else:
                hi = mid
        return self._time_from_u(0.5 * (lo + hi))

    def _tau_integral(self, u):
        nodes, weights = _legendre_quadrature(48)
        lo, hi = (u, 1.0) if u < 1.0 else (1.0, u)
        mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
        values = mid + half * nodes
        integral = half * np.sum(weights / (np.sqrt(values) * np.sqrt(
            self.omega_m + self.omega_lambda * values**3)))
        return (-integral if u < 1.0 else integral)

    def _time_from_u(self, u):
        if self.omega_lambda == 0.0:
            age = self._age_ref * u**1.5
        else:
            age = 2.0 / (3.0 * self._hubble_ref * np.sqrt(self.omega_lambda)) * np.arcsinh(
                np.sqrt(self.omega_lambda / self.omega_m) * u**1.5)
        return self._big_bang_time + age

    def scale_factor_from_supercomoving(self, tau):
        return self.scale_factor(self.cosmic_time_from_supercomoving(tau))

    def hubble_from_supercomoving(self, tau):
        return self.hubble(self.cosmic_time_from_supercomoving(tau))

    def background_state_from_supercomoving(self, tau):
        """Return ``(cosmic_time, scale_factor, hubble)`` at ``tau``."""
        cosmic_time = self.cosmic_time_from_supercomoving(tau)
        scale_factor = self.scale_factor(cosmic_time)
        ratio = self.a_ref / scale_factor
        hubble = self._hubble_ref * np.sqrt(self.omega_m * ratio**3 + self.omega_lambda)
        return cosmic_time, scale_factor, hubble

    def physical_radius(self, x, tau):
        return self.scale_factor_from_supercomoving(tau) * np.asarray(x, dtype=float)

    def physical_density(self, varrho, tau):
        a = self.scale_factor_from_supercomoving(tau)
        return np.asarray(varrho, dtype=float) / a**3

    def physical_pressure(self, pressure, tau, gamma):
        a = self.scale_factor_from_supercomoving(tau)
        return np.asarray(pressure, dtype=float) / a**(3.0 * gamma)

    def physical_velocity(self, x, velocity, tau):
        a = self.scale_factor_from_supercomoving(tau)
        return self.hubble_from_supercomoving(tau) * a * np.asarray(x, dtype=float) + np.asarray(velocity, dtype=float) / a

    @property
    def type_name(self):
        return "lambda_cdm"
