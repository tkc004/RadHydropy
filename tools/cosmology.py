"""Small physical-unit background cosmology calculator.

Times are in Gyr, Hubble parameters in km/s/Mpc, and densities in
``Msun/Mpc**3``.  The default factory returns an Einstein--de Sitter model;
select ``cosmology_type='lambda_cdm'`` for a flat matter--Lambda model.
"""

from dataclasses import dataclass

import numpy as np


_KM_PER_MPC = 3.0856775814913673e19
_SECONDS_PER_GYR = 365.25 * 24.0 * 3600.0 * 1.0e9
_G_MPC_KMS_MSUN = 4.300917270e-9


def _validate_flat_parameters(omega_m, omega_lambda):
    omega_m = float(omega_m)
    omega_lambda = float(omega_lambda)
    if omega_m <= 0.0 or omega_lambda < 0.0:
        raise ValueError("require omega_m > 0 and omega_lambda >= 0")
    if not np.isclose(omega_m + omega_lambda, 1.0):
        raise ValueError("flat cosmology requires omega_m + omega_lambda = 1")
    return omega_m, omega_lambda


@dataclass(frozen=True)
class EinsteinDeSitter:
    """Einstein--de Sitter cosmology in physical astronomical units."""

    h0: float = 70.0

    def __post_init__(self):
        if self.h0 <= 0.0:
            raise ValueError("h0 must be positive")

    @property
    def hubble_0(self):
        return float(self.h0)

    @property
    def age_0(self):
        return 2.0 / (3.0 * self.hubble_0_gyr)

    @property
    def hubble_0_gyr(self):
        return self.hubble_0 / _KM_PER_MPC * _SECONDS_PER_GYR

    def scale_factor(self, cosmic_time):
        time = np.asarray(cosmic_time, dtype=float)
        if np.any(time <= 0.0):
            raise ValueError("cosmic time must be positive")
        return (time / self.age_0) ** (2.0 / 3.0)

    def cosmic_time_from_scale_factor(self, scale_factor):
        scale_factor = np.asarray(scale_factor, dtype=float)
        if np.any(scale_factor <= 0.0):
            raise ValueError("scale factor must be positive")
        return self.age_0 * scale_factor**1.5

    def hubble(self, cosmic_time):
        return self.hubble_0 / self.scale_factor(cosmic_time) ** 1.5

    def critical_density(self, cosmic_time):
        h = self.hubble(cosmic_time)
        return 3.0 * h**2 / (8.0 * np.pi * _G_MPC_KMS_MSUN)

    def matter_density(self, cosmic_time):
        return self.critical_density(cosmic_time)

    def dark_energy_density(self, cosmic_time):
        return np.zeros_like(np.asarray(cosmic_time, dtype=float))

    def background_density(self, cosmic_time):
        return self.matter_density(cosmic_time)

    def redshift(self, cosmic_time):
        return 1.0 / self.scale_factor(cosmic_time) - 1.0

    @property
    def type_name(self):
        return "einstein_de_sitter"


@dataclass(frozen=True)
class LambdaCDM:
    """Flat matter-plus-cosmological-constant background."""

    h0: float = 70.0
    omega_m: float = 0.3
    omega_lambda: float = 0.7

    def __post_init__(self):
        if self.h0 <= 0.0:
            raise ValueError("h0 must be positive")
        _validate_flat_parameters(self.omega_m, self.omega_lambda)

    @property
    def hubble_0(self):
        return float(self.h0)

    @property
    def hubble_0_gyr(self):
        return self.hubble_0 / _KM_PER_MPC * _SECONDS_PER_GYR

    @property
    def age_0(self):
        return 2.0 * np.arcsinh(np.sqrt(self.omega_lambda / self.omega_m)) / (
            3.0 * self.hubble_0_gyr * np.sqrt(self.omega_lambda)
        ) if self.omega_lambda > 0.0 else 2.0 / (3.0 * self.hubble_0_gyr)

    def scale_factor(self, cosmic_time):
        time = np.asarray(cosmic_time, dtype=float)
        if np.any(time <= 0.0):
            raise ValueError("cosmic time must be positive")
        argument = 1.5 * self.hubble_0_gyr * np.sqrt(self.omega_lambda) * time
        return (
            np.sinh(argument) / np.sqrt(self.omega_lambda / self.omega_m)
        ) ** (2.0 / 3.0) if self.omega_lambda > 0.0 else (
            time / self.age_0
        ) ** (2.0 / 3.0)

    def cosmic_time_from_scale_factor(self, scale_factor):
        scale_factor = np.asarray(scale_factor, dtype=float)
        if np.any(scale_factor <= 0.0):
            raise ValueError("scale factor must be positive")
        if self.omega_lambda == 0.0:
            return self.age_0 * scale_factor**1.5
        return 2.0 / (3.0 * self.hubble_0_gyr * np.sqrt(self.omega_lambda)) * np.arcsinh(
            np.sqrt(self.omega_lambda / self.omega_m) * scale_factor**1.5
        )

    def hubble(self, cosmic_time):
        a = self.scale_factor(cosmic_time)
        return self.hubble_0 * np.sqrt(self.omega_m / a**3 + self.omega_lambda)

    def critical_density(self, cosmic_time):
        h = self.hubble(cosmic_time)
        return 3.0 * h**2 / (8.0 * np.pi * _G_MPC_KMS_MSUN)

    def matter_density(self, cosmic_time):
        return (
            3.0 * self.hubble_0**2 * self.omega_m
            / (8.0 * np.pi * _G_MPC_KMS_MSUN)
            / self.scale_factor(cosmic_time) ** 3
        )

    def dark_energy_density(self, cosmic_time):
        return np.full_like(
            np.asarray(cosmic_time, dtype=float),
            3.0 * self.hubble_0**2 * self.omega_lambda
            / (8.0 * np.pi * _G_MPC_KMS_MSUN),
        )

    def background_density(self, cosmic_time):
        return self.matter_density(cosmic_time)

    def redshift(self, cosmic_time):
        return 1.0 / self.scale_factor(cosmic_time) - 1.0

    @property
    def type_name(self):
        return "lambda_cdm"


def make_cosmology(cosmology_type="einstein_de_sitter", **kwargs):
    """Construct an EdS or flat ΛCDM cosmology; EdS is the default."""
    if cosmology_type in (None, "einstein_de_sitter", "EinsteinDeSitter", "eds"):
        return EinsteinDeSitter(**kwargs)
    if cosmology_type in ("lambda_cdm", "LambdaCDM", "lcdm"):
        return LambdaCDM(**kwargs)
    raise ValueError("unsupported cosmology_type: %s" % cosmology_type)
