"""Hydrogen thermo-chemistry rates and source terms."""

import numpy as np

from radhydropy.constants import (
    DEFAULT_EPSILON_GAMMA,
    DEFAULT_SIGMA_GAMMA,
)


def photon_cross_section(sigma_gamma=DEFAULT_SIGMA_GAMMA):
    """Return the photon absorption cross-section in cgs ``cm**2`` units."""
    if sigma_gamma is None:
        return DEFAULT_SIGMA_GAMMA
    return np.asarray(sigma_gamma, dtype=float)


def photon_excess_energy(epsilon_gamma=DEFAULT_EPSILON_GAMMA):
    """Return photoheating energy per ionization in cgs ``erg`` units."""
    if epsilon_gamma is None:
        return DEFAULT_EPSILON_GAMMA
    return np.asarray(epsilon_gamma, dtype=float)


def clip_neutral_fraction(xHI):
    """Return neutral hydrogen fraction limited to the physical range."""
    return np.clip(np.asarray(xHI, dtype=float), 0.0, 1.0)


def mean_molecular_weight_mu(xHI, hydrogen_mass_fraction=1.0):
    """Return mean molecular weight for a pure H mixture with neutral fraction ``xHI``."""
    xHI = clip_neutral_fraction(xHI)
    return 1.0 / (hydrogen_mass_fraction * (2.0 - xHI))
