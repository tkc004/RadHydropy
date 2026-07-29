"""Hydrogen thermo-chemistry rates and source terms."""

import numpy as np
import unyt


DEFAULT_SIGMA_GAMMA = 1.62e-18 * unyt.cm**2
DEFAULT_EPSILON_GAMMA = 0.0 * unyt.erg
SPEED_OF_LIGHT = unyt.c.to(unyt.cm / unyt.s)


def photon_cross_section(sigma_gamma=DEFAULT_SIGMA_GAMMA):
    """Return the photon absorption cross-section in ``cm**2`` units."""
    if sigma_gamma is None:
        return DEFAULT_SIGMA_GAMMA
    if hasattr(sigma_gamma, "to"):
        return sigma_gamma.to(unyt.cm**2)
    return sigma_gamma * unyt.cm**2


def photon_excess_energy(epsilon_gamma=DEFAULT_EPSILON_GAMMA):
    """Return photoheating energy per ionization in ``erg`` units."""
    if epsilon_gamma is None:
        return DEFAULT_EPSILON_GAMMA
    if hasattr(epsilon_gamma, "to"):
        return epsilon_gamma.to(unyt.erg)
    return epsilon_gamma * unyt.erg


def clip_neutral_fraction(xHI):
    """Return neutral hydrogen fraction limited to the physical range."""
    return np.clip(np.asarray(xHI, dtype=float), 0.0, 1.0)


def mean_molecular_weight_mu(xHI, hydrogen_mass_fraction=1.0):
    """Return mean molecular weight for a pure H mixture with neutral fraction ``xHI``."""
    xHI = clip_neutral_fraction(xHI)
    return 1.0 / (hydrogen_mass_fraction * (2.0 - xHI))
