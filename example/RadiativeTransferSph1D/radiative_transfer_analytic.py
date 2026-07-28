"""Analytic optically thin spherical radiation-density profiles."""

import numpy as np
import unyt

import radhydropy.chemistry_species.hydrogen as rh


def finite_volume_density(boundary, volume, source_photon_rate):
    """Return the finite-volume average photon density."""

    dr = (boundary[1:] - boundary[:-1]).to(unyt.cm)
    volume = volume.to(unyt.cm**3)
    return (source_photon_rate * dr / volume / rh.SPEED_OF_LIGHT).to(
        1.0 / unyt.cm**3
    )


def point_density(radius, source_photon_rate):
    """Return pointwise ``Q / (4 pi r^2 c)`` photon density."""

    return (
        source_photon_rate
        / (4.0 * np.pi * radius.to(unyt.cm)**2 * rh.SPEED_OF_LIGHT)
    ).to(1.0 / unyt.cm**3)
