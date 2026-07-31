"""Analytic optically thin spherical radiation-density profiles."""

import numpy as np
import unyt

from radhydropy.constants import SPEED_OF_LIGHT_CGS


def finite_volume_density(boundary, volume, source_photon_rate):
    """Return the finite-volume average photon density."""

    dr = (boundary[1:] - boundary[:-1]).to(unyt.cm)
    volume = volume.to(unyt.cm**3)
    speed_of_light = SPEED_OF_LIGHT_CGS * unyt.cm / unyt.s
    return (source_photon_rate * dr / volume / speed_of_light).to(
        1.0 / unyt.cm**3
    )


def point_density(radius, source_photon_rate):
    """Return pointwise ``Q / (4 pi r^2 c)`` photon density."""

    speed_of_light = SPEED_OF_LIGHT_CGS * unyt.cm / unyt.s
    return (
        source_photon_rate
        / (4.0 * np.pi * radius.to(unyt.cm)**2 * speed_of_light)
    ).to(1.0 / unyt.cm**3)
