"""Analytic reference profile for the spherical outflow example."""


def density_profile(radius, density_outflow, injection_radius):
    """Return the steady spherical outflow ``rho proportional r^-2`` profile."""

    return density_outflow * injection_radius**2 / radius**2


def front_position(time, velocity_outflow):
    """Return the reference outflow-front position."""

    return time * velocity_outflow
