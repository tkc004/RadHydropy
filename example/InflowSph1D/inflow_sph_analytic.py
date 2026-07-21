"""Analytic reference profile for the spherical inflow example."""


def density_profile(radius, density_inflow, reference_radius):
    """Return the steady spherical inflow ``rho proportional r^-2`` profile."""

    return density_inflow * reference_radius**2 / radius**2


def front_position(reference_radius, time, velocity_inflow):
    """Return the reference inflow-front position."""

    return reference_radius + time * velocity_inflow
