"""Analytic reference profile for the spherical outflow example."""


def density_profile_proper_unyt(
    radius_proper_unyt,
    rho_outflow_proper_unyt,
    radius_injection_proper_unyt,
):
    """Return the steady spherical outflow ``rho proportional r^-2`` profile."""

    return (
        rho_outflow_proper_unyt
        * radius_injection_proper_unyt**2
        / radius_proper_unyt**2
    )


def front_position_proper_unyt(time_proper_unyt, vel_outflow_proper_unyt):
    """Return the reference outflow-front position."""

    return time_proper_unyt * vel_outflow_proper_unyt
