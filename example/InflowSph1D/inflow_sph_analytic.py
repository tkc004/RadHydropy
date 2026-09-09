"""Analytic reference profile for the spherical inflow example."""


def density_profile_proper_unyt(
    radius_proper_unyt, rho_inflow_proper_unyt, radius_reference_proper_unyt
):
    """Return the steady spherical inflow ``rho proportional r^-2`` profile."""

    return (
        rho_inflow_proper_unyt
        * radius_reference_proper_unyt**2
        / radius_proper_unyt**2
    )


def front_position_proper_unyt(
    radius_reference_proper_unyt, time_proper_unyt, vel_inflow_proper_unyt
):
    """Return the reference inflow-front position."""

    return radius_reference_proper_unyt + time_proper_unyt * vel_inflow_proper_unyt
