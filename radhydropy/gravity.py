"""Gravity helpers for optional self-gravity and external potentials."""

import numpy as np
import unyt

from radhydropy.units import _code_units, _to_code_quantity


def point_mass_potential(radius, mass, softening=0.0 * unyt.cm, code_units=None):
    r"""Return the gravitational potential of a softened point mass.

    The potential is

    .. math::

       \Phi(r) = -\frac{GM}{\max(r, \epsilon)}.

    Parameters
    ----------
    radius : array-like or ``unyt`` quantity
        Radius at which to evaluate the potential.
    mass : array-like or ``unyt`` quantity
        Point mass.
    softening : array-like or ``unyt`` quantity, optional
        Small radius floor used to avoid the singularity at ``r = 0``.
    """
    length_unit = code_units.length_unit if code_units is not None else unyt.cm
    mass_unit = code_units.mass_unit if code_units is not None else unyt.g
    potential_unit = (
        code_units.velocity_unit**2
        if code_units is not None
        else unyt.cm**2 / unyt.s**2
    )
    radius = _to_code_quantity(radius, length_unit)
    mass = _to_code_quantity(mass, mass_unit)
    softening = _to_code_quantity(softening, length_unit)
    radius_eff = np.maximum(radius, softening)
    return (-unyt.physical_constants.gravitational_constant * mass / radius_eff).to(
        potential_unit
    )


def singular_isothermal_potential(radius, sigma, reference_radius=1.0 * unyt.cm, softening=0.0 * unyt.cm, code_units=None):
    r"""Return the potential for a singular isothermal sphere.

    The potential is defined up to an additive constant as

    .. math::

       \Phi(r) = 2 \sigma^2 \ln\left(\frac{r}{r_0}\right).

    Parameters
    ----------
    radius : array-like or ``unyt`` quantity
        Radius at which to evaluate the potential.
    sigma : array-like or ``unyt`` quantity
        One-dimensional velocity dispersion.
    reference_radius : array-like or ``unyt`` quantity, optional
        Fiducial radius ``r_0`` that sets the zero-point of the potential.
    softening : array-like or ``unyt`` quantity, optional
        Small radius floor used to avoid the logarithmic singularity.
    """
    length_unit = code_units.length_unit if code_units is not None else unyt.cm
    velocity_unit = (
        code_units.velocity_unit if code_units is not None else unyt.cm / unyt.s
    )
    potential_unit = (
        code_units.velocity_unit**2
        if code_units is not None
        else unyt.cm**2 / unyt.s**2
    )
    radius = _to_code_quantity(radius, length_unit)
    sigma = _to_code_quantity(sigma, velocity_unit)
    reference_radius = _to_code_quantity(reference_radius, length_unit)
    softening = _to_code_quantity(softening, length_unit)
    radius_eff = np.maximum(radius, softening)
    return (2.0 * sigma**2 * np.log(radius_eff / reference_radius)).to(
        potential_unit
    )


def nfw_potential(
    radius,
    rho_s,
    r_s,
    softening=0.0 * unyt.cm,
    code_units=None,
):
    r"""Return the gravitational potential for an NFW halo.

    The potential is written in the common zero-at-infinity convention,

    .. math::

       \Phi(r) = -4 \pi G \rho_s r_s^2 \frac{\ln(1 + x)}{x},

    where ``x = r / r_s`` and the ``x -> 0`` limit is ``1``.

    Parameters
    ----------
    radius : array-like or ``unyt`` quantity
        Radius at which to evaluate the potential.
    rho_s : array-like or ``unyt`` quantity
        Characteristic NFW density.
    r_s : array-like or ``unyt`` quantity
        NFW scale radius.
    softening : array-like or ``unyt`` quantity, optional
        Small radius floor used to avoid the numerical singularity at
    ``r = 0`` when evaluating ``ln(1 + x) / x``.
    """
    length_unit = code_units.length_unit if code_units is not None else unyt.cm
    density_unit = (
        code_units.density_unit if code_units is not None else unyt.g / unyt.cm**3
    )
    potential_unit = (
        code_units.velocity_unit**2
        if code_units is not None
        else unyt.cm**2 / unyt.s**2
    )
    radius = _to_code_quantity(radius, length_unit)
    rho_s = _to_code_quantity(rho_s, density_unit)
    r_s = _to_code_quantity(r_s, length_unit)
    softening = _to_code_quantity(softening, length_unit)
    radius_eff = np.maximum(radius, softening)
    x = radius_eff / r_s
    x_value = np.asarray(x.to_value(unyt.dimensionless), dtype=float)
    log_over_x = np.ones_like(x_value)
    nonzero = x_value != 0.0
    log_over_x[nonzero] = np.log1p(x_value[nonzero]) / x_value[nonzero]
    potential = -4.0 * np.pi * unyt.physical_constants.gravitational_constant * rho_s * r_s**2 * log_over_x
    return potential.to(potential_unit)


class Gravity:
    """Store gravity settings and evaluate an optional external potential.

    Parameters
    ----------
    selfgravity : bool, optional
        Flag reserved for future self-gravity support.
    externalgravity : bool, optional
        Enable an externally supplied gravitational potential.
    potential : callable, array-like, or ``unyt`` quantity, optional
        Gravitational potential ``Phi``. When an array is supplied, it is
        interpreted on ``coordinate``.
    coordinate : array-like or ``unyt`` quantity, optional
        Coordinate values associated with a tabulated potential.
    acceleration : callable, array-like, or ``unyt`` quantity, optional
        Direct gravitational acceleration. When omitted, it is derived from the
        potential via ``g = -dPhi/dx``.
    """

    def __init__(
        self,
        selfgravity=0,
        externalgravity=0,
        potential=None,
        coordinate=None,
        acceleration=None,
        code_units=None,
    ):
        self.selfgravity = bool(selfgravity)
        self.externalgravity = bool(externalgravity)
        self.potential = potential
        self.coordinate = coordinate
        self.acceleration = acceleration
        self.code_units = code_units

    def has_external_field(self):
        """Return ``True`` when an external field has been configured."""
        return self.externalgravity and (self.potential is not None or self.acceleration is not None)

    def set_potential(self, potential, coordinate=None):
        """Update the stored gravitational potential."""
        self.potential = potential
        if coordinate is not None:
            self.coordinate = coordinate

    def set_acceleration(self, acceleration):
        """Update the stored gravitational acceleration."""
        self.acceleration = acceleration

    def _tabulated_quantity(self, values, coordinate, label):
        """Interpolate a tabulated quantity onto the requested coordinates."""
        if coordinate is None:
            raise ValueError(f"{label} requires coordinates when it is tabulated")
        code_units = _code_units(self)
        coord_unit = code_units.length_unit if code_units is not None else unyt.cm
        value_unit = (
            code_units.velocity_unit**2
            if label == "potential"
            else (
                code_units.length_unit / code_units.time_unit**2
                if code_units is not None
                else unyt.cm / unyt.s**2
            )
        )
        values = _to_code_quantity(values, value_unit)
        coord = _to_code_quantity(coordinate, coord_unit)
        if self.coordinate is None:
            if np.shape(values) != np.shape(coord):
                raise ValueError(f"{label} requires a tabulated coordinate axis")
            return values.to(value_unit)
        grid = _to_code_quantity(self.coordinate, coord_unit)
        grid_values = np.asarray(grid.to_value(coord_unit), dtype=float)
        sample_values = np.asarray(values.to_value(value_unit), dtype=float)
        target_values = np.asarray(coord.to_value(coord_unit), dtype=float)
        interpolated = np.interp(target_values, grid_values, sample_values)
        return interpolated * value_unit

    def potential_on(self, coordinate):
        """Return the external gravitational potential on ``coordinate``."""
        if self.potential is None:
            raise ValueError("No gravitational potential has been configured")
        if callable(self.potential):
            code_units = _code_units(self)
            potential_unit = (
                code_units.velocity_unit**2
                if code_units is not None
                else unyt.cm**2 / unyt.s**2
            )
            return _to_code_quantity(self.potential(coordinate), potential_unit)
        return self._tabulated_quantity(self.potential, coordinate, "potential")

    def acceleration_on(self, coordinate):
        """Return the gravitational acceleration on ``coordinate``."""
        code_units = _code_units(self)
        acc_unit = (
            code_units.length_unit / code_units.time_unit**2
            if code_units is not None
            else unyt.cm / unyt.s**2
        )
        if self.acceleration is not None:
            if callable(self.acceleration):
                return _to_code_quantity(self.acceleration(coordinate), acc_unit)
            if hasattr(self.acceleration, "units"):
                return self._tabulated_quantity(self.acceleration, coordinate, "acceleration")
            return np.asarray(self.acceleration, dtype=float) * acc_unit

        if self.potential is None:
            raise ValueError("Either a potential or an acceleration must be configured")

        coord_unit = code_units.length_unit if code_units is not None else unyt.cm
        coord = _to_code_quantity(coordinate, coord_unit)
        potential = self.potential_on(coord)
        coord_values = np.asarray(coord.to_value(coord.units), dtype=float)
        potential_values = np.asarray(potential.to_value(potential.units), dtype=float)

        if potential_values.size < 2:
            raise ValueError("At least two coordinate points are required to differentiate the potential")

        gradient = np.gradient(potential_values, coord_values)
        return (-gradient) * potential.units / coord.units

    def potential_on_mesh(self, mesh):
        """Return the potential evaluated on a mesh coordinate array."""
        if not hasattr(mesh, "coordinate"):
            raise AttributeError("mesh does not provide cell coordinates")
        return self.potential_on(mesh.coordinate)

    def acceleration_on_mesh(self, mesh):
        """Return the acceleration evaluated on a mesh coordinate array."""
        if not hasattr(mesh, "coordinate"):
            raise AttributeError("mesh does not provide cell coordinates")
        return self.acceleration_on(mesh.coordinate)

    def force_density_on_mesh(self, mesh, rho):
        """Return the gravitational force density ``rho * g`` on a mesh."""
        return rho * self.acceleration_on_mesh(mesh)
