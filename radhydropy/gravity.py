"""Gravity helpers for optional self-gravity and external potentials."""

import numpy as np
import unyt

from radhydropy.units import (
    _acceleration_unit,
    _gravitational_constant_code,
    _potential_unit,
    _code_units,
    quantity_to_value,
)


def _require_code_units(code_units):
    if code_units is None:
        raise ValueError("gravity helpers require code_units")
    return code_units


def _as_quantity(value, unit):
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(unit), dtype=float) * unit
    return np.asarray(value, dtype=float) * unit


def point_mass_potential(radius, mass, softening=0.0 * unyt.cm, code_units=None):
    r"""Return the gravitational potential of a softened point mass."""
    if code_units is None:
        radius_q = _as_quantity(radius, unyt.cm)
        mass_q = _as_quantity(mass, unyt.g)
        softening_q = _as_quantity(softening, unyt.cm)
        radius_eff = np.maximum(radius_q, softening_q)
        return (-unyt.physical_constants.gravitational_constant * mass_q / radius_eff).to(
            _potential_unit(None)
        )
    radius_value = quantity_to_value(radius, code_units.length_unit)
    mass_value = quantity_to_value(mass, code_units.mass_unit)
    softening_value = quantity_to_value(softening, code_units.length_unit)
    radius_eff = np.maximum(radius_value, softening_value)
    potential_value = -_gravitational_constant_code(code_units) * mass_value / radius_eff
    return potential_value * _potential_unit(code_units)


def singular_isothermal_potential(
    radius,
    sigma,
    reference_radius=1.0 * unyt.cm,
    softening=0.0 * unyt.cm,
    code_units=None,
):
    r"""Return the potential for a singular isothermal sphere."""
    if code_units is None:
        radius_q = _as_quantity(radius, unyt.cm)
        sigma_q = _as_quantity(sigma, unyt.cm / unyt.s)
        reference_radius_q = _as_quantity(reference_radius, unyt.cm)
        softening_q = _as_quantity(softening, unyt.cm)
        radius_eff = np.maximum(radius_q, softening_q)
        return (2.0 * sigma_q**2 * np.log(radius_eff / reference_radius_q)).to(
            _potential_unit(None)
        )
    radius_value = quantity_to_value(radius, code_units.length_unit)
    sigma_value = quantity_to_value(sigma, code_units.velocity_unit)
    reference_radius_value = quantity_to_value(reference_radius, code_units.length_unit)
    softening_value = quantity_to_value(softening, code_units.length_unit)
    radius_eff = np.maximum(radius_value, softening_value)
    potential_value = 2.0 * sigma_value**2 * np.log(radius_eff / reference_radius_value)
    return potential_value * _potential_unit(code_units)


def nfw_potential(
    radius,
    rho_s,
    r_s,
    softening=0.0 * unyt.cm,
    code_units=None,
):
    r"""Return the gravitational potential for an NFW halo."""
    if code_units is None:
        radius_q = _as_quantity(radius, unyt.cm)
        rho_s_q = _as_quantity(rho_s, unyt.g / unyt.cm**3)
        r_s_q = _as_quantity(r_s, unyt.cm)
        softening_q = _as_quantity(softening, unyt.cm)
        radius_eff = np.maximum(radius_q, softening_q)
        x = radius_eff / r_s_q
        x_value = np.asarray(x.to_value(unyt.dimensionless), dtype=float)
        log_over_x = np.ones_like(x_value)
        nonzero = x_value != 0.0
        log_over_x[nonzero] = np.log1p(x_value[nonzero]) / x_value[nonzero]
        potential = (
            -4.0
            * np.pi
            * unyt.physical_constants.gravitational_constant
            * rho_s_q
            * r_s_q**2
            * log_over_x
        )
        return potential.to(_potential_unit(None))
    radius_value = quantity_to_value(radius, code_units.length_unit)
    rho_s_value = quantity_to_value(rho_s, code_units.density_unit)
    r_s_value = quantity_to_value(r_s, code_units.length_unit)
    softening_value = quantity_to_value(softening, code_units.length_unit)
    radius_eff = np.maximum(radius_value, softening_value)
    x_value = np.asarray(radius_eff / r_s_value, dtype=float)
    log_over_x = np.ones_like(x_value)
    nonzero = x_value != 0.0
    log_over_x[nonzero] = np.log1p(x_value[nonzero]) / x_value[nonzero]
    potential_value = (
        -4.0
        * np.pi
        * _gravitational_constant_code(code_units)
        * rho_s_value
        * r_s_value**2
        * log_over_x
    )
    return potential_value * _potential_unit(code_units)


class Gravity:
    """Evaluate external and gas self-gravity fields in code units."""

    def __init__(
        self,
        selfgravity=0,
        externalgravity=0,
        potential=None,
        coordinate=None,
        acceleration=None,
        code_units=None,
        selfgravity_softening=0.0,
        selfgravity_boundary_acceleration=0.0,
    ):
        self.selfgravity = bool(selfgravity)
        self.externalgravity = bool(externalgravity)
        self.potential = potential
        self.coordinate = coordinate
        self.acceleration = acceleration
        self.CodeUnits = code_units
        self.selfgravity_softening = float(selfgravity_softening)
        self.selfgravity_boundary_acceleration = float(selfgravity_boundary_acceleration)

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
        code_units = _require_code_units(_code_units(self))
        if coordinate is None:
            raise ValueError(f"{label} requires coordinates when it is tabulated")

        coord_unit = code_units.length_unit
        value_unit = _potential_unit(code_units) if label == "potential" else _acceleration_unit(code_units)
        values = quantity_to_value(values, value_unit)
        coordinate = quantity_to_value(coordinate, coord_unit)
        if self.coordinate is None:
            if np.shape(values) != np.shape(coordinate):
                raise ValueError(f"{label} requires a tabulated coordinate axis")
            return values

        grid = quantity_to_value(self.coordinate, coord_unit)
        if np.shape(grid) != np.shape(values):
            raise ValueError(f"{label} requires tabulated values with the same shape as coordinate")
        return np.interp(coordinate, grid, values)

    def potential_on(self, coordinate):
        """Return the external gravitational potential on ``coordinate``."""
        code_units = _require_code_units(_code_units(self))
        if self.potential is None:
            raise ValueError("No gravitational potential has been configured")
        if callable(self.potential):
            return quantity_to_value(self.potential(coordinate), _potential_unit(code_units))
        return self._tabulated_quantity(self.potential, coordinate, "potential")

    def acceleration_on(self, coordinate):
        """Return the gravitational acceleration on ``coordinate``."""
        code_units = _require_code_units(_code_units(self))
        if self.acceleration is not None:
            if callable(self.acceleration):
                return quantity_to_value(self.acceleration(coordinate), _acceleration_unit(code_units))
            return self._tabulated_quantity(self.acceleration, coordinate, "acceleration")

        if self.potential is None:
            raise ValueError("Either a potential or an acceleration must be configured")

        coord = quantity_to_value(coordinate, code_units.length_unit)
        potential = self.potential_on(coord)
        if potential.size < 2:
            raise ValueError("At least two coordinate points are required to differentiate the potential")

        gradient = np.gradient(potential, coord)
        return -gradient

    def potential_on_mesh(self, mesh):
        """Return the potential evaluated on a mesh coordinate array."""
        if not hasattr(mesh, "coordinate"):
            raise AttributeError("mesh does not provide cell coordinates")
        return self.potential_on(mesh.coordinate)

    def self_acceleration_on_mesh(self, mesh, rho, par):
        """Return the gas self-gravity acceleration on a one-dimensional mesh.

        Spherical meshes use the enclosed gas mass. Cartesian meshes use the
        plane-parallel relation ``dg/dx = -4 pi G rho`` and therefore require
        ``selfgravity_boundary_acceleration`` as the left-boundary field.
        """
        code_units = _require_code_units(_code_units(self))
        rho = np.asarray(quantity_to_value(rho, code_units.density_unit), dtype=float)
        coordinate = np.asarray(quantity_to_value(mesh.coordinate, code_units.length_unit), dtype=float)
        volume = np.asarray(quantity_to_value(mesh.vol, code_units.volume_unit), dtype=float)
        if rho.shape != coordinate.shape or volume.shape != coordinate.shape:
            raise ValueError("self-gravity inputs must match the mesh cell shape")

        first = par.noghost
        last = first + par.nogrid
        interior = slice(first, last)
        result = np.zeros_like(coordinate)
        g_code = _gravitational_constant_code(code_units)

        if mesh.coordsys == "spherical":
            boundaries = np.asarray(
                quantity_to_value(mesh.boundary, code_units.length_unit),
                dtype=float,
            )
            radii = coordinate[interior]
            inner = np.maximum(boundaries[first:last], 0.0)
            outer = np.maximum(boundaries[first + 1:last + 1], 0.0)
            density = rho[interior]
            enclosed_before = np.concatenate(([0.0], np.cumsum(density[:-1] * volume[interior][:-1])))
            radius = np.maximum(radii, self.selfgravity_softening)
            partial_volume = 4.0 * np.pi / 3.0 * np.maximum(radius**3 - inner**3, 0.0)
            enclosed_mass = enclosed_before + density * partial_volume
            result[interior] = -g_code * enclosed_mass / np.maximum(radius**2, np.finfo(float).tiny)
            # The field at the cell containing the spherical origin is zero by symmetry.
            origin = np.flatnonzero((inner <= 0.0) & (outer > 0.0))
            if origin.size:
                result[first + origin[0]] = 0.0
            return result

        if mesh.coordsys == "cartesian":
            result[first] = self.selfgravity_boundary_acceleration
            dx = np.asarray(quantity_to_value(mesh.xdelta, code_units.length_unit), dtype=float)
            for index in range(first + 1, last):
                result[index] = result[index - 1] - 4.0 * np.pi * g_code * rho[index - 1] * dx[index - 1]
            if first < last:
                result[last:] = result[last - 1]
            return result

        raise ValueError("self-gravity is not implemented for %r meshes" % mesh.coordsys)

    def acceleration_on_mesh(self, mesh, rho=None, par=None):
        """Return the total external plus self-gravity acceleration."""
        if not self.externalgravity and not self.selfgravity:
            return np.zeros_like(mesh.coordinate, dtype=float)
        total = np.zeros_like(mesh.coordinate, dtype=float)
        if self.externalgravity:
            total += self.acceleration_on(mesh.coordinate)
        if self.selfgravity:
            if rho is None or par is None:
                raise ValueError("rho and par are required when selfgravity is enabled")
            total += self.self_acceleration_on_mesh(mesh, rho, par)
        return total

    def force_density_on_mesh(self, mesh, rho):
        """Return the gravitational force density ``rho * g`` on a mesh."""
        return np.asarray(rho, dtype=float) * self.acceleration_on_mesh(mesh)
