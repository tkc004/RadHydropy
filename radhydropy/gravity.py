"""Gravity helpers for optional self-gravity and external potentials."""

import numpy as np
import unyt

from radhydropy.units import _code_units, to_code_value


GRAVITATIONAL_CONSTANT_CGS = float(
    unyt.physical_constants.gravitational_constant.to_value(
        unyt.cm**3 / (unyt.g * unyt.s**2)
    )
)


def _require_code_units(code_units):
    if code_units is None:
        raise ValueError("gravity helpers require code_units")
    return code_units


def _as_quantity(value, unit):
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(unit), dtype=float) * unit
    return np.asarray(value, dtype=float) * unit


def _as_float_array(value, unit):
    if value is None:
        raise ValueError("value must be provided")
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(unit), dtype=float)
    return np.asarray(value, dtype=float)


def _potential_unit(code_units):
    if code_units is None:
        return unyt.cm**2 / unyt.s**2
    return code_units.velocity_unit**2


def _acceleration_unit(code_units):
    if code_units is None:
        return unyt.cm / unyt.s**2
    return code_units.length_unit / code_units.time_unit**2


def _gravitational_constant_code(code_units):
    """Return the gravitational constant in the supplied code units."""
    code_units = _require_code_units(code_units)
    return (
        GRAVITATIONAL_CONSTANT_CGS
        * code_units.mass_in_cgs
        / (code_units.length_in_cgs * code_units.velocity_in_cgs**2)
    )


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
    radius_value = to_code_value(radius, code_units.length_unit)
    mass_value = to_code_value(mass, code_units.mass_unit)
    softening_value = to_code_value(softening, code_units.length_unit)
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
    radius_value = to_code_value(radius, code_units.length_unit)
    sigma_value = to_code_value(sigma, code_units.velocity_unit)
    reference_radius_value = to_code_value(reference_radius, code_units.length_unit)
    softening_value = to_code_value(softening, code_units.length_unit)
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
    radius_value = to_code_value(radius, code_units.length_unit)
    rho_s_value = to_code_value(rho_s, code_units.density_unit)
    r_s_value = to_code_value(r_s, code_units.length_unit)
    softening_value = to_code_value(softening, code_units.length_unit)
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
    """Store gravity settings and evaluate an optional external potential."""

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
        code_units = _require_code_units(_code_units(self))
        if coordinate is None:
            raise ValueError(f"{label} requires coordinates when it is tabulated")

        coord_unit = code_units.length_unit
        value_unit = _potential_unit(code_units) if label == "potential" else _acceleration_unit(code_units)
        values = to_code_value(values, value_unit)
        coordinate = to_code_value(coordinate, coord_unit)
        if self.coordinate is None:
            if np.shape(values) != np.shape(coordinate):
                raise ValueError(f"{label} requires a tabulated coordinate axis")
            return values

        grid = to_code_value(self.coordinate, coord_unit)
        if np.shape(grid) != np.shape(values):
            raise ValueError(f"{label} requires tabulated values with the same shape as coordinate")
        return np.interp(coordinate, grid, values)

    def potential_on(self, coordinate):
        """Return the external gravitational potential on ``coordinate``."""
        code_units = _require_code_units(_code_units(self))
        if self.potential is None:
            raise ValueError("No gravitational potential has been configured")
        if callable(self.potential):
            return to_code_value(self.potential(coordinate), _potential_unit(code_units))
        return self._tabulated_quantity(self.potential, coordinate, "potential")

    def acceleration_on(self, coordinate):
        """Return the gravitational acceleration on ``coordinate``."""
        code_units = _require_code_units(_code_units(self))
        if self.acceleration is not None:
            if callable(self.acceleration):
                return to_code_value(self.acceleration(coordinate), _acceleration_unit(code_units))
            return self._tabulated_quantity(self.acceleration, coordinate, "acceleration")

        if self.potential is None:
            raise ValueError("Either a potential or an acceleration must be configured")

        coord = to_code_value(coordinate, code_units.length_unit)
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

    def acceleration_on_mesh(self, mesh):
        """Return the acceleration evaluated on a mesh coordinate array."""
        if not hasattr(mesh, "coordinate"):
            raise AttributeError("mesh does not provide cell coordinates")
        return self.acceleration_on(mesh.coordinate)

    def force_density_on_mesh(self, mesh, rho):
        """Return the gravitational force density ``rho * g`` on a mesh."""
        return np.asarray(rho, dtype=float) * self.acceleration_on_mesh(mesh)
