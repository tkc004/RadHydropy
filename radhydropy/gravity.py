"""Gravity helpers for optional self-gravity and external potentials."""

import numpy as np
import unyt
from radhydropy.dark_matter import prepare_enclosed_gas_mass

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
        dark_matter=None,
        cosmological=False,
        cosmology=None,
    ):
        self.selfgravity = bool(selfgravity)
        self.externalgravity = bool(externalgravity)
        self.potential = potential
        self.coordinate = coordinate
        self.acceleration = acceleration
        self.CodeUnits = code_units
        self.selfgravity_softening = float(selfgravity_softening)
        self.selfgravity_boundary_acceleration = float(selfgravity_boundary_acceleration)
        self.dark_matter = dark_matter
        self.cosmological = bool(cosmological)
        self.cosmology = cosmology

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

    def cosmological_acceleration_on_mesh(self, mesh, rho, par):
        """Return supercomoving acceleration from enclosed density contrast.

        The mesh coordinate is comoving radius ``x`` and ``rho`` is comoving
        density ``varrho``. The homogeneous background is removed before
        applying the spherical Poisson equation,

        ``g_sc = -G * a * DeltaM(<x) / x**2``.
        """
        if not self.cosmological:
            return np.zeros_like(mesh.coordinate, dtype=float)
        if getattr(mesh, "coordsys", None) != "spherical":
            raise ValueError("cosmological gravity currently requires a spherical mesh")
        cosmology = self.cosmology or getattr(par, "cosmology", None)
        if cosmology is None:
            raise ValueError("cosmological gravity requires par.cosmology")
        if not getattr(par, "supercomoving_coordinates", False):
            raise ValueError("cosmological gravity requires supercomoving coordinates")
        code_units = _require_code_units(_code_units(self))
        density = np.asarray(
            quantity_to_value(rho, code_units.density_unit), dtype=float
        )
        boundaries = np.asarray(
            quantity_to_value(mesh.boundary, code_units.length_unit), dtype=float
        )
        coordinate = np.asarray(
            quantity_to_value(mesh.coordinate, code_units.length_unit), dtype=float
        )
        volume = np.asarray(
            quantity_to_value(mesh.vol, code_units.volume_unit), dtype=float
        )
        first = int(par.noghost)
        last = first + int(par.nogrid)
        interior = slice(first, last)
        if not (density.shape == coordinate.shape == volume.shape):
            raise ValueError("cosmological gravity inputs must match mesh shape")
        tau = float(np.asarray(getattr(par, "time", 0.0), dtype=float))
        # A simulation's fluid time is authoritative once the run has started.
        if hasattr(par, "fluid_time"):
            tau = float(np.asarray(par.fluid_time, dtype=float))
        cosmic_time = cosmology.cosmic_time_from_supercomoving(tau)
        scale_factor = float(cosmology.scale_factor_from_supercomoving(tau))
        background_physical = float(cosmology.background_density(cosmic_time))
        background_comoving = background_physical * scale_factor**3
        dm_fraction = float(getattr(par, "dark_matter_background_fraction", 0.0))
        gas_fraction = float(
            getattr(par, "gas_background_fraction", 1.0 - dm_fraction)
        )

        inner = np.maximum(boundaries[first:last], 0.0)
        outer = np.maximum(boundaries[first + 1:last + 1], 0.0)
        # ``rho`` is the gas density when live DM is coupled.  Subtract the
        # homogeneous gas share, not the total matter background; the live
        # shell term below supplies the separate (1-f_b) contribution.
        density_excess = density[interior] - gas_fraction * background_comoving
        shell_volume = 4.0 * np.pi / 3.0 * (outer**3 - inner**3)
        enclosed_before = np.concatenate(
            ([0.0], np.cumsum(density_excess[:-1] * shell_volume[:-1]))
        )
        radii = coordinate[interior]
        partial_volume = 4.0 * np.pi / 3.0 * np.maximum(
            radii**3 - inner**3, 0.0
        )
        enclosed_excess = enclosed_before + density_excess * partial_volume
        if self.dark_matter is not None:
            # ``density_excess`` has already removed the gas background.  Live
            # shells carry the full DM background plus the perturbation, so
            # remove the DM share here before adding their enclosed mass.
            dm_enclosed = self.dark_matter.gravitating_enclosed_mass(
                radii, include_shell_mass_with_fixed=True
            )
            dm_background = dm_fraction * background_comoving * (4.0 * np.pi / 3.0) * radii**3
            enclosed_excess += dm_enclosed - dm_background
        g_code = _gravitational_constant_code(code_units)
        radius = np.maximum(radii, np.finfo(float).tiny)
        result = np.zeros_like(coordinate)
        result[interior] = -g_code * scale_factor * enclosed_excess / radius**2
        origin = np.flatnonzero((inner <= 0.0) & (outer > 0.0))
        if origin.size:
            result[first + origin[0]] = 0.0
        return result

    def acceleration_on_mesh(self, mesh, rho=None, par=None):
        """Return the total external plus self-gravity acceleration."""
        if (
            not self.externalgravity
            and not self.selfgravity
            and self.dark_matter is None
        ):
            return np.zeros_like(mesh.coordinate, dtype=float)
        total = np.zeros_like(mesh.coordinate, dtype=float)
        if self.externalgravity:
            total += self.acceleration_on(mesh.coordinate)
        if self.selfgravity or self.cosmological:
            if rho is None or par is None:
                raise ValueError("rho and par are required when selfgravity is enabled")
            if self.cosmological:
                total += self.cosmological_acceleration_on_mesh(mesh, rho, par)
            else:
                total += self.self_acceleration_on_mesh(mesh, rho, par)
        if self.dark_matter is not None and not self.cosmological:
            total += self.dark_matter_acceleration_on_mesh(mesh, rho, par)
        return total

    def dark_matter_acceleration_on_mesh(self, mesh, rho, par):
        """Return the acceleration from live dark-matter shells."""
        if self.dark_matter is None:
            return np.zeros_like(mesh.coordinate, dtype=float)
        code_units = _require_code_units(_code_units(self))
        coordinate = np.asarray(
            quantity_to_value(mesh.coordinate, code_units.length_unit), dtype=float
        )
        enclosed = self.dark_matter.gravitating_enclosed_mass(coordinate)
        radius = np.maximum(coordinate, np.finfo(float).tiny)
        acceleration = -_gravitational_constant_code(code_units) * enclosed / (
            radius + self.dark_matter.softening
        ) ** 2
        return acceleration

    def advance_dark_matter(self, dt, mesh, rho, par, crossing_safety_factor=0.1):
        """Advance live dark-matter shells using the current gas mass field."""
        if self.dark_matter is None:
            return 0.0
        # The gas state is fixed during this gravity/source update.  Build its
        # cumulative mass profile once; shell sub-cycling only evaluates the
        # cached piecewise-constant profile at the current shell radii.
        gas_mass = prepare_enclosed_gas_mass(mesh, rho, par)
        if self.cosmological:
            cosmology = self.cosmology or getattr(par, "cosmology", None)
            if cosmology is None or not getattr(par, "supercomoving_coordinates", False):
                raise ValueError(
                    "cosmological dark-matter shells require supercomoving cosmology"
                )
            tau = float(np.asarray(getattr(par, "fluid_time", getattr(par, "time", 0.0)), dtype=float))
            cosmic_time = cosmology.cosmic_time_from_supercomoving(tau)
            scale_factor = float(cosmology.scale_factor_from_supercomoving(tau))
            tau_start = tau - float(dt)
            if tau_start < 0.0:
                tau_start = tau
            scale_factor_start = float(cosmology.scale_factor_from_supercomoving(tau_start))
            background_density = float(cosmology.background_density(cosmic_time)) * scale_factor**3
            # Re-evaluate the homogeneous mass at the shell radius used by
            # each kick.  Passing a frozen array here applies the old-radius
            # background after the drift and corrupts linear growth.
            background_mass = lambda radius: (
                4.0 * np.pi / 3.0 * background_density * np.asarray(radius)**3
            )
        else:
            scale_factor = 1.0
            scale_factor_start = 1.0
            background_mass = None
        return self.dark_matter.step(
            dt,
            crossing_safety_factor=crossing_safety_factor,
            crossing_batch_fraction=getattr(
                par, "dark_matter_crossing_batch_fraction", 0.0
            ),
            gas_enclosed_mass=gas_mass,
            background_enclosed_mass=background_mass,
            scale_factor=scale_factor_start,
            scale_factor_end=scale_factor,
            cosmological=self.cosmological,
            include_shell_mass_with_fixed=self.cosmological,
        )

    def force_density_on_mesh(self, mesh, rho):
        """Return the gravitational force density ``rho * g`` on a mesh."""
        return np.asarray(rho, dtype=float) * self.acceleration_on_mesh(mesh)
