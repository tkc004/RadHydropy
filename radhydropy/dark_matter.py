"""Spherical collisionless dark-matter shell dynamics."""

import numpy as np

from radhydropy.units import _code_units, _gravitational_constant_code, quantity_to_value


def enclosed_gas_mass(mesh, rho, radius, par):
    """Return spherical gas mass enclosed by arbitrary code-unit radii."""
    code_units = _code_units(par)
    if code_units is None:
        raise ValueError("gas mass coupling requires par.CodeUnits")
    radius = np.asarray(radius, dtype=float)
    boundaries = np.asarray(
        quantity_to_value(mesh.boundary, code_units.length_unit), dtype=float
    )
    density = np.asarray(
        quantity_to_value(rho, code_units.density_unit), dtype=float
    )
    first = par.noghost
    last = first + par.nogrid
    inner = boundaries[first:last]
    outer = boundaries[first + 1:last + 1]
    shell_volume = 4.0 * np.pi / 3.0 * (outer**3 - inner**3)
    shell_mass = density[first:last] * shell_volume
    prefix = np.concatenate(([0.0], np.cumsum(shell_mass)))
    clipped = np.clip(radius, inner[0], outer[-1])
    cell = np.searchsorted(outer, clipped, side="right")
    cell = np.clip(cell, 0, len(inner) - 1)
    before = prefix[cell]
    partial = density[first + cell] * 4.0 * np.pi / 3.0 * (
        clipped**3 - inner[cell]**3
    )
    result = before + np.maximum(partial, 0.0)
    result[radius <= inner[0]] = 0.0
    result[radius >= outer[-1]] = prefix[-1]
    return result


class DarkMatterShells:
    """Evolve infinitesimally thin spherical dark-matter shells.

    The shell arrays are kept sorted by radius.  Shell crossings are allowed:
    the shell identities (mass, radial velocity, and angular momentum) are
    exchanged in the sorted order after each drift.
    """

    def __init__(
        self,
        radius,
        velocity,
        mass,
        angular_momentum=None,
        softening=0.0,
        fixed_enclosed_mass=None,
        code_units=None,
    ):
        self.CodeUnits = code_units
        if code_units is None:
            raise ValueError("dark-matter shells require code_units")
        length = code_units.length_unit
        velocity_unit = code_units.velocity_unit
        mass_unit = code_units.mass_unit
        self.radius = np.asarray(quantity_to_value(radius, length), dtype=float).copy()
        self.velocity = np.asarray(quantity_to_value(velocity, velocity_unit), dtype=float).copy()
        self.mass = np.asarray(quantity_to_value(mass, mass_unit), dtype=float).copy()
        if angular_momentum is None:
            angular_momentum = np.zeros_like(self.radius)
        self.angular_momentum = np.asarray(
            quantity_to_value(angular_momentum, length * velocity_unit),
            dtype=float,
        ).copy()
        self.softening = float(quantity_to_value(softening, length))
        if callable(fixed_enclosed_mass):
            self.fixed_enclosed_mass = fixed_enclosed_mass
        elif fixed_enclosed_mass is None:
            self.fixed_enclosed_mass = None
        else:
            self.fixed_enclosed_mass = float(quantity_to_value(fixed_enclosed_mass, mass_unit))
        if not (
            self.radius.ndim == self.velocity.ndim == self.mass.ndim == self.angular_momentum.ndim == 1
        ):
            raise ValueError("dark-matter shell state must be one-dimensional")
        if not (
            self.radius.size == self.velocity.size == self.mass.size == self.angular_momentum.size
        ):
            raise ValueError("dark-matter shell arrays must have equal lengths")
        if np.any(self.radius <= 0.0):
            raise ValueError("dark-matter shell radii must be positive")
        if np.any(self.mass <= 0.0):
            raise ValueError("dark-matter shell masses must be positive")
        self.sort_by_radius()

    @property
    def number_of_shells(self):
        return self.radius.size

    @property
    def total_mass(self):
        return float(np.sum(self.mass))

    def sort_by_radius(self):
        """Sort shells by radius while preserving shell identities."""
        order = np.argsort(self.radius, kind="stable")
        self.radius = self.radius[order]
        self.velocity = self.velocity[order]
        self.mass = self.mass[order]
        self.angular_momentum = self.angular_momentum[order]
        return order

    def enclosed_mass(self, radius=None):
        """Return enclosed shell mass using half the mass at a shell radius."""
        if radius is None:
            radius = self.radius
        radius = np.asarray(radius, dtype=float)
        prefix = np.concatenate(([0.0], np.cumsum(self.mass)))
        left = np.searchsorted(self.radius, radius, side="left")
        right = np.searchsorted(self.radius, radius, side="right")
        result = prefix[left]
        result = result + 0.5 * (prefix[right] - prefix[left])
        return np.asarray(result, dtype=float)

    def gravitating_enclosed_mass(self, radius=None):
        """Return dynamic plus configured fixed enclosed mass."""
        if radius is None:
            radius = self.radius
        radius = np.asarray(radius, dtype=float)
        if self.fixed_enclosed_mass is None:
            return self.enclosed_mass(radius)
        if callable(self.fixed_enclosed_mass):
            fixed = np.asarray(self.fixed_enclosed_mass(radius), dtype=float)
        else:
            fixed = np.full_like(radius, self.fixed_enclosed_mass)
        return fixed

    def acceleration(
        self,
        gas_enclosed_mass=None,
        background_enclosed_mass=None,
        scale_factor=1.0,
        cosmological=False,
    ):
        """Return shell gravity and angular-momentum accelerations.

        In cosmological supercomoving coordinates, ``gas_enclosed_mass`` and
        shell masses are comoving masses, ``background_enclosed_mass`` is the
        homogeneous cosmological mass, and the gravitational term is
        ``-G*a*DeltaM/(x+softening)**2``.
        """
        g_code = _gravitational_constant_code(self.CodeUnits)
        enclosed = self.gravitating_enclosed_mass()
        if gas_enclosed_mass is not None:
            enclosed = enclosed + np.asarray(gas_enclosed_mass, dtype=float)
        if cosmological and background_enclosed_mass is not None:
            enclosed = enclosed - np.asarray(background_enclosed_mass, dtype=float)
        radius = np.maximum(self.radius, np.finfo(float).tiny)
        gravity = -g_code * float(scale_factor) * enclosed / (self.radius + self.softening) ** 2
        centrifugal = self.angular_momentum**2 / radius**3
        return gravity + centrifugal

    def crossing_timestep(self, safety_factor=0.1):
        """Return a timestep that stops before the first predicted crossing."""
        if self.number_of_shells < 2:
            return np.inf
        separation = self.radius[1:] - self.radius[:-1]
        closing_speed = self.velocity[:-1] - self.velocity[1:]
        candidates = separation[closing_speed > 0.0] / closing_speed[closing_speed > 0.0]
        if candidates.size == 0:
            return np.inf
        return float(safety_factor * np.min(candidates))

    def step(
        self,
        dt,
        crossing_safety_factor=0.1,
        gas_enclosed_mass=None,
        background_enclosed_mass=None,
        scale_factor=1.0,
        scale_factor_end=None,
        cosmological=False,
    ):
        """Advance one kick-drift-kick step, limiting ``dt`` before crossing."""
        dt = float(dt)
        crossing_dt = self.crossing_timestep(safety_factor=1.0)
        if crossing_dt < dt:
            # Advance just beyond the event so the stable sort exchanges the
            # shell records.  A strict safety factor would approach the event
            # asymptotically and never allow a crossing to occur.
            actual_dt = crossing_dt * (1.0 + max(crossing_safety_factor, 1.0e-10))
        else:
            actual_dt = dt
        acceleration = self.acceleration(
            gas_enclosed_mass=gas_enclosed_mass,
            background_enclosed_mass=background_enclosed_mass,
            scale_factor=scale_factor,
            cosmological=cosmological,
        )
        velocity_half = self.velocity + 0.5 * actual_dt * acceleration
        self.radius = self.radius + actual_dt * velocity_half
        self.velocity = velocity_half
        self.sort_by_radius()
        if scale_factor_end is None:
            scale_factor_end = scale_factor
        acceleration_new = self.acceleration(
            gas_enclosed_mass=gas_enclosed_mass,
            background_enclosed_mass=background_enclosed_mass,
            scale_factor=scale_factor_end,
            cosmological=cosmological,
        )
        self.velocity += 0.5 * actual_dt * acceleration_new
        return actual_dt

    def specific_energy(self):
        """Return a diagnostic specific energy for each shell.

        The potential uses the enclosed half-shell mass and is intended for
        monitoring, not as a substitute for a global pairwise energy audit.
        """
        g_code = _gravitational_constant_code(self.CodeUnits)
        radius = np.maximum(self.radius, np.finfo(float).tiny)
        kinetic = 0.5 * self.velocity**2
        angular = 0.5 * self.angular_momentum**2 / radius**2
        if self.fixed_enclosed_mass is None:
            enclosed = self.enclosed_mass()
        elif callable(self.fixed_enclosed_mass):
            enclosed = np.asarray(self.fixed_enclosed_mass(self.radius), dtype=float)
        else:
            enclosed = np.full_like(self.radius, self.fixed_enclosed_mass)
        potential = -g_code * enclosed / (self.radius + self.softening)
        return kinetic + angular + potential
