"""Spherical collisionless dark-matter shell dynamics."""

import numpy as np

from radhydropy.units import _code_units, _gravitational_constant_code, quantity_to_value


class EnclosedGasMassProfile:
    """Piecewise-constant spherical gas-mass profile for one hydro state.

    The gas density is unchanged while live dark-matter sub-cycling advances
    the shells.  Store the cell geometry and cumulative cell masses once, and
    only evaluate the partial-cell contribution for each new shell radius.
    """

    def __init__(self, mesh, rho, par):
        code_units = _code_units(par)
        if code_units is None:
            raise ValueError("gas mass coupling requires par.CodeUnits")
        boundaries = np.asarray(
            quantity_to_value(mesh.boundary, code_units.length_unit), dtype=float
        )
        density = np.asarray(
            quantity_to_value(rho, code_units.density_unit), dtype=float
        )
        first = int(par.noghost)
        last = first + int(par.nogrid)
        self.inner = boundaries[first:last]
        self.outer = boundaries[first + 1:last + 1]
        self.density = density[first:last]
        shell_volume = 4.0 * np.pi / 3.0 * (
            self.outer**3 - self.inner**3
        )
        shell_mass = self.density * shell_volume
        self.prefix = np.concatenate(([0.0], np.cumsum(shell_mass)))

    def __call__(self, radius):
        radius = np.asarray(radius, dtype=float)
        clipped = np.clip(radius, self.inner[0], self.outer[-1])
        cell = np.searchsorted(self.outer, clipped, side="right")
        cell = np.clip(cell, 0, len(self.inner) - 1)
        before = self.prefix[cell]
        partial = self.density[cell] * 4.0 * np.pi / 3.0 * (
            clipped**3 - self.inner[cell]**3
        )
        result = before + np.maximum(partial, 0.0)
        return np.where(
            radius <= self.inner[0],
            0.0,
            np.where(radius >= self.outer[-1], self.prefix[-1], result),
        )


def prepare_enclosed_gas_mass(mesh, rho, par):
    """Build a reusable enclosed-gas-mass profile for one hydro state."""
    return EnclosedGasMassProfile(mesh, rho, par)


def enclosed_gas_mass(mesh, rho, radius, par):
    """Return spherical gas mass enclosed by arbitrary code-unit radii."""
    return EnclosedGasMassProfile(mesh, rho, par)(radius)


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
        central_core_radius=0.0,
        core_absorption_velocity=0.0,
        core_absorption_energy=0.0,
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
        self.central_core_radius = float(quantity_to_value(central_core_radius, length))
        self.core_absorption_velocity = float(
            quantity_to_value(core_absorption_velocity, velocity_unit)
        )
        if self.central_core_radius < 0.0:
            raise ValueError("central core radius must be non-negative")
        if self.core_absorption_velocity < 0.0:
            raise ValueError("core absorption velocity must be non-negative")
        self.core_absorption_energy = float(
            quantity_to_value(core_absorption_energy, velocity_unit**2)
        )
        if callable(fixed_enclosed_mass):
            self.fixed_enclosed_mass = fixed_enclosed_mass
        elif fixed_enclosed_mass is None:
            self.fixed_enclosed_mass = None
        else:
            self.fixed_enclosed_mass = float(quantity_to_value(fixed_enclosed_mass, mass_unit))
        self.central_core_mass = (
            0.0 if self.fixed_enclosed_mass is None or callable(self.fixed_enclosed_mass)
            else float(self.fixed_enclosed_mass)
        )
        self._mass_prefix_cache = None
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
        if self.radius.size < 2 or np.all(self.radius[1:] >= self.radius[:-1]):
            return np.arange(self.radius.size)
        order = np.argsort(self.radius, kind="stable")
        self.radius = self.radius[order]
        self.velocity = self.velocity[order]
        self.mass = self.mass[order]
        self.angular_momentum = self.angular_momentum[order]
        self._mass_prefix_cache = None
        return order

    def _mass_prefix(self):
        """Return the cached cumulative shell-mass prefix."""
        if self._mass_prefix_cache is None:
            self._mass_prefix_cache = np.concatenate(([0.0], np.cumsum(self.mass)))
        return self._mass_prefix_cache

    def enclosed_mass(self, radius=None):
        """Return enclosed shell mass using half the mass at a shell radius."""
        if radius is None:
            # Shells are kept sorted by radius.  When callers request the
            # enclosed mass at every current shell radius (the hot path in
            # ``acceleration``), the shell indices are already known, so
            # avoid two searchsorted calls per shell.  For coincident shells,
            # assign half of the total coincident-group mass to every shell,
            # matching the generic arbitrary-radius convention.
            prefix = self._mass_prefix()
            starts = np.flatnonzero(
                np.r_[True, self.radius[1:] > self.radius[:-1]]
            )
            ends = np.r_[starts[1:], self.number_of_shells]
            group_values = prefix[starts] + 0.5 * (
                prefix[ends] - prefix[starts]
            )
            return np.repeat(group_values, ends - starts)
        radius = np.asarray(radius, dtype=float)
        prefix = self._mass_prefix()
        left = np.searchsorted(self.radius, radius, side="left")
        right = np.searchsorted(self.radius, radius, side="right")
        result = prefix[left]
        result = result + 0.5 * (prefix[right] - prefix[left])
        return np.asarray(result, dtype=float)

    def _enclosed_mass_at_current_positions(self, radius):
        """Return shell mass enclosed at post-drift shell positions.

        During ``step`` candidate positions can be temporarily out of order
        before shell identities are sorted.  Sort positions together with
        their masses, evaluate coincident groups in one pass, and map the
        result back to the original shell order.
        """
        radius = np.asarray(radius, dtype=float)
        order = np.argsort(radius, kind="stable")
        sorted_radius = radius[order]
        sorted_mass = self.mass[order]
        prefix = np.concatenate(([0.0], np.cumsum(sorted_mass)))
        starts = np.flatnonzero(
            np.r_[True, sorted_radius[1:] > sorted_radius[:-1]]
        )
        ends = np.r_[starts[1:], radius.size]
        group_values = prefix[starts] + 0.5 * (
            prefix[ends] - prefix[starts]
        )
        sorted_result = np.repeat(group_values, ends - starts)
        result = np.empty_like(sorted_result)
        result[order] = sorted_result
        return result

    def gravitating_enclosed_mass(self, radius=None, include_shell_mass_with_fixed=False):
        """Return dynamic plus configured fixed enclosed mass."""
        if radius is None:
            dynamic = self.enclosed_mass()
            if self.fixed_enclosed_mass is None:
                return dynamic
            radius = self.radius
            if callable(self.fixed_enclosed_mass):
                fixed = np.asarray(self.fixed_enclosed_mass(radius), dtype=float)
            else:
                fixed = np.full_like(radius, self.fixed_enclosed_mass)
            if include_shell_mass_with_fixed:
                return dynamic + fixed
            return fixed
        radius = np.asarray(radius, dtype=float)
        if self.fixed_enclosed_mass is None:
            return self.enclosed_mass(radius)
        if callable(self.fixed_enclosed_mass):
            fixed = np.asarray(self.fixed_enclosed_mass(radius), dtype=float)
        else:
            fixed = np.full_like(radius, self.fixed_enclosed_mass)
        if include_shell_mass_with_fixed:
            return fixed + self.enclosed_mass(radius)
        return fixed

    def acceleration(
        self,
        gas_enclosed_mass=None,
        background_enclosed_mass=None,
        scale_factor=1.0,
        cosmological=False,
        include_shell_mass_with_fixed=False,
    ):
        """Return shell gravity and angular-momentum accelerations.

        In cosmological supercomoving coordinates, ``gas_enclosed_mass`` and
        shell masses are comoving masses, ``background_enclosed_mass`` is the
        homogeneous cosmological mass, and the gravitational term is
        ``-G*a*DeltaM/(x+softening)**2``.
        """
        g_code = _gravitational_constant_code(self.CodeUnits)
        enclosed = self.gravitating_enclosed_mass(
            include_shell_mass_with_fixed=include_shell_mass_with_fixed
        )
        if gas_enclosed_mass is not None:
            if callable(gas_enclosed_mass):
                gas_mass = np.asarray(gas_enclosed_mass(self.radius), dtype=float)
            else:
                gas_mass = np.asarray(gas_enclosed_mass, dtype=float)
            enclosed = enclosed + gas_mass
        if cosmological and background_enclosed_mass is not None:
            if callable(background_enclosed_mass):
                background = np.asarray(
                    background_enclosed_mass(self.radius), dtype=float)
            else:
                background = np.asarray(background_enclosed_mass, dtype=float)
            enclosed = enclosed - background
        radius = np.maximum(self.radius, np.finfo(float).tiny)
        gravity = -g_code * float(scale_factor) * enclosed / (self.radius + self.softening) ** 2
        # Apply the same softening scale to the centrifugal term as to the
        # radial gravitational term.  Without this, a shell that crosses the
        # origin receives an unbounded j^2/r^3 kick and is launched to an
        # unphysical radius on the next drift.
        effective_radius = radius + self.softening
        centrifugal = np.divide(
            self.angular_momentum**2,
            effective_radius**3,
            out=np.zeros_like(radius),
            where=effective_radius > 0.0,
        )
        return gravity + centrifugal

    def crossing_timestep(self, safety_factor=0.1):
        """Return a timestep that stops before the first predicted crossing."""
        if self.number_of_shells < 2:
            return np.inf
        separation = self.radius[1:] - self.radius[:-1]
        closing_speed = self.velocity[:-1] - self.velocity[1:]
        # Coincident shells are already at the crossing event.  Returning
        # zero here can make the global hydro loop advance with dt=0 forever;
        # ``step`` resolves the crossing by exchanging shell states first.
        tolerance = 32.0 * np.finfo(float).eps * max(
            1.0, float(np.max(np.abs(self.radius)))
        )
        closing = (closing_speed > 0.0) & (separation > tolerance)
        candidates = separation[closing] / closing_speed[closing]
        if candidates.size == 0:
            return np.inf
        return float(safety_factor * np.min(candidates))

    def _resolve_coincident_crossings(self):
        """Exchange states for shells that meet while moving through one another."""
        if self.number_of_shells < 2:
            return
        separation = self.radius[1:] - self.radius[:-1]
        closing_speed = self.velocity[:-1] - self.velocity[1:]
        tolerance = 32.0 * np.finfo(float).eps * max(
            1.0, float(np.max(np.abs(self.radius)))
        )
        pairs = np.flatnonzero((separation <= tolerance) & (closing_speed > 0.0))
        self._exchange_shell_states(pairs)

    def _exchange_shell_states(self, pairs):
        """Exchange state across the supplied neighboring crossing pairs."""
        for index in pairs:
            self.velocity[index:index + 2] = self.velocity[index:index + 2][::-1]
            self.mass[index:index + 2] = self.mass[index:index + 2][::-1]
            self.angular_momentum[index:index + 2] = self.angular_momentum[index:index + 2][::-1]
        if pairs.size:
            self._mass_prefix_cache = None

    def _crossing_event_pairs(self, crossing_dt):
        """Return pairs whose predicted crossing is the current event."""
        separation = self.radius[1:] - self.radius[:-1]
        closing_speed = self.velocity[:-1] - self.velocity[1:]
        tolerance = 32.0 * np.finfo(float).eps * max(
            1.0, float(np.max(np.abs(self.radius)))
        )
        closing = (closing_speed > 0.0) & (separation > tolerance)
        event = closing & (
            separation <= crossing_dt * closing_speed * (1.0 + 1.0e-12)
        )
        return np.flatnonzero(event)

    def _reflect_at_origin(self):
        """Reflect shells that crossed the spherical coordinate origin."""
        crossed = self.radius < 0.0
        if np.any(crossed):
            self.radius[crossed] *= -1.0
            self.velocity[crossed] *= -1.0
        # Keep the radial coordinate consistent with the softened force law.
        # This prevents near-origin shells from creating arbitrarily small
        # crossing substeps after repeated cold collapse.
        radius_floor = max(self.softening, np.finfo(float).tiny)
        self.radius = np.maximum(self.radius, radius_floor)

    def _absorb_into_core(
        self,
        radius,
        velocity,
        scale_factor=1.0,
        gas_enclosed_mass=None,
        background_enclosed_mass=None,
        cosmological=False,
        include_shell_mass_with_fixed=True,
    ):
        """Absorb energetically bound shells that enter the unresolved core."""
        if self.central_core_radius <= 0.0 or self.number_of_shells == 0:
            return 0.0
        radius = np.asarray(radius, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        # A negative radius means that the shell crossed the spherical
        # origin during this drift; it has necessarily traversed the core,
        # even if the finite step overshot beyond ``central_core_radius``.
        entered_core = (radius <= self.central_core_radius) | (
            (radius < 0.0) & (np.abs(radius) <= 4.0 * self.central_core_radius)
        )
        # A velocity sign alone is not a binding criterion.  Use the
        # instantaneous specific total energy in the same softened potential
        # used by the shell integrator.  This includes live DM, the fixed
        # core, gas, angular momentum, and cosmological background subtraction.
        safe_radius = np.maximum(np.abs(radius), np.finfo(float).tiny)
        enclosed = self._enclosed_mass_at_current_positions(safe_radius)
        if self.fixed_enclosed_mass is not None:
            if callable(self.fixed_enclosed_mass):
                fixed = np.asarray(
                    self.fixed_enclosed_mass(safe_radius), dtype=float
                )
            else:
                fixed = np.full_like(safe_radius, self.fixed_enclosed_mass)
            if include_shell_mass_with_fixed:
                enclosed = enclosed + fixed
            else:
                enclosed = fixed
        if gas_enclosed_mass is not None:
            gas_mass = (
                np.asarray(gas_enclosed_mass(safe_radius), dtype=float)
                if callable(gas_enclosed_mass)
                else np.asarray(gas_enclosed_mass, dtype=float)
            )
            enclosed = enclosed + gas_mass
        if cosmological and background_enclosed_mass is not None:
            background = (
                np.asarray(background_enclosed_mass(safe_radius), dtype=float)
                if callable(background_enclosed_mass)
                else np.asarray(background_enclosed_mass, dtype=float)
            )
            enclosed = enclosed - background
        g_code = _gravitational_constant_code(self.CodeUnits)
        potential = -g_code * float(scale_factor) * enclosed / (
            safe_radius + self.softening
        )
        angular = 0.5 * self.angular_momentum**2 / (
            safe_radius + self.softening
        )**2
        total_energy = 0.5 * velocity**2 + angular + potential
        bound = total_energy <= self.core_absorption_energy
        if self.core_absorption_velocity > 0.0:
            bound &= velocity <= -self.core_absorption_velocity
        absorbed = entered_core & bound
        if not np.any(absorbed):
            return 0.0
        absorbed_mass = float(np.sum(self.mass[absorbed]))
        self.central_core_mass += absorbed_mass
        if self.fixed_enclosed_mass is not None and not callable(self.fixed_enclosed_mass):
            self.fixed_enclosed_mass = self.central_core_mass
        keep = ~absorbed
        self.radius = self.radius[keep]
        self.velocity = self.velocity[keep]
        self.mass = self.mass[keep]
        self.angular_momentum = self.angular_momentum[keep]
        self._mass_prefix_cache = None
        return absorbed_mass

    def step(
        self,
        dt,
        crossing_safety_factor=0.1,
        gas_enclosed_mass=None,
        background_enclosed_mass=None,
        scale_factor=1.0,
        scale_factor_end=None,
        cosmological=False,
        include_shell_mass_with_fixed=False,
    ):
        """Advance one kick-drift-kick step, limiting ``dt`` before crossing."""
        dt = float(dt)
        if dt < 0.0:
            raise ValueError("dark-matter timestep must be non-negative")
        if scale_factor_end is None:
            scale_factor_end = scale_factor
        if dt == 0.0:
            return 0.0

        # Crossing control must not silently shorten the DM evolution while
        # the hydro state advances by the requested ``dt``.  The old code
        # advanced only to the first crossing and returned that shorter time;
        # this progressively desynchronised shell and fluid time and erased
        # the collapse of the top-hat.  Resolve crossings with substeps and
        # always cover the complete requested interval.
        remaining = dt
        elapsed = 0.0
        minimum_step = 64.0 * np.finfo(float).eps * max(1.0, dt)
        while remaining > minimum_step:
            self._resolve_coincident_crossings()
            crossing_dt = self.crossing_timestep(safety_factor=1.0)
            substep = remaining
            event_pairs = np.empty(0, dtype=int)
            if crossing_dt < substep:
                # Advance exactly to the first crossing, exchange the
                # neighboring shell states at the event, then continue with
                # the remaining time.  This avoids repeated near-crossing
                # overshoot steps and makes the event treatment independent
                # of the legacy safety-factor parameter.
                event_pairs = self._crossing_event_pairs(crossing_dt)
                substep = crossing_dt
            if substep <= minimum_step:
                substep = min(remaining, max(minimum_step, 1.0e-12 * dt))

            fraction_start = elapsed / dt
            fraction_end = (elapsed + substep) / dt
            a_start = float(scale_factor) + fraction_start * (
                float(scale_factor_end) - float(scale_factor)
            )
            a_end = float(scale_factor) + fraction_end * (
                float(scale_factor_end) - float(scale_factor)
            )
            acceleration = self.acceleration(
                gas_enclosed_mass=gas_enclosed_mass,
                background_enclosed_mass=background_enclosed_mass,
                scale_factor=a_start,
                cosmological=cosmological,
                include_shell_mass_with_fixed=include_shell_mass_with_fixed,
            )
            velocity_half = self.velocity + 0.5 * substep * acceleration
            self.radius = self.radius + substep * velocity_half
            self.velocity = velocity_half
            if event_pairs.size:
                # Roundoff can leave a tiny residual separation after the
                # exact event step.  Place each event pair at the common
                # crossing radius before exchanging states so the next loop
                # iteration cannot generate a zero-progress crossing event.
                for index in event_pairs:
                    crossing_radius = 0.5 * (
                        self.radius[index] + self.radius[index + 1]
                    )
                    self.radius[index:index + 2] = crossing_radius
                self._exchange_shell_states(event_pairs)
            self._absorb_into_core(
                self.radius,
                self.velocity,
                scale_factor=0.5 * (a_start + a_end),
                gas_enclosed_mass=gas_enclosed_mass,
                background_enclosed_mass=background_enclosed_mass,
                cosmological=cosmological,
                include_shell_mass_with_fixed=include_shell_mass_with_fixed,
            )
            self._reflect_at_origin()
            self.sort_by_radius()
            acceleration_new = self.acceleration(
                gas_enclosed_mass=gas_enclosed_mass,
                background_enclosed_mass=background_enclosed_mass,
                scale_factor=a_end,
                cosmological=cosmological,
                include_shell_mass_with_fixed=include_shell_mass_with_fixed,
            )
            self.velocity += 0.5 * substep * acceleration_new
            elapsed += substep
            remaining = dt - elapsed

        return dt

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
