"""Numerical solver subsystem helpers."""

import numpy as np

import radhydropy.gravity as rg
from radhydropy.runtime_fields import runtime_fields


def _gravity_model(solver, par):
    """Return the configured gravity model, if any."""
    gravity = getattr(par, "gravity", None)
    if isinstance(gravity, rg.Gravity):
        return gravity
    if gravity is not None and hasattr(gravity, "acceleration_on_mesh"):
        return gravity
    if not (
        getattr(par, "externalgravity", False)
        or getattr(par, "selfgravity", False)
        or getattr(par, "cosmological_gravity", False)
        or getattr(par, "dark_matter", None) is not None
    ):
        return None
    return rg.Gravity(
        selfgravity=getattr(par, "selfgravity", False),
        externalgravity=getattr(par, "externalgravity", False),
        potential=getattr(par, "gravity_potential", None),
        coordinate=getattr(par, "gravity_coordinate", None),
        acceleration=getattr(par, "gravity_acceleration", None),
        code_units=getattr(par, "CodeUnits", None),
        selfgravity_softening=getattr(par, "selfgravity_softening", 0.0),
        selfgravity_boundary_acceleration=getattr(
            par,
            "selfgravity_boundary_acceleration",
            0.0,
        ),
        dark_matter=getattr(par, "dark_matter", None),
        cosmological=getattr(par, "cosmological_gravity", False),
        cosmology=getattr(par, "cosmology", None),
    )


def _synchronize_gravity_energy_roundoff(solver, mesh, fluid, par, momentum):
    """Remove a sub-ULP-scale ``E < K`` deficit after a gravity update.

    Gravity updates momentum and total energy with mathematically identical
    midpoint work, but evaluating the two expressions separately can leave a
    tiny negative thermal remainder in kinetic-energy-dominated cells.  Only
    correct deficits within the admissibility roundoff allowance; substantive
    deficits remain visible to the normal conserved-state check.
    """
    if not hasattr(fluid, "Energy_code"):
        return 0.0
    mass = np.asarray(fluid.Mass_code, dtype=float)
    energy = np.asarray(fluid.Energy_code, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    kinetic = np.zeros_like(energy)
    np.divide(
        0.5 * momentum**2,
        mass,
        out=kinetic,
        where=mass > 0.0,
    )
    required = kinetic
    if getattr(par, "gas_rotational_energy", False):
        angular = np.asarray(fluid.AngularMomentum_code, dtype=float)
        radius = np.abs(
            np.asarray(
                solver._geometry_state(mesh, par).coordinate_runtime_code,
                dtype=float,
            )
        )
        rotational = np.zeros_like(energy)
        valid = (mass > 0.0) & (radius > 0.0) & np.isfinite(angular)
        rotational[valid] = 0.5 * angular[valid] ** 2 / (mass[valid] * radius[valid] ** 2)
        required = required + rotational
    density_floor = max(
        0.0,
        float(np.asarray(getattr(par, "cfl_density_floor", 0.0))),
    )
    volume_runtime_code = np.asarray(
        solver._geometry_state(mesh, par).volume_runtime_code,
        dtype=float,
    )
    resolved = mass > density_floor * np.maximum(volume_runtime_code, 0.0)
    finite = np.isfinite(mass) & np.isfinite(energy) & np.isfinite(required)
    scale = np.maximum(
        np.maximum(np.abs(energy), required),
        np.finfo(float).tiny,
    )
    relative_tolerance = 1.0e-6 if getattr(par, "dual_energy", False) else 1.0e-7
    # The midpoint work and kinetic-energy expressions have different
    # evaluation order.  Allow a small margin at this source boundary before
    # handing the state to the stricter global admissibility check.
    synchronization_tolerance = 1.05 * relative_tolerance
    deficit = required - energy
    correct = (
        resolved
        & finite
        & (mass > 0.0)
        & (deficit > 0.0)
        & (deficit <= synchronization_tolerance * scale)
    )
    if not np.any(correct):
        solver.last_gravity_roundoff_energy = 0.0
        solver.last_gravity_roundoff_energy_by_cell = None
        return 0.0
    correction = np.zeros_like(energy)
    correction[correct] = deficit[correct]
    fluid.Energy_code[...] = energy + correction
    first = int(par.mesh.ghost_cells)
    last = first + int(par.mesh.grid_cells)
    solver.last_gravity_roundoff_energy = float(np.sum(correction[first:last]))
    solver.last_gravity_roundoff_energy_by_cell = correction[first:last].copy()
    return solver.last_gravity_roundoff_energy


def ApplyGravity(solver, dt, mesh, fluid, par):
    """Apply the combined external and gas self-gravity source update."""
    interior = solver._interior_slice(par)
    gravity = solver._gravity_model(par)
    rotational_support = solver._rotational_energy_enabled(par)
    if rotational_support and getattr(mesh, "coordsys", None) != "spherical":
        raise ValueError("gas_rotational_energy requires a spherical mesh")
    if gravity is None and not rotational_support:
        solver.last_centrifugal_work = 0.0
        solver.last_centrifugal_work_by_cell = None
        solver.last_gravity_work_by_cell = (
            np.zeros(
                int(par.mesh.grid_cells),
                dtype=float,
            )
            if getattr(par, "energy_diagnostics", False)
            else None
        )
        return 0
    if gravity is not None and getattr(gravity, "cosmological", False):
        time_field = runtime_fields(par).time
        gravity.tau_supercomoving_code = getattr(fluid, time_field)
    crossing_safety_factor = getattr(par, "dark_matter_crossing_safety_factor", 0.1)
    if gravity is not None and getattr(gravity, "dark_matter", None) is not None:
        gravity.advance_dark_matter(
            dt,
            mesh,
            getattr(fluid, runtime_fields(par).density),
            par,
            crossing_safety_factor=crossing_safety_factor,
            tau_supercomoving_code=getattr(fluid, runtime_fields(par).time),
        )
    # ApplyGravity follows the conservative hydro flux update and precedes
    # the primitive-state refresh.  Therefore the primitive runtime state can
    # still describe the pre-hydro state, while Mass and Mom already
    # describe the post-hydro state.  Derive both quantities from the
    # current conserved fields so the gravity momentum and work updates
    # use the same state.
    fields = runtime_fields(par)
    volume_runtime_code = np.asarray(
        solver._geometry_state(mesh, par).volume_runtime_code,
        dtype=float,
    )
    mass = np.asarray(fluid.Mass_code, dtype=float)
    momentum = np.asarray(fluid.Mom_code, dtype=float)
    current_rho = np.zeros_like(mass)
    np.divide(
        mass,
        volume_runtime_code,
        out=current_rho,
        where=volume_runtime_code > 0.0,
    )
    current_vel = np.zeros_like(momentum)
    np.divide(momentum, mass, out=current_vel, where=mass > 0.0)
    if gravity is None:
        acceleration = np.zeros_like(current_rho)
    else:
        acceleration = gravity.acceleration_on_mesh(
            mesh,
            rho=current_rho,
            par=par,
        )
    code_units = getattr(par, "CodeUnits", None)
    if code_units is not None:
        target_unit = code_units.length_unit / code_units.time_unit**2
        if hasattr(acceleration, "to_value"):
            acceleration = np.asarray(acceleration.to_value(target_unit), dtype=float)
        else:
            acceleration = np.asarray(acceleration, dtype=float)
    density_field = current_rho
    if np.shape(acceleration) != np.shape(density_field):
        raise ValueError(
            "Gravity acceleration shape %s does not match fluid state shape %s"
            % (np.shape(acceleration), np.shape(density_field)),
        )
    gravity_acceleration = acceleration.copy()
    rotational_acceleration = np.zeros_like(current_rho)
    if rotational_support:
        angular_momentum = np.asarray(fluid.AngularMomentum_code, dtype=float)
        specific = np.zeros_like(current_rho)
        np.divide(
            angular_momentum,
            mass,
            out=specific,
            where=mass > 0.0,
        )
        radius = np.abs(
            np.asarray(
                solver._geometry_state(mesh, par).coordinate_runtime_code,
                dtype=float,
            )
        )
        valid_radius = (radius > 0.0) & np.isfinite(radius) & np.isfinite(specific) & (mass > 0.0)
        rotational_acceleration[valid_radius] = (
            specific[valid_radius] ** 2 / radius[valid_radius] ** 3
        )
        rotational_acceleration[~valid_radius] = 0.0

    # Apply gravity and centrifugal momentum sources sequentially. Gravity
    # work updates the gas energy. Centrifugal acceleration is an internal
    # transfer from rotational to radial kinetic energy, so it must not
    # add energy to the total-energy field.  Limit the local momentum
    # increment when the split source step would otherwise make
    # E_total < E_kin + E_rot.
    dt_value = float(np.asarray(dt, dtype=float))
    gravity_momentum = momentum + mass * gravity_acceleration * dt_value
    gravity_work = 0.5 * (momentum + gravity_momentum) * gravity_acceleration * dt_value
    new_energy = np.asarray(fluid.Energy_code, dtype=float) + gravity_work

    source_increment = mass * rotational_acceleration * dt_value
    source_factors = np.ones_like(source_increment)
    if rotational_support:
        angular = np.asarray(fluid.AngularMomentum_code, dtype=float)
        radius = np.abs(
            np.asarray(
                solver._geometry_state(mesh, par).coordinate_runtime_code,
                dtype=float,
            )
        )
        rotational_energy = np.zeros_like(mass)
        valid_rotational = (
            (mass > 0.0) & (radius > 0.0) & np.isfinite(angular) & np.isfinite(radius)
        )
        rotational_energy[valid_rotational] = (
            0.5
            * angular[valid_rotational] ** 2
            / (mass[valid_rotational] * radius[valid_rotational] ** 2)
        )
        available_radial_energy = new_energy - rotational_energy
        base_admissible = (
            np.isfinite(mass)
            & (mass > 0.0)
            & np.isfinite(gravity_momentum)
            & np.isfinite(available_radial_energy)
            & (0.5 * gravity_momentum**2 / mass <= available_radial_energy)
        )

        def source_admissible(index, factor):
            trial_momentum = gravity_momentum[index] + factor * source_increment[index]
            trial_kinetic = 0.5 * trial_momentum**2 / mass[index] if mass[index] > 0.0 else 0.0
            tolerance = 1.0e-12 * max(
                abs(new_energy[index]),
                abs(rotational_energy[index]),
                np.finfo(float).tiny,
            )
            return trial_kinetic <= available_radial_energy[index] + tolerance

        for index in np.flatnonzero(base_admissible & (source_increment != 0.0)):
            if source_admissible(index, 1.0):
                continue
            low, high = 0.0, 1.0
            for _ in range(48):
                middle = 0.5 * (low + high)
                if source_admissible(index, middle):
                    low = middle
                else:
                    high = middle
            source_factors[index] = low

    new_momentum = gravity_momentum + source_factors * source_increment
    centrifugal_work = 0.5 * (gravity_momentum + new_momentum) * rotational_acceleration * dt_value
    fluid.Mom_code[...] = new_momentum
    fluid.Energy_code[...] = new_energy
    _synchronize_gravity_energy_roundoff(
        solver,
        mesh,
        fluid,
        par,
        new_momentum,
    )
    if gravity is not None and hasattr(fluid, "GravitationalPotentialEnergy_code"):
        # The explicit potential-energy reservoir receives the opposite
        # of the gravity work. Centrifugal work is retained as a diagnostic
        # only because E_rot is already part of total Energy.
        fluid.GravitationalPotentialEnergy_code[...] -= gravity_work
    solver.last_gravity_work = float(
        np.sum(gravity_work[interior]),
    )
    solver.last_gravity_work_by_cell = (
        np.asarray(gravity_work[interior], dtype=float).copy()
        if getattr(par, "energy_diagnostics", False)
        else None
    )
    solver.last_centrifugal_work = float(np.sum(centrifugal_work[interior]))
    solver.last_centrifugal_source_factors = source_factors.copy()
    solver.centrifugal_source_limited_count = int(
        np.count_nonzero(source_factors[interior] < 1.0 - 1.0e-12),
    )
    solver.last_centrifugal_work_by_cell = (
        np.asarray(centrifugal_work[interior], dtype=float).copy()
        if getattr(par, "energy_diagnostics", False)
        else None
    )
    solver.last_dark_matter_substeps = int(
        getattr(getattr(gravity, "dark_matter", None), "last_substep_count", 0),
    )
    return 1
