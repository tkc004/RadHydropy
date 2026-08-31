"""Rsim execution subsystem helpers."""

import numpy as np
import radhydropy.thermo_chemistry as rtc
import radhydropy.diagnostics as diagnostics


def GetStepTime(sim, dt=None, final_time=None):
    """Return a timestep, clipped to ``final_time`` when supplied."""
    if dt is None:
        dt = sim.solver.GetTimeStep(sim.mesh, sim.fluid, sim.par)
    gravity = getattr(sim.par, 'gravity', None)
    dark_matter = getattr(gravity, 'dark_matter', None)
    if (
        dark_matter is not None
        and getattr(sim.par, "dark_matter_global_timestep_limit", True)
    ):
        dm_dt = dark_matter.crossing_timestep(safety_factor=1.0)
        if np.isfinite(dm_dt):
            if dm_dt <= 0.0:
                raise RuntimeError(
                    "dark-matter crossing timestep is non-positive; "
                    "coincident shell crossings must be resolved before "
                    "advancing the simulation"
                )
            dt = min(dt, dm_dt)
    current_time = sim.fluid.time
    if final_time is not None:
        if hasattr(final_time, "units"):
            target_units = final_time.units
            if not hasattr(current_time, "to_value"):
                current_time = current_time * target_units
            if not hasattr(dt, "to_value"):
                dt = dt * target_units
        elif hasattr(current_time, "units") and not hasattr(dt, "to_value"):
            dt = dt * current_time.units
    if final_time is not None and current_time + dt > final_time:
        dt = final_time - current_time
    return dt

def PrepareConservedStep(sim, fluid=None):
    """Apply boundaries and refresh conserved variables before a step."""
    if fluid is None:
        fluid = sim.fluid
    sim.solver.SetBoundary(sim.mesh, fluid, sim.par)
    sim.solver.SetConserved(sim.mesh, fluid, verbose=getattr(sim.par, 'verbose', 0))
    diagnostics.check_conserved_energy_admissibility(
        sim, stage='pre-hydro SetConserved synchronization'
    )

def AdvanceHydroFluxes(sim, dt, fluid=None):
    """Advance the Euler flux update and return mass data for scalar advection."""
    if fluid is None:
        fluid = sim.fluid
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    rho = np.asarray(fluid.rho[first:last], dtype=float)
    old_energy = np.asarray(fluid.Energy[first:last], dtype=float).copy()
    pressure = np.asarray(fluid.pre[first:last], dtype=float)
    velocity = np.asarray(fluid.vel[first:last], dtype=float)
    coordinate = np.asarray(sim.mesh.coordinate[first:last], dtype=float)
    if getattr(sim.mesh, "coordsys", None) == "spherical":
        radius = np.maximum(np.abs(coordinate), np.finfo(float).tiny)
        divergence = np.gradient(radius**2 * velocity, coordinate) / radius**2
    else:
        divergence = np.gradient(velocity, coordinate)
    sim.last_compression_work_by_cell = (
        -pressure * divergence * np.asarray(sim.mesh.vol[first:last], dtype=float)
        * float(np.asarray(dt, dtype=float))
        if sim.energy_diagnostics_enabled else None
    )
    old_mass = fluid.Mass.copy()
    gravity = getattr(sim.par, 'gravity', None)
    potential_cell = None
    potential_face = None
    if (
        gravity is not None
        and hasattr(gravity, 'potential_on')
        and (
            not hasattr(gravity, 'potential')
            or getattr(gravity, 'potential') is not None
        )
    ):
        potential_cell = np.asarray(
            gravity.potential_on(sim.mesh.coordinate), dtype=float
        )
        potential_face = np.asarray(
            gravity.potential_on(sim.mesh.boundary[:-1]), dtype=float
        )
    sim.solver.SetInterFaceFlux(
        sim.mesh,
        fluid,
        sim.par.boundary.condition,
        method=sim.par.hydrodynamics.riemann_solver,
        verbose=getattr(sim.par, 'verbose', 0),
        order=sim.par.hydrodynamics.order,
    )
    mass_flux = fluid.Mass.flux.copy()
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    area = np.asarray(sim.mesh.area, dtype=float)
    energy_flux = np.asarray(fluid.Energy.flux, dtype=float)
    if potential_face is not None:
        mass_flux_area = np.asarray(mass_flux, dtype=float) * area
        potential_flux_area = potential_face * mass_flux_area
        sim.last_hydro_potential_flux = float(
            dt * (potential_flux_area[first] - potential_flux_area[last])
        )
    else:
        sim.last_hydro_potential_flux = 0.0
    # AddFluxes uses inner-face minus outer-face fluxes.  This is the
    # net energy entering the physical domain through its boundaries.
    sim.last_hydro_boundary_energy_flux = float(
            dt * (
                energy_flux[first] * area[first]
            - energy_flux[last] * area[last]
            )
    )
    sim.cumulative_hydro_boundary_energy += sim.last_hydro_boundary_energy_flux
    diagnostics.check_conserved_energy_admissibility(
        sim, stage='hydro face reconstruction'
    )
    sim.solver.AddFluxes(
        dt,
        sim.mesh,
        fluid,
        sim.par.boundary.condition,
    )
    if potential_cell is not None:
        old_mass_active = np.asarray(old_mass[first:last], dtype=float)
        new_mass_active = np.asarray(fluid.Mass[first:last], dtype=float)
        phi_active = potential_cell[first:last]
        sim.last_hydro_potential_change = float(
            np.sum((new_mass_active - old_mass_active) * phi_active)
        )
        sim.cumulative_gravity_potential_change += (
            sim.last_hydro_potential_change
        )
        sim.cumulative_gravity_potential_flux += sim.last_hydro_potential_flux
    else:
        sim.last_hydro_potential_change = 0.0
    if sim.energy_diagnostics_enabled:
        sim.last_hydro_energy_change_by_cell = (
            np.asarray(fluid.Energy[first:last], dtype=float) - old_energy
        )
        sim.cumulative_hydro_energy_change_by_cell += (
            sim.last_hydro_energy_change_by_cell
        )
    return old_mass, mass_flux

def _sync_hydro_state(sim, fluid=None):
    """Refresh primitive and conserved variables after a hydro update."""
    if fluid is None:
        fluid = sim.fluid
    sim.solver.SetPrimitive(
        sim.mesh,
        fluid,
        par=sim.par,
        verbose=getattr(sim.par, 'verbose', 0),
    )
    if rtc.thermochemistry_enabled(fluid, sim.par):
        sim.UpdateThermochemistryPrimitiveState(update_pressure=True, fluid=fluid)
    elif getattr(fluid.eos, 'is_polytropic', False):
        # Hydro-only adiabatic runs still need their primitive temperature
        # refreshed from the conserved internal energy.  Previously this
        # was done only by the thermochemistry path, leaving ``fluid.temp``
        # at its initial value and making adiabatic temperature plots lie.
        fluid.SetTemperature()
    sim.solver.SetConserved(sim.mesh, fluid, verbose=getattr(sim.par, 'verbose', 0))

def _hydro_step_once(
    sim, dt, fluid=None, advect_chemistry=True, apply_gravity=True
):
    """Advance one explicit hydro step on the supplied fluid state."""
    if fluid is None:
        fluid = sim.fluid
    sim.PrepareConservedStep(fluid=fluid)
    old_mass, mass_flux = sim.AdvanceHydroFluxes(dt, fluid=fluid)
    sim.FinalizeHydroStep(
        dt,
        old_mass,
        mass_flux,
        advect_chemistry=advect_chemistry,
        fluid=fluid,
        apply_gravity=apply_gravity,
    )
    return {
        "dt": dt,
        "hydro_steps": 1,
        "source_steps": 0,
    }

def _hydro_step_ssprk2(
    sim, dt, advect_chemistry=True, apply_gravity=True
):
    """Advance hydro variables with the SSPRK2 strong-stability-preserving scheme."""
    initial_state = sim._clone_fluid()
    stage1 = sim._clone_fluid()
    sim._hydro_step_once(
        dt,
        fluid=stage1,
        advect_chemistry=advect_chemistry,
        apply_gravity=apply_gravity,
    )
    stage2 = sim._clone_fluid(stage1)
    sim._hydro_step_once(
        dt,
        fluid=stage2,
        advect_chemistry=advect_chemistry,
        apply_gravity=apply_gravity,
    )

    conserved_fields = ["Mass", "Mom", "Energy"]
    if hasattr(initial_state, "AngularMomentum") and hasattr(stage2, "AngularMomentum"):
        conserved_fields.append("AngularMomentum")
    if (
        hasattr(initial_state, "GravitationalPotentialEnergy")
        and hasattr(stage2, "GravitationalPotentialEnergy")
    ):
        conserved_fields.append("GravitationalPotentialEnergy")
    if hasattr(initial_state, "InternalEnergy") and hasattr(stage2, "InternalEnergy"):
        conserved_fields.append("InternalEnergy")
    for attr in conserved_fields:
        setattr(
            sim.fluid,
            attr,
            0.5 * getattr(initial_state, attr) + 0.5 * getattr(stage2, attr),
        )
    if advect_chemistry and hasattr(initial_state, "xHI") and hasattr(stage2, "xHI"):
        sim.fluid.xHI = 0.5 * initial_state.xHI + 0.5 * stage2.xHI

    sim.fluid.time = initial_state.time + dt
    sim._sync_hydro_state()
    return {
        "dt": dt,
        "hydro_steps": 1,
        "source_steps": 0,
    }

def Step(
    sim,
    dt=None,
    mode="hydro_sources",
    advect_chemistry=True,
    hydro_integrator="euler",
):
    """Advance one canonical simulation step in the requested mode."""
    valid_modes = ("hydro", "hydro_sources", "sources")
    if mode not in valid_modes:
        raise ValueError(
            "Unknown step mode %r; valid modes are %s"
            % (mode, ", ".join(valid_modes))
        )
    valid_hydro_integrators = ("euler", "ssprk2")
    if hydro_integrator not in valid_hydro_integrators:
        raise ValueError(
            "Unknown hydro integrator %r; valid options are %s"
            % (hydro_integrator, ", ".join(valid_hydro_integrators))
        )
    source_integrator = str(getattr(sim.par, 'source_integrator', 'lie')).lower()
    if source_integrator not in ('lie', 'strang'):
        raise ValueError(
            "Unknown source integrator %r; valid options are lie, strang"
            % source_integrator
        )
    if source_integrator == 'strang' and mode != 'hydro':
        raise ValueError("source_integrator='strang' requires mode='hydro'")
    dt = sim.GetStepTime(dt=dt)
    temperature_before = diagnostics.temperature_physical_K(sim)
    if temperature_before is not None:
        temperature_before = temperature_before.copy()
    sim.last_hydro_boundary_energy_flux = 0.0
    sim.last_hydro_potential_change = 0.0
    sim.last_hydro_potential_flux = 0.0
    sim.last_gravity_work = 0.0
    sim.last_gravity_work_by_cell = None
    sim.last_centrifugal_work = 0.0
    sim.last_centrifugal_work_by_cell = None
    sim.last_compression_work_by_cell = None
    sim.last_shock_work_by_cell = None
    sim.last_thermochemistry_energy_change = 0.0
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    if mode != 'sources':
        mass_before = np.asarray(sim.fluid.Mass[first:last], dtype=float)
        momentum_before = np.asarray(sim.fluid.Mom[first:last], dtype=float)
        energy_before = np.asarray(sim.fluid.Energy[first:last], dtype=float)
        kinetic_before = np.zeros_like(mass_before)
        np.divide(
            0.5 * momentum_before**2,
            mass_before,
            out=kinetic_before,
            where=mass_before > 0.0,
        )
        if sim.energy_diagnostics_enabled:
            sim._thermal_energy_before_hydro = energy_before - kinetic_before
    result = {
        "dt": dt,
        "hydro_steps": 0,
        "source_steps": 0,
    }
    source_enabled = bool(
        getattr(sim.par, 'gravity', None) is not None
        or getattr(sim.par, 'externalgravity', False)
        or getattr(sim.par, 'simgravity', False)
        or getattr(sim.par, 'cosmological_gravity', False)
        or getattr(sim.par, 'dark_matter', None) is not None
        or getattr(sim.par, 'gas_rotational_energy', False)
    )
    if source_integrator == 'strang':
        sim.solver.ApplyGravity(0.5 * dt, sim.mesh, sim.fluid, sim.par)
        sim._accumulate_gravity_work()
        sim._sync_hydro_state()

    if mode in ("hydro", "hydro_sources"):
        if hydro_integrator == "ssprk2":
            result = sim._hydro_step_ssprk2(
                dt,
                advect_chemistry=advect_chemistry,
                apply_gravity=False,
            )
            # SSPRK2 integrates only the hydro operator.  Apply the
            # source operator outside the RK stages so Strang has the
            # ordering S(dt/2) -> H(dt) -> S(dt/2).  With Lie splitting,
            # this is the corresponding H(dt) -> S(dt) ordering.
            if source_enabled:
                source_dt = 0.5 * dt if source_integrator == 'strang' else dt
                sim.solver.ApplyGravity(
                    source_dt, sim.mesh, sim.fluid, sim.par
                )
                sim._accumulate_gravity_work()
                diagnostics.check_conserved_energy_admissibility(
                    sim, stage='gravity update'
                )
                diagnostics.check_temperature_jump(
                    sim, temperature_before, stage='gravity update'
                )
                sim._sync_hydro_state()
        else:
            sim.PrepareConservedStep()
            old_mass, mass_flux = sim.AdvanceHydroFluxes(dt)
            diagnostics.check_conserved_energy_admissibility(
                sim, stage='hydro flux update'
            )
            diagnostics.check_temperature_jump(
                sim, temperature_before, stage='hydro flux update'
            )
            sim.FinalizeHydroStep(
                dt,
                old_mass,
                mass_flux,
                advect_chemistry=advect_chemistry,
                temperature_before=temperature_before,
                gravity_dt=(0.5 * dt if source_integrator == 'strang' else dt),
            )
            sim._accumulate_gravity_work()
            first = int(sim.par.mesh.ghost_cells)
            last = first + int(sim.par.mesh.grid_cells)
            mass = np.asarray(sim.fluid.Mass[first:last], dtype=float)
            momentum = np.asarray(sim.fluid.Mom[first:last], dtype=float)
            energy = np.asarray(sim.fluid.Energy[first:last], dtype=float)
            kinetic = np.zeros_like(mass)
            np.divide(
                0.5 * momentum**2,
                mass,
                out=kinetic,
                where=mass > 0.0,
            )
            thermal_after_hydro = energy - kinetic
            thermal_before_hydro = getattr(
                sim, "_thermal_energy_before_hydro", thermal_after_hydro
            )
            if sim.energy_diagnostics_enabled:
                sim.last_shock_work_by_cell = (
                    thermal_after_hydro - thermal_before_hydro
                    - np.asarray(sim.last_compression_work_by_cell, dtype=float)
                )
                sim.cumulative_compression_work_by_cell += (
                    sim.last_compression_work_by_cell
                )
                sim.cumulative_shock_work_by_cell += sim.last_shock_work_by_cell
            sim.last_dark_matter_substeps = int(
                getattr(sim.solver, "last_dark_matter_substeps", 0)
            )
            sim.cumulative_dark_matter_substeps += (
                sim.last_dark_matter_substeps
            )
            sim.dark_matter_substep_history.append(
                sim.last_dark_matter_substeps
            )
            result["dark_matter_substeps"] = sim.last_dark_matter_substeps
            result["hydro_steps"] = 1
            # ``solver.AddFluxes`` advances the fluid clock for the
            # Euler update.  Do not advance it again here; source-only
            # steps below are the cases that need an explicit clock
            # update.
            diagnostics.check_temperature_jump(sim, temperature_before, stage='hydro')

    if mode == "hydro" and hydro_integrator == "ssprk2":
        diagnostics.check_conserved_energy_admissibility(
            sim, stage='hydro flux update'
        )
        diagnostics.check_temperature_jump(sim, temperature_before, stage='hydro')

    if mode == "hydro_sources":
        energy_before_sources_by_cell = np.asarray(
            sim.fluid.Energy[first:last], dtype=float
        ).copy()
        energy_before_sources = float(np.sum(energy_before_sources_by_cell))

    if mode in ("hydro_sources", "sources"):
        source_result = sim.ApplyThermochemistrySources(
            dt,
        )
        diagnostics.check_conserved_energy_admissibility(
            sim, stage='thermochemistry update'
        )
        sim.last_source_result = source_result
        sim.last_source_dt = dt
        sim._synchronize_thermochemistry_internal_energy()
        # Source updates can change temperature, pressure, and chemistry
        # fields, so refresh the boundary state before the next loop.
        if mode == "sources":
            sim.fluid.time += dt
        sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
        sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=getattr(sim.par, 'verbose', 0))
        diagnostics.check_conserved_energy_admissibility(
            sim, stage='thermochemistry SetConserved synchronization'
        )
        pressure_applied = sim.solver.ApplyRadiationPressure(
            dt,
            sim.mesh,
            sim.fluid,
            sim.par,
            source_result,
        )
        if pressure_applied:
            sim._sync_hydro_state()
        if mode == "hydro_sources":
            energy_after_sources = float(
                np.sum(np.asarray(sim.fluid.Energy[first:last], dtype=float))
            )
            sim.last_thermochemistry_energy_change = (
                energy_after_sources - energy_before_sources
            )
            if sim.energy_diagnostics_enabled:
                sim.last_thermochemistry_energy_change_by_cell = (
                    np.asarray(sim.fluid.Energy[first:last], dtype=float)
                    - energy_before_sources_by_cell
                )
                sim.cumulative_thermochemistry_energy_change_by_cell += (
                    sim.last_thermochemistry_energy_change_by_cell
                )
        diagnostics.check_temperature_jump(
            sim,
            temperature_before,
            stage='thermochemistry',
            source_result=source_result,
        )
        result["source_steps"] = int(source_result.get("source_steps", 0))

    result.update({
        "dual_energy_pressure_fallback_count": int(
            getattr(sim.solver, "dual_energy_pressure_fallback_count", 0)
        ),
        "dual_energy_synchronization_count": int(
            getattr(sim.solver, "dual_energy_synchronization_count", 0)
        ),
        "dual_energy_floor_count": int(
            getattr(sim.solver, "dual_energy_floor_count", 0)
        ),
        "dual_energy_floor_injected_energy": float(
            getattr(sim.solver, "dual_energy_floor_injected_energy", 0.0)
        ),
        "dual_energy_entropy_limiter_count": int(
            getattr(sim.solver, "dual_energy_entropy_limiter_count", 0)
        ),
        "gravity_potential_flux": float(
            getattr(sim, "last_hydro_potential_flux", 0.0)
        ),
        "gravity_potential_change": float(
            getattr(sim, "last_hydro_potential_change", 0.0)
        ),
    })
    return result
