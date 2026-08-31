"""Rsim execution subsystem helpers."""

import numpy as np
import radhydropy.thermo_chemistry as rtc
import radhydropy.diagnostics as diagnostics


def AdvectChemistryScalars(sim, dt, old_mass, mass_flux, fluid=None):
    """Advect passive thermo-chemistry scalars after a hydro flux update."""
    if fluid is None:
        fluid = sim.fluid
    sim.solver.AdvectIonizationFraction(
        dt,
        sim.mesh,
        fluid,
        sim.par,
        old_mass,
        mass_flux,
    )

def UpdateThermochemistryPrimitiveState(sim, update_pressure=True, fluid=None):
    """Refresh temperature, mean molecular weight, and optionally pressure."""
    if fluid is None:
        fluid = sim.fluid
    if not rtc.thermochemistry_enabled(fluid, sim.par):
        return
    if getattr(fluid.eos, 'is_isothermal', False) and not getattr(
        sim.par,
        'hydrogen_thermal_coupling',
        False,
    ):
        return
    if getattr(sim.par, 'hydrogen_update_mu', False):
        fluid.SetHydrogenMu(
            hydrogen_mass_fraction=getattr(
                sim.par,
                'hydrogen_mass_fraction',
                1.0,
            )
        )
    fluid.SetTemperature()
    if update_pressure:
        fluid.SetPressure()

def FinalizeHydroStep(
    sim,
    dt,
    old_mass,
    mass_flux,
    advect_chemistry=True,
    fluid=None,
    temperature_before=None,
    gravity_dt=None,
    apply_gravity=True,
):
    """Complete a hydro step after conserved variables have been advanced."""
    if fluid is None:
        fluid = sim.fluid
    if gravity_dt is None:
        gravity_dt = dt
    if apply_gravity:
        sim.solver.ApplyGravity(gravity_dt, sim.mesh, fluid, sim.par)
        diagnostics.check_conserved_energy_admissibility(
            sim, stage='gravity update'
        )
        diagnostics.check_temperature_jump(
            sim, temperature_before, stage='gravity update'
        )
    if advect_chemistry:
        sim.AdvectChemistryScalars(dt, old_mass, mass_flux, fluid=fluid)
    sim._sync_hydro_state(fluid=fluid)
    sim.solver.ApplyHydrostaticCore(sim.mesh, fluid, sim.par)
    sim.solver.SetConserved(sim.mesh, fluid, verbose=getattr(sim.par, 'verbose', 0))
    diagnostics.check_conserved_energy_admissibility(
        sim, stage='hydro SetConserved synchronization'
    )
    diagnostics.check_temperature_jump(
        sim, temperature_before,
        stage='hydro SetConserved synchronization'
    )

def ApplyThermochemistrySources(sim, dt):
    """Apply radiative transport and thermo-chemistry source updates."""
    transport_result = None
    if getattr(sim.par, 'radiative_transfer_temporal_scheme', 'instantaneous') != 'c2ray':
        transport_result = sim.solver.ApplyRadiativeTransfer(
            sim.mesh,
            sim.fluid,
            sim.par,
        )
    source_result = sim.solver.ApplyThermochemistryFast(
        dt,
        sim.mesh,
        sim.fluid,
        sim.par,
        transport_result=transport_result,
    )
    return source_result

def _synchronize_thermochemistry_internal_energy(sim):
    """Refresh the dual energy from the source-updated conservative state.

    Thermochemistry updates the authoritative total energy directly.  The
    auxiliary internal energy must be refreshed from that same state,
    otherwise a stale dual estimate can later overwrite a legitimate
    cooling/heating change when primitive variables are reconstructed.
    """
    if not (
        sim.solver._dual_energy_enabled(sim.par)
        and hasattr(sim.fluid, 'InternalEnergy')
        and rtc.thermochemistry_enabled(sim.fluid, sim.par)
    ):
        return
    first = int(sim.par.mesh.ghost_cells)
    count = int(sim.par.mesh.grid_cells)
    stop = first + count
    mass = np.asarray(sim.fluid.Mass, dtype=float)
    momentum = np.asarray(sim.fluid.Mom, dtype=float)
    total_energy = np.asarray(sim.fluid.Energy, dtype=float)
    kinetic = np.zeros_like(total_energy)
    np.divide(
        0.5 * momentum**2,
        mass,
        out=kinetic,
        where=mass > 0.0,
    )
    thermal = total_energy - kinetic
    if getattr(sim.par, 'gas_rotational_energy', False):
        rotational = sim.solver._rotational_energy_from_conserved(
            sim.mesh, sim.fluid, sim.par
        )
        thermal = thermal - rotational
    valid = (
        np.isfinite(thermal[first:stop])
        & (thermal[first:stop] > 0.0)
    )
    internal = np.asarray(sim.fluid.InternalEnergy, dtype=float).copy()
    internal_slice = internal[first:stop]
    internal_slice[valid] = thermal[first:stop][valid]
    sim.fluid.InternalEnergy = internal

def _accumulate_gravity_work(sim):
    """Accumulate work from one completed gravity/source substep."""
    gravity_work = float(getattr(sim.solver, "last_gravity_work", 0.0))
    centrifugal_work = float(
        getattr(sim.solver, "last_centrifugal_work", 0.0)
    )
    sim.last_gravity_work += gravity_work
    sim.last_centrifugal_work = getattr(
        sim, "last_centrifugal_work", 0.0
    ) + centrifugal_work
    sim.cumulative_gravity_work += gravity_work
    if sim.energy_diagnostics_enabled:
        work_by_cell = getattr(sim.solver, "last_gravity_work_by_cell", None)
        if work_by_cell is not None:
            sim.cumulative_gravity_work_by_cell += np.asarray(
                work_by_cell, dtype=float
            )
