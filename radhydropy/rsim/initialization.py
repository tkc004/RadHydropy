"""Rsim execution subsystem helpers."""

import time
import numpy as np
import unyt

import radhydropy.io as rio
from radhydropy.units import (
    _CODE_UNIT_GROUPS, apply_code_unit_specs, code_units_from_system,
    code_quantity_to_cgs, time_seconds, quantity_to_value,
)
from radhydropy.eos import EOS

def Callreadhdf5(sim):
    """Read the configured initial-condition HDF5 file."""
    print("--- Read Initial Condition ---")
    print("--- %s seconds ---" % (
        time.time() - getattr(sim, "_start_time", time.time())
    ))
    sim._require_code_units()
    rio.readhdf5(
        sim.par,
        sim.mesh,
        sim.fluid,
        sim.par.simulation.initial_condition_filename,
    )
    sim.energy_diagnostics_enabled = bool(
        getattr(sim.par, "energy_diagnostics", False)
    )
    diagnostic_count = int(
        sim.par.mesh.grid_cells
    )
    sim.cumulative_gravity_work_by_cell = np.zeros(
        diagnostic_count, dtype=float
    )
    sim.cumulative_hydro_energy_change_by_cell = np.zeros(
        diagnostic_count, dtype=float
    )
    sim.cumulative_thermochemistry_energy_change_by_cell = np.zeros(
        diagnostic_count, dtype=float
    )
    sim.cumulative_compression_work_by_cell = np.zeros(
        diagnostic_count, dtype=float
    )
    sim.cumulative_shock_work_by_cell = np.zeros(
        diagnostic_count, dtype=float
    )
    # ``readhdf5`` restores EOS parameters and code units from the file
    # header.  The EOS object was created before that restoration in
    # ``__init__``, so rebuild it to prevent stale gamma/units from being
    # used for pressure, energy, and temperature conversions.
    sim.fluid.eos = EOS(
        sim.par.hydrodynamics.eos_type,
        sim.par.hydrodynamics.gamma,
        sim.par.units.CodeUnits,
    )
    sim.checkparams()
    sim.fluid.SetFluidTime(
        sim.par.simulation.current_time
    )
    print("--- Start Initial Time ---")

def SetMesh(sim):
    """Initialize mesh geometry and ghost cells."""
    print("--- Set up the Mesh ---") 
    print("--- %s seconds ---" % (
        time.time() - getattr(sim, "_start_time", time.time())
    ))
    sim.mesh.SetUpMesh(sim.par)

def SetFluid(sim):
    """Initialize fluid ghost cells and pressure."""
    print("--- Set up the fluid ---") 
    print("--- %s seconds ---" % (
        time.time() - getattr(sim, "_start_time", time.time())
    ))
    sim.fluid.SetUpFluid(sim.par, mesh=sim.mesh)

def SetInitFluid(sim):
    """Apply initial boundaries and populate conserved variables."""
    print("--- Fill up the fluid---") 
    print("--- %s seconds ---" % (
        time.time() - getattr(sim, "_start_time", time.time())
    ))
    sim.ConvertParametersToCodeUnits()
    sim.mesh._par = sim.par
    sim.solver.InitializeHydrostaticCore(sim.mesh, sim.fluid, sim.par)
    sim.solver.SetBoundary(sim.mesh,sim.fluid,sim.par)
    sim.solver.SetConserved(sim.mesh,sim.fluid, verbose=getattr(sim.par, 'verbose', 0))
    if getattr(sim.par, 'radiative_transfer_temporal_scheme', 'instantaneous') != 'c2ray':
        sim.solver.ApplyRadiativeTransfer(sim.mesh, sim.fluid, sim.par)

def ConvertParametersToCodeUnits(sim):
    """Convert only the runtime parameters into the internal unit system."""
    code = sim._require_code_units()
    if getattr(sim, '_runtime_parameters_converted_to_code_units', False):
        return

    units = code_units_from_system(code)
    length_unit = units['length']
    mass_unit = units['mass']
    time_unit = units['time']
    unit_map = {
        **units,
        'length_inv': 1.0 / length_unit,
        'area': length_unit ** 2,
        'volume': length_unit ** 3,
        'number_density': 1.0 / (length_unit ** 3),
        'momentum': mass_unit * units['velocity'],
        'mass_flux': mass_unit / (length_unit ** 2 * time_unit),
        'photon_flux': 1.0 / (length_unit ** 2 * time_unit),
        'photon_rate': 1.0 / time_unit,
        'alpha': length_unit ** 3 / time_unit,
        'acceleration': length_unit / time_unit ** 2,
        'potential': units['velocity'] ** 2,
        'specific_angular_momentum': length_unit * units['velocity'],
    }
    apply_code_unit_specs(sim.par, _CODE_UNIT_GROUPS[-1].specs, unit_map)
    source_rate = getattr(
        sim.par,
        'radiative_transfer_source_photon_rate',
        0.0 / unyt.s,
    )
    if hasattr(source_rate, 'to_value'):
        sim.par._static_source_rate_s = float(source_rate.to_value(1.0 / unyt.s))
    else:
        sim.par._static_source_rate_s = float(
            code_quantity_to_cgs(
                source_rate,
                code,
                'photon_rate_per_s',
            )
        )
    sim._runtime_parameters_converted_to_code_units = True

def _require_code_units(sim):
    """Return the active code-unit system or fail fast during startup."""
    code = getattr(getattr(sim.par, 'units', None), 'CodeUnits', None)
    if code is None:
        raise ValueError("simulation startup requires configured code units")
    return code
