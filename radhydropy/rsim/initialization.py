"""Rsim execution subsystem helpers."""

import time
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
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
        'time_inv': 1.0 / time_unit,
        'area': length_unit ** 2,
        'volume': length_unit ** 3,
        'number_density': 1.0 / (length_unit ** 3),
        'momentum': mass_unit * units['velocity'],
        'mass_flux': mass_unit / (length_unit ** 2 * time_unit),
        'photon_flux': 1.0 / (length_unit ** 2 * time_unit),
        'photon_rate': 1.0 / time_unit,
        'luminosity': units['energy'] / time_unit,
        'alpha': length_unit ** 3 / time_unit,
        'acceleration': length_unit / time_unit ** 2,
        'potential': units['velocity'] ** 2,
        'specific_angular_momentum': length_unit * units['velocity'],
    }
    apply_code_unit_specs(sim.par, _CODE_UNIT_GROUPS[-1].specs, unit_map)
    nested_specs = {
        "simulation": (
            ("final_time", "time"),
            ("current_time", "time"),
            ("initial_time", "time"),
            ("box_size", "length"),
        ),
        "hydrodynamics": (
            ("hydro_temperature_floor", "temperature"),
        ),
        "output": (("cadence", "time"), ("time_interval", "time")),
        "timestep": (
            ("dtmin", "time"),
            ("dtmax", "time"),
            ("hydrogen_source_dtmin", "time"),
            ("chemistry_timestep", "time"),
            ("evolution_timestep", "time"),
            ("output_interval", "time"),
            ("supercomoving_timestep", "time"),
        ),
        "mesh": (("area", "area"),),
        "boundary": (
            ("inflow_velocity", "velocity"),
            ("inflow_density", "density"),
            ("inflow_temperature", "temperature"),
            ("outflow_velocity", "velocity"),
            ("outflow_density", "density"),
            ("outflow_temperature", "temperature"),
        ),
        "radiation": (
            ("boundary_flux", "photon_flux"),
            ("source_photon_rate", "photon_rate"),
            ("cmb_temperature_0", "temperature"),
            ("hydrogen_ngamma_initial", "number_density"),
            ("hydrogen_ngamma_inflow", "number_density"),
            ("hydrogen_ngamma_outflow", "number_density"),
            ("hydrogen_sigma_gamma", "area"),
            ("hydrogen_epsilon_gamma", "energy"),
            ("radiation_pressure_source_luminosity", "luminosity"),
        ),
        "chemistry": (
            ("hydrogen_alpha_B", "alpha"),
            ("hydrogen_beta", "alpha"),
            ("hydrogen_photon_energy", "energy"),
        ),
        "thermochemistry": (
            ("cooling_temperature_floor", "temperature"),
            ("hydrogen_implicit_absolute_temperature_tolerance", "temperature"),
        ),
    }
    for group_name, specs in nested_specs.items():
        group = getattr(sim.par, group_name, None)
        if group is not None:
            apply_code_unit_specs(group, specs, unit_map)
    # The structured parameter groups are created before this conversion.
    # Refresh the groups that contain values consumed directly by the runtime;
    # otherwise, for example, ``simulation.final_time`` can retain the
    # unitful YAML value while the flat ``timesim`` attribute is code-valued.
    initialize_groups = getattr(sim.par, "_initialize_parameter_groups", None)
    if initialize_groups is not None:
        initialize_groups()
        sync_simulation = getattr(
            sim.par, "_sync_simulation_parameters", None
        )
        if sync_simulation is not None:
            sync_simulation()
        configure_cosmology = getattr(sim.par, "_configure_cosmology", None)
        if configure_cosmology is not None and getattr(
            sim.par, "cosmological_expansion", False
        ):
            configure_cosmology()
    else:
        for sync_name in (
            "_sync_simulation_parameters",
            "_sync_output_parameters",
            "_sync_timestep_parameters",
        ):
            sync = getattr(sim.par, sync_name, None)
            if sync is not None:
                sync()
    # Preserve compatibility with lightweight parameter namespaces used by
    # component-level callers, which do not provide Par's sync methods.
    simulation = getattr(sim.par, "simulation", None)
    if simulation is not None and not hasattr(sim.par, "_sync_simulation_parameters"):
        for nested_name, flat_name in (
            ("final_time", "timesim"),
            ("current_time", "time"),
            ("box_size", "boxsize"),
        ):
            if hasattr(simulation, nested_name) and hasattr(sim.par, flat_name):
                setattr(simulation, nested_name, getattr(sim.par, flat_name))
    _require_unitless_runtime_parameters(sim)
    source_rate = getattr(sim.par, 'radiative_transfer_source_photon_rate', None)
    if source_rate is None and hasattr(sim.par, '_parameter'):
        source_rate = sim.par._parameter(
            'radiative_transfer_source_photon_rate', 0.0 / unyt.s
        )
    if source_rate is None:
        source_rate = 0.0 / unyt.s
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


def _require_unitless_runtime_parameters(sim):
    """Fail fast if any unitful value leaked into runtime parameters.

    ``runparams`` is deliberately excluded because it preserves the original
    unit-aware configuration for provenance and output.  ``units`` is also
    excluded because it contains the code-unit definitions themselves.
    """
    leaked = []
    visited = set()

    def visit(value, path):
        if value is None or id(value) in visited:
            return
        if hasattr(value, "to_value"):
            leaked.append(path)
            return
        if isinstance(value, (str, bytes, int, float, bool, np.number)):
            return
        visited.add(id(value))
        if isinstance(value, Mapping):
            for name, child in value.items():
                if name in {"CodeUnits", "unit_system"}:
                    continue
                visit(child, f"{path}.{name}")
        elif is_dataclass(value):
            for field in fields(value):
                if field.name in {"CodeUnits", "unit_system", "model"}:
                    continue
                visit(getattr(value, field.name), f"{path}.{field.name}")
        elif isinstance(value, (list, tuple, set)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, np.ndarray) and value.dtype == object:
            for index, child in np.ndenumerate(value):
                visit(child, f"{path}{index}")
        elif hasattr(value, "__dict__"):
            for name, child in vars(value).items():
                if name.startswith("_") or name in {
                    "runparams", "units", "unit_system", "CodeUnits", "model",
                }:
                    continue
                visit(child, f"{path}.{name}")

    for name, value in vars(sim.par).items():
        if name in {"runparams", "units", "unit_system", "CodeUnits"}:
            continue
        visit(value, f"par.{name}")

    if leaked:
        names = ", ".join(leaked[:20])
        if len(leaked) > 20:
            names += f", ... ({len(leaked)} total)"
        raise TypeError(
            "runtime parameters must be unitless code values after startup "
            f"conversion; unitful value(s) found in: {names}"
        )

def _require_code_units(sim):
    """Return the active code-unit system or fail fast during startup."""
    code = getattr(getattr(sim.par, 'units', None), 'CodeUnits', None)
    if code is None:
        raise ValueError("simulation startup requires configured code units")
    return code
