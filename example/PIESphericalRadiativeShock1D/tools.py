"""Initial conditions and diagnostics for a gravity-free spherical PIE shock."""

import numpy as np
import unyt
from radhydropy.arrays import as_named_array
import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import CodeUnits, quantity_to_value
from example.basic_hydro_utils import finalize_initial_condition


PROTON_MASS_G = unyt.mp.to_value(unyt.g)
SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
KPC_CM = (1.0 * unyt.kpc).to_value(unyt.cm)


def build_initial_condition(config):
    initial = config['initial_condition']
    par = config['par']
    code_units = config.get('_code_units')
    if code_units is None:
        code_units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    elif 'units' not in par:
        # A component-level caller may attach CodeUnits at the call site when
        # the minimal YAML-shaped parameter mapping has no units group yet.
        par['units'] = {'CodeUnits': code_units.to_dict()}
    grid_cells = int(par['mesh']['grid_cells'])
    result = Rsim(config["par"])
    result.par.simulation.time_proper_code = quantity_to_value(initial['time_proper'], code_units.time_unit)
    result.par.simulation.box_size_proper_code = quantity_to_value(initial['box_size_proper'], code_units.length_unit)
    result.par.simulation.coordinate_system = 'spherical'
    result.par.mesh.grid_cells = grid_cells
    boundary_proper_code = as_named_array(quantity_to_value(
        np.linspace(initial['radius_inner_proper'], initial['radius_outer_proper'], grid_cells + 1), code_units.length_unit
    ))
    result.mesh.boundary_proper_code = boundary_proper_code
    width = np.diff(boundary_proper_code)
    x_proper_code = 0.75 * (boundary_proper_code[1:] ** 4 - boundary_proper_code[:-1] ** 4) / (boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3)
    volume_proper_code = 4.0 * np.pi / 3.0 * (boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3)
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=x_proper_code, boundary_proper_code=boundary_proper_code,
        width_proper_code=width, area_proper_code=4.0 * np.pi * boundary_proper_code[:-1] ** 2, volume_proper_code=volume_proper_code,
    )
    rho_proper_unyt = (
        initial['hydrogen_number_density'] * unyt.mp
        / float(par['thermochemistry']['hydrogen_mass_fraction'])
    )
    rho_proper_code = quantity_to_value(
        np.ones(grid_cells) * rho_proper_unyt, code_units.density_unit
    )
    result.fluid.rho_proper_code = as_named_array(rho_proper_code)
    midpoint_proper_code = quantity_to_value(
        0.5 * (initial['radius_inner_proper'] + initial['radius_outer_proper']), code_units.length_unit
    )
    outflow_velocity_proper_code = quantity_to_value(
        initial['vel_outflow_proper'], code_units.velocity_unit
    )
    inflow_velocity_proper_code = quantity_to_value(
        initial['vel_inflow_proper'], code_units.velocity_unit
    )
    result.fluid.vel_proper_code = as_named_array(np.where(
        x_proper_code < midpoint_proper_code,
        outflow_velocity_proper_code,
        inflow_velocity_proper_code,
    ))
    result.fluid.temp_proper_code = as_named_array(quantity_to_value(np.ones(grid_cells) * initial['temperature_inflow_proper'], code_units.temperature_unit))
    result.fluid.mu = np.ones(grid_cells) * initial['mean_molecular_weight']
    result.fluid.time_proper_code = 0.0
    result.SetMesh()
    result.fluid.SetUpFluid(result.par, result.mesh)
    result.fluid.SetFluidTime(0.0)
    result.fluid.SetEnergyDensity()
    result.fluid._refresh_runtime_state()
    result.mesh._par = result.par
    result.solver.SetConserved(result.mesh, result.fluid, verbose=0)
    finalize_initial_condition(result, grid_cells)
    result.ConvertParametersToCodeUnits()
    return result


def load_output_state(filename, config):
    """Load one snapshot through the configured canonical runtime state."""
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    snapshot = Rsim(config['par'])
    rio.readhdf5(snapshot.par, snapshot.mesh, snapshot.fluid, str(filename))
    first = int(snapshot.par.mesh.ghost_cells)
    count = int(snapshot.par.mesh.grid_cells)
    physical = slice(first, first + count)
    boundary_proper_code = np.asarray(snapshot.mesh.boundary_proper_code)[
        first:first + count + 1
    ]
    return {
        'time_proper_Myr': (
            float(np.asarray(snapshot.fluid.time_proper_code).reshape(-1)[0])
            * float(code_units.time_unit.to_value('s')) / SECONDS_PER_MYR
        ),
        'boundary_proper_cgs_cm': boundary_proper_code * float(
            code_units.length_unit.to_value('cm')
        ),
        'rho_proper_cgs_g_cm3': np.asarray(snapshot.fluid.rho_proper_code)[physical] * float(
            code_units.density_unit.to_value('g/cm**3')
        ),
        'vel_peculiar_proper_cgs_cm_s': np.asarray(snapshot.fluid.vel_proper_code)[physical] * float(
            code_units.velocity_unit.to_value('cm/s')
        ),
        'temperature_proper_cgs_K': np.asarray(snapshot.fluid.temp_proper_code)[physical] * float(
            code_units.temperature_unit.to_value('K')
        ),
    }


def shock_radius(snapshot):
    """Locate the strongest compression near the colliding-stream interface."""
    boundary_proper_cgs_cm = snapshot['boundary_proper_cgs_cm']
    centers = 0.5 * (boundary_proper_cgs_cm[1:] + boundary_proper_cgs_cm[:-1])
    density_proper_cgs_g_cm3 = np.maximum(snapshot['rho_proper_cgs_g_cm3'], 1.0e-99)
    gradient = np.abs(np.diff(np.log(density_proper_cgs_g_cm3)))
    start = max(2, int(0.2 * len(gradient)))
    stop = min(len(gradient) - 1, int(0.9 * len(gradient)))
    index = start + int(np.argmax(gradient[start:stop]))
    return float(centers[index] / KPC_CM)


def shock_history(filenames, config, output_interval_myr=None):
    rows = []
    for filename in filenames:
        snapshot = load_output_state(filename, config)
        if output_interval_myr is None:
            time_myr = snapshot['time_proper_Myr']
        else:
            # Current HDF5 output headers do not preserve the evolving time
            # for this non-cosmological run.  The numbered output and the
            # configured fixed cadence are unambiguous.
            try:
                output_index = int(filename.stem.rsplit('_', 1)[1])
            except (AttributeError, IndexError, ValueError):
                output_index = len(rows)
            time_myr = output_index * float(output_interval_myr)
        rows.append((time_myr, shock_radius(snapshot)))
    return np.asarray(rows, dtype=float)


def estimate_cooling_length(snapshot, table, metallicity, hydrogen_mass_fraction, mu):
    """Estimate post-shock cooling length and cooling time from one snapshot."""
    radius_proper_cgs_cm = snapshot['boundary_proper_cgs_cm']
    centers = 0.5 * (radius_proper_cgs_cm[1:] + radius_proper_cgs_cm[:-1])
    shock_kpc = shock_radius(snapshot)
    shock_index = int(np.argmin(np.abs(centers / KPC_CM - shock_kpc)))
    left = slice(max(0, shock_index - 4), shock_index)
    right = slice(shock_index + 1, min(len(centers), shock_index + 5))
    left_temperature = float(np.median(snapshot['temperature_proper_cgs_K'][left]))
    right_temperature = float(np.median(snapshot['temperature_proper_cgs_K'][right]))
    post_slice = right if right_temperature >= left_temperature else left
    density_proper_cgs_g_cm3 = float(np.median(snapshot['rho_proper_cgs_g_cm3'][post_slice]))
    temperature_proper_cgs_K = float(np.median(snapshot['temperature_proper_cgs_K'][post_slice]))
    vel_peculiar_proper_cgs_cm_s = float(np.median(np.abs(snapshot['vel_peculiar_proper_cgs_cm_s'][post_slice])))
    n_h = hydrogen_mass_fraction * density_proper_cgs_g_cm3 / PROTON_MASS_G
    heating, cooling = table.rates(
        temperature_proper_cgs_K, n_h, metallicity=metallicity, redshift=0.0
    )
    net_rate = max(float(np.asarray(cooling) - np.asarray(heating)), 1.0e-99)
    thermal_energy = 1.5 * density_proper_cgs_g_cm3 * 1.380649e-16 * temperature_proper_cgs_K / (
        mu * PROTON_MASS_G
    )
    cooling_time_s = thermal_energy / net_rate
    cooling_length_cgs_cm = abs(vel_peculiar_proper_cgs_cm_s) * cooling_time_s
    cell_width_cgs_cm = float(np.median(np.diff(radius_proper_cgs_cm)))
    return {
        'shock_radius_proper_kpc': shock_kpc,
        'post_density_cgs_g_cm3': density_proper_cgs_g_cm3,
        'post_temperature_cgs_K': temperature_proper_cgs_K,
        'cooling_time_proper_Myr': cooling_time_s / SECONDS_PER_MYR,
        'cooling_length_proper_kpc': cooling_length_cgs_cm / KPC_CM,
        'cooling_cells': cooling_length_cgs_cm / cell_width_cgs_cm,
    }
