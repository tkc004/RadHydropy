"""Initial conditions and diagnostics for a gravity-free spherical PIE shock."""

import h5py
import numpy as np
import unyt
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import quantity_to_value
from basic_hydro_utils import finalize_initial_condition


PROTON_MASS_G = unyt.mp.to_value(unyt.g)
SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
KPC_CM = (1.0 * unyt.kpc).to_value(unyt.cm)


def build_initial_condition(config):
    initial = config['initial_condition']
    par = config['par']
    code_units = config['_code_units']
    grid_cells = int(par['mesh']['grid_cells'])
    # Component-level callers may provide the already-resolved private unit
    # object without repeating the YAML ``units`` group. Complete the nested
    # runtime mapping at this boundary_proper_code before constructing Rsim.
    if 'units' not in config['par']:
        config['par'] = dict(config['par'])
        config['par']['units'] = {'CodeUnits': code_units.to_dict()}
    result = Rsim(config["par"])
    result.par.simulation.time_proper_code = quantity_to_value(initial['time'], code_units.time_unit)
    result.par.simulation.box_size = quantity_to_value(initial['boxsize'], code_units.length_unit)
    result.par.simulation.coordinate_system = 'spherical'
    result.par.mesh.grid_cells = grid_cells
    boundary_proper_code = as_named_array(quantity_to_value(
        np.linspace(initial['rmin'], initial['rmax'], grid_cells + 1), code_units.length_unit
    ))
    result.mesh.boundary_proper_code = boundary_proper_code
    width = np.diff(boundary_proper_code)
    x_proper_code = 0.75 * (boundary_proper_code[1:] ** 4 - boundary_proper_code[:-1] ** 4) / (boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3)
    volume_proper_code = 4.0 * np.pi / 3.0 * (boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3)
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=x_proper_code, boundary_proper_code=boundary_proper_code,
        width_proper_code=width, area_proper_code=4.0 * np.pi * boundary_proper_code[:-1] ** 2, volume_proper_code=volume_proper_code,
    )
    rho_proper_code = initial['hydrogen_density'] * unyt.mp / float(par['thermochemistry']['hydrogen_mass_fraction'])
    result.fluid.rho_proper_code = as_named_array(quantity_to_value(np.ones(grid_cells) * rho_proper_code, code_units.density_unit))
    midpoint_proper_code = quantity_to_value(
        0.5 * (initial['rmin'] + initial['rmax']), code_units.length_unit
    )
    outflow_velocity_proper_code = quantity_to_value(
        initial['outflow_velocity'], code_units.velocity_unit
    )
    inflow_velocity_proper_code = quantity_to_value(
        initial['inflow_velocity'], code_units.velocity_unit
    )
    result.fluid.vel_proper_code = as_named_array(np.where(
        x_proper_code < midpoint_proper_code,
        outflow_velocity_proper_code,
        inflow_velocity_proper_code,
    ))
    result.fluid.temp_proper_code = as_named_array(quantity_to_value(np.ones(grid_cells) * initial['inflow_temperature'], code_units.temperature_unit))
    result.fluid.mu = np.ones(grid_cells) * initial['muini']
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


def physical_cells(header):
    noghost = int(header.attrs.get('GhostCells', 0))
    nogrid = int(header.attrs['GridCells'])
    return slice(noghost, noghost + nogrid)


def load_snapshot(filename):
    with h5py.File(filename, 'r') as handle:
        data = handle['Data']
        header = handle['Header']
        physical = physical_cells(header)
        noghost = int(header.attrs.get('GhostCells', 0))
        nogrid = int(header.attrs['GridCells'])
        boundary_proper_code = np.asarray(
            data['boundary_proper_code'][()]
        )[noghost:noghost + nogrid + 1]
        return {
            'time_Myr': float(header['time_proper_code'][()]) / SECONDS_PER_MYR,
            'boundary_cgs_cm': boundary_proper_code,
            'density_cgs_g_cm3': np.asarray(data['rho_proper_code'][()])[physical],
            'velocity_cgs_cm_s': np.asarray(data['vel_proper_code'][()])[physical],
            'temperature_cgs_K': np.asarray(data['temp_proper_code'][()])[physical],
        }


def shock_radius(snapshot):
    """Locate the strongest compression near the colliding-stream interface."""
    boundary_proper_code = snapshot['boundary_cgs_cm']
    centers = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
    density = np.maximum(snapshot['density_cgs_g_cm3'], 1.0e-99)
    gradient = np.abs(np.diff(np.log(density)))
    start = max(2, int(0.2 * len(gradient)))
    stop = min(len(gradient) - 1, int(0.9 * len(gradient)))
    index = start + int(np.argmax(gradient[start:stop]))
    return float(centers[index] / KPC_CM)


def shock_history(filenames, output_interval_myr=None):
    rows = []
    for filename in filenames:
        snapshot = load_snapshot(filename)
        if output_interval_myr is None:
            time_myr = snapshot['time_Myr']
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
    radius = snapshot['boundary_cgs_cm']
    centers = 0.5 * (radius[1:] + radius[:-1])
    shock_kpc = shock_radius(snapshot)
    shock_index = int(np.argmin(np.abs(centers / KPC_CM - shock_kpc)))
    left = slice(max(0, shock_index - 4), shock_index)
    right = slice(shock_index + 1, min(len(centers), shock_index + 5))
    left_temperature = float(np.median(snapshot['temperature_cgs_K'][left]))
    right_temperature = float(np.median(snapshot['temperature_cgs_K'][right]))
    post_slice = right if right_temperature >= left_temperature else left
    density = float(np.median(snapshot['density_cgs_g_cm3'][post_slice]))
    temperature = float(np.median(snapshot['temperature_cgs_K'][post_slice]))
    velocity = float(np.median(np.abs(snapshot['velocity_cgs_cm_s'][post_slice])))
    n_h = hydrogen_mass_fraction * density / PROTON_MASS_G
    heating, cooling = table.rates(
        temperature, n_h, metallicity=metallicity, redshift=0.0
    )
    net_rate = max(float(np.asarray(cooling) - np.asarray(heating)), 1.0e-99)
    thermal_energy = 1.5 * density * 1.380649e-16 * temperature / (
        mu * PROTON_MASS_G
    )
    cooling_time_s = thermal_energy / net_rate
    cooling_length_cgs_cm = abs(velocity) * cooling_time_s
    cell_width_cgs_cm = float(np.median(np.diff(radius)))
    return {
        'shock_radius_kpc': shock_kpc,
        'post_density_cgs_g_cm3': density,
        'post_temperature_cgs_K': temperature,
        'cooling_time_Myr': cooling_time_s / SECONDS_PER_MYR,
        'cooling_length_kpc': cooling_length_cgs_cm / KPC_CM,
        'cooling_cells': cooling_length_cgs_cm / cell_width_cgs_cm,
    }
