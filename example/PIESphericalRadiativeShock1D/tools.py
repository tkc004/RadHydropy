"""Initial conditions and diagnostics for a gravity-free spherical PIE shock."""

from types import SimpleNamespace

import h5py
import numpy as np
import unyt


PROTON_MASS_G = unyt.mp.to_value(unyt.g)
SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
KPC_CM = (1.0 * unyt.kpc).to_value(unyt.cm)


def build_initial_condition(config):
    initial = config['initial_condition']
    par = config['par']
    code_units = config['_code_units']
    grid_cells = int(par['mesh']['grid_cells'])
    result = SimpleNamespace()
    result.par = SimpleNamespace(
        units=SimpleNamespace(CodeUnits=code_units),
        time=initial['time'] * np.ones(1),
        simulation=SimpleNamespace(
            current_time=initial['time'], box_size=initial['boxsize'],
            coordinate_system='spherical',
        ),
        mesh=SimpleNamespace(grid_cells=grid_cells, ghost_cells=0),
    )
    result.mesh = SimpleNamespace()
    result.fluid = SimpleNamespace()
    result.mesh.boundary = np.linspace(initial['rmin'], initial['rmax'], grid_cells + 1)
    rho = initial['hydrogen_density'] * unyt.mp / float(par['thermochemistry']['hydrogen_mass_fraction'])
    result.fluid.rho_code = np.ones(grid_cells) * rho
    coordinate = 0.5 * (result.mesh.boundary[1:] + result.mesh.boundary[:-1])
    midpoint = 0.5 * (initial['rmin'] + initial['rmax'])
    result.fluid.vel_code = np.where(
        coordinate < midpoint, initial['outflow_velocity'], initial['inflow_velocity']
    )
    result.fluid.temp_code = np.ones(grid_cells) * initial['inflow_temperature']
    result.fluid.mu = np.ones(grid_cells) * initial['muini']
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
        boundary = np.asarray(
            data['Boundary'][()]
        )[noghost:noghost + nogrid + 1]
        return {
            'time_Myr': float(header.attrs['Time']) / SECONDS_PER_MYR,
            'boundary_cgs_cm': boundary,
            'density_cgs_g_cm3': np.asarray(data['Density'][()])[physical],
            'velocity_cgs_cm_s': np.asarray(data['Velocity'][()])[physical],
            'temperature_cgs_K': np.asarray(data['Temperature'][()])[physical],
        }


def shock_radius(snapshot):
    """Locate the strongest compression near the colliding-stream interface."""
    boundary = snapshot['boundary_cgs_cm']
    centers = 0.5 * (boundary[1:] + boundary[:-1])
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

