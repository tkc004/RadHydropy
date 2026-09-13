"""Initial conditions and diagnostics for a gravity-free spherical PIE shock."""

import numpy as np
import unyt
import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits


PROTON_MASS_G = unyt.mp.to_value(unyt.g)
SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
KPC_CM = (1.0 * unyt.kpc).to_value(unyt.cm)


def build_initial_condition(config):
    initial = config['initial_condition']
    par = config['par']
    code_units = config['_code_units']
    grid_cells = int(par['mesh']['grid_cells'])
    boundary_proper_unyt = np.linspace(initial['radius_inner_proper'], initial['radius_outer_proper'], grid_cells + 1)
    rho_proper_unyt = (
        initial['hydrogen_number_density'] * unyt.mp
        / float(par['thermochemistry']['hydrogen_mass_fraction'])
    )
    coordinate_proper_unyt = 0.5 * (boundary_proper_unyt[:-1] + boundary_proper_unyt[1:])
    velocity_proper_unyt = np.where(
        coordinate_proper_unyt < 0.5 * (initial['radius_inner_proper'] + initial['radius_outer_proper']),
        initial['vel_outflow_proper'], initial['vel_inflow_proper'])
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=code_units, ic_config=initial,
    )
    writer.box_size = writer.radquantity(boundary_proper_unyt[-1])
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(np.ones(grid_cells) * rho_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(velocity_proper_unyt)
    writer.fluid.temp_radarray = writer.radarray(np.ones(grid_cells) * initial['temperature_inflow_proper'])
    writer.simulation.fluid.mu = np.full(grid_cells, initial['mean_molecular_weight'])
    return writer


def load_output_state(filename, config):
    """Load one snapshot through the configured canonical runtime state."""
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    snapshot = rio.loadhdf5(config, str(filename))
    first = int(snapshot.par.mesh.ghost_cells)
    count = int(snapshot.par.mesh.grid_cells)
    physical = slice(first, first + count)
    boundary_proper_code = snapshot.mesh.boundary_radarray.to_value(code_units.length_unit)[
        first:first + count + 1
    ]
    return {
        'time_proper_Myr': (
            float(np.asarray(snapshot.fluid.time_proper_code).reshape(-1)[0])
            * float(code_units.time_unit.to_value('s')) / SECONDS_PER_MYR
        ),
        'boundary_proper_cgs_cm': snapshot.mesh.boundary_radarray.to(unyt.cm).value[first:first + count + 1],
        'rho_proper_cgs_g_cm3': snapshot.fluid.rho_radarray.to(unyt.g / unyt.cm**3).value[physical],
        'vel_peculiar_proper_cgs_cm_s': snapshot.fluid.vel_radarray.to(unyt.cm / unyt.s).value[physical],
        'temperature_proper_cgs_K': snapshot.fluid.temp_radarray.to(unyt.K).value[physical],
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
            time_proper_Myr = snapshot['time_proper_Myr']
        else:
            # Current HDF5 output headers do not preserve the evolving time
            # for this non-cosmological run.  The numbered output and the
            # configured fixed cadence are unambiguous.
            try:
                output_index = int(filename.stem.rsplit('_', 1)[1])
            except (AttributeError, IndexError, ValueError):
                output_index = len(rows)
            time_proper_Myr = output_index * float(output_interval_myr)
        rows.append((time_proper_Myr, shock_radius(snapshot)))
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
