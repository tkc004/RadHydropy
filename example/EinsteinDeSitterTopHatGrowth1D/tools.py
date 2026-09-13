"""Helpers for the Einstein--de Sitter linear-growth benchmark."""

import numpy as np
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, quantity_to_value
import radhydropy.io as rio


def spherical_cell_centers(boundary_comoving_code):
    inner, outer = boundary_comoving_code[:-1], boundary_comoving_code[1:]
    return 0.75 * (outer**4 - inner**4) / (outer**3 - inner**3)


def growing_mode_velocity(radius_comoving_code, overdensity, scale_factor, hubble):
    """Supercomoving peculiar velocity for the EdS growing mode."""
    return -(scale_factor**2 * hubble * overdensity / 3.0) * np.asarray(
        radius_comoving_code
    )


def enclosed_mass_radius(
    boundary_comoving_code,
    rho_comoving_code,
    volume_comoving_code,
    target_mass_comoving_code,
):
    """Interpolate the radius enclosing ``target_mass`` from cell masses."""
    cumulative_mass_comoving_code = np.concatenate((
        [0.0], np.cumsum(np.asarray(rho_comoving_code) * volume_comoving_code)
    ))
    target_mass_comoving_code = float(np.clip(
        target_mass_comoving_code,
        cumulative_mass_comoving_code[0],
        cumulative_mass_comoving_code[-1],
    ))
    return float(np.interp(
        target_mass_comoving_code,
        cumulative_mass_comoving_code,
        np.asarray(boundary_comoving_code),
    ))


def linear_overdensity(delta_initial, scale_factor, initial_scale_factor):
    return float(delta_initial) * float(scale_factor) / float(initial_scale_factor)


def build_initial_condition(config):
    code_units = config['_code_units']
    cosmology = config['_cosmology']
    initial_condition = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    time_cosmic_code = quantity_to_value(initial_condition['time_cosmic'], code_units.time_unit)
    scale_factor = cosmology.scale_factor(time_cosmic_code)
    hubble = cosmology.hubble(time_cosmic_code)
    writer = InitialConditionWriter(
        ic_config=config["initial_condition"],
        par_config=config['par'], code_units=code_units,
    )
    sim = writer.simulation
    sim.par.simulation.coordinate_system = 'spherical'
    sim.par.tau_supercomoving_code = np.ones(1) * cosmology.supercomoving_time(time_cosmic_code)
    sim.par.simulation.tau_supercomoving_code = sim.par.tau_supercomoving_code
    sim.par.set_cosmology_model(cosmology)

    rmin_code = quantity_to_value(initial_condition['radius_inner_comoving'], code_units.length_unit)
    rmax_code = quantity_to_value(initial_condition['radius_outer_comoving'], code_units.length_unit)
    boundary = np.linspace(rmin_code, rmax_code, grid_cells + 1)
    x = spherical_cell_centers(boundary)
    writer.box_size = writer.radquantity(initial_condition['box_size_proper'])
    writer.mesh.boundary_radarray = writer.radarray(
        boundary * code_units.length_unit, representation='comoving'
    )
    writer.set_field('x_comoving_code', x)

    rho_background = cosmology.background_density(time_cosmic_code)
    rho_comoving = rho_background * scale_factor**3
    delta = float(initial_condition['overdensity'])
    inside = x < quantity_to_value(
        initial_condition['radius_perturbation_comoving'], code_units.length_unit
    )
    rho_comoving_code = rho_comoving * (1.0 + delta * inside) * np.ones(grid_cells)
    vel = growing_mode_velocity(x, delta, scale_factor, hubble)
    temp_supercomoving_code = np.ones(grid_cells) * quantity_to_value(
        initial_condition['temperature_proper'], code_units.temperature_unit,
    ) * scale_factor**2
    writer.fluid.rho_radarray = writer.radarray(
        rho_comoving_code * code_units.density_unit, representation='comoving'
    )
    writer.fluid.vel_radarray = writer.radarray(
        vel * code_units.velocity_unit, representation='supercomoving'
    )
    writer.fluid.temp_radarray = writer.radarray(
        temp_supercomoving_code * code_units.temperature_unit,
        representation='supercomoving'
    )
    sim.fluid.mu = np.ones(grid_cells) * float(initial_condition['mean_molecular_weight'])
    sim.fluid.tau_supercomoving_code = float(
        np.asarray(sim.par.tau_supercomoving_code, dtype=float).reshape(-1)[0]
    )
    return writer

def load_output_state(filename, config):
    return rio.loadhdf5(config, filename)
