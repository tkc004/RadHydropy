"""Initial conditions and analytic solution for the EdS top-hat test."""

import numpy as np
import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.cosmology_context import CosmologyContext
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, quantity_to_value


def spherical_cell_centers(boundary_comoving_code):
    inner = boundary_comoving_code[:-1]
    outer = boundary_comoving_code[1:]
    return 0.75 * (outer**4 - inner**4) / (outer**3 - inner**3)


def top_hat_acceleration(radius_comoving_code, radius_top_hat_comoving_code,
                         overdensity_dimensionless, rho_background_comoving_code,
                         scale_factor, gravitational_constant):
    """Analytic supercomoving acceleration from a spherical density excess."""
    radius_comoving_code = np.asarray(radius_comoving_code, dtype=float)
    enclosed_radius_comoving_code = np.minimum(
        radius_comoving_code, float(radius_top_hat_comoving_code)
    )
    acceleration = (
        -4.0 * np.pi / 3.0
        * gravitational_constant
        * scale_factor
        * float(overdensity_dimensionless) * float(rho_background_comoving_code)
        * enclosed_radius_comoving_code**3
        / np.maximum(radius_comoving_code, 1.0e-300)**2
    )
    return np.where(radius_comoving_code > 0.0, acceleration, 0.0)


def build_initial_condition(config):
    code_units = config['_code_units']
    cosmology = config['_cosmology']
    initial_condition = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    time_cosmic_code = quantity_to_value(initial_condition['time_cosmic'], code_units.time_unit)
    scale_factor = float(cosmology.scale_factor(time_cosmic_code))
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=code_units,
        cosmology_context=CosmologyContext(
            gamma=float(config['par'].get('hydrodynamics', {}).get('gamma', 5.0 / 3.0)),
            cosmology=cosmology.type_name, scale_factor=scale_factor,
            hubble_parameter_km_s_Mpc=float(cosmology.hubble(time_cosmic_code)) * (
                code_units.velocity_unit.to_value('km/s')
                / code_units.length_unit.to_value('Mpc')
            ),
        ),
    )
    sim = writer.simulation
    sim.par.simulation.coordinate_system = 'spherical'
    sim.par.tau_supercomoving_code = np.ones(1) * cosmology.supercomoving_time(time_cosmic_code)
    sim.par.simulation.tau_supercomoving_code = sim.par.tau_supercomoving_code
    sim.par.set_cosmology_model(cosmology)

    radius_inner_comoving_code = quantity_to_value(
        initial_condition['radius_inner_comoving'], code_units.length_unit
    )
    radius_outer_comoving_code = quantity_to_value(
        initial_condition['radius_outer_comoving'], code_units.length_unit
    )
    boundary_comoving_code = np.linspace(
        radius_inner_comoving_code, radius_outer_comoving_code, grid_cells + 1,
    )
    x_comoving_code = spherical_cell_centers(boundary_comoving_code)
    writer.box_size = writer.radquantity(initial_condition['box_size_proper'])
    writer.mesh.boundary_radarray = writer.radarray(
        boundary_comoving_code * code_units.length_unit,
        representation='comoving',
    )
    writer.set_field('x_comoving_code', x_comoving_code)

    background = cosmology.background_density(time_cosmic_code)
    background_comoving = background * cosmology.scale_factor(time_cosmic_code)**3
    inside = x_comoving_code < quantity_to_value(
        initial_condition['radius_perturbation_comoving'], code_units.length_unit
    )
    rho_comoving_code = background_comoving * (
        1.0 + float(initial_condition['overdensity']) * inside
    ) * np.ones(grid_cells)
    gamma = 5.0 / 3.0
    temperature_cgs_K = quantity_to_value(
        initial_condition['temperature_proper'], code_units.temperature_unit
    )
    temp_supercomoving_code = temperature_cgs_K * scale_factor**2 * np.ones(grid_cells)
    pre_supercomoving_code = rho_comoving_code * temp_supercomoving_code
    writer.fluid.rho_radarray = writer.radarray(
        rho_comoving_code * code_units.density_unit, representation='comoving'
    )
    writer.fluid.vel_radarray = writer.radarray(
        np.zeros(grid_cells) * code_units.velocity_unit,
        representation='supercomoving'
    )
    writer.fluid.temp_radarray = writer.radarray(
        temp_supercomoving_code * code_units.temperature_unit,
        representation='supercomoving',
    )
    writer.fluid.pre_radarray = writer.radarray(
        pre_supercomoving_code * code_units.pressure_unit,
        representation='supercomoving',
    )
    sim.fluid.mu = np.ones(grid_cells) * float(initial_condition['mean_molecular_weight'])
    sim.fluid.tau_supercomoving_code = float(sim.par.simulation.tau_supercomoving_code[0])
    return writer

def load_output_state(filename, config):
    return rio.loadhdf5(config, filename)
