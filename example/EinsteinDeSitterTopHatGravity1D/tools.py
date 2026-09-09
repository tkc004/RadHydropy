"""Initial conditions and analytic solution for the EdS top-hat test."""

import numpy as np
import unyt

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits, quantity_to_value
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, SUPERCOMOVING_RUNTIME_FIELDS


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
    sim = Rsim(config['par'])
    sim.par.mesh.grid_cells = grid_cells
    sim.par.mesh.ghost_cells = 0
    sim.par.simulation.coordinate_system = 'spherical'
    sim.par.simulation.box_size_comoving_code = np.ones(1) * quantity_to_value(
        initial_condition['box_size_proper'], code_units.length_unit
    )
    cosmic_time = quantity_to_value(initial_condition['time_cosmic'], code_units.time_unit)
    sim.par.simulation.tau_supercomoving_code = np.ones(1) * cosmology.supercomoving_time(cosmic_time)
    sim.par.tau_supercomoving_code = np.ones(1) * cosmology.supercomoving_time(cosmic_time)
    sim.par.cosmological_expansion = True
    sim.par.supercomoving_coordinates = True
    sim.par.cosmological_gravity = True
    sim.par.selfgravity = True
    sim.par.externalgravity = False
    sim.par.cosmology = cosmology
    sim.par.cosmology_type = cosmology.type_name
    sim.par.cosmology_t_ref = cosmology.t_ref
    sim.par.cosmology_a_ref = cosmology.a_ref
    sim.par.coordinate_frame = 'comoving'
    sim.par.time_coordinate = 'supercomoving'
    sim.par.velocity_representation = 'supercomoving_peculiar'
    sim.par.density_representation = 'comoving'
    sim.par.pressure_representation = 'supercomoving'
    sim.par.temperature_representation = 'supercomoving'

    sim.mesh.boundary_comoving_code = np.linspace(
        initial_condition['radius_inner_comoving'],
        initial_condition['radius_outer_comoving'], grid_cells + 1,
    )
    sim.mesh.x_comoving_code = spherical_cell_centers(sim.mesh.boundary_comoving_code)
    sim.mesh.area_comoving_code = 4.0 * np.pi * sim.mesh.boundary_comoving_code[:-1]**2
    sim.mesh.volume_comoving_code = 4.0 * np.pi / 3.0 * (
        sim.mesh.boundary_comoving_code[1:]**3 - sim.mesh.boundary_comoving_code[:-1]**3
    )

    background = cosmology.background_density(cosmic_time)
    background_comoving = background * cosmology.scale_factor(cosmic_time)**3
    inside = sim.mesh.x_comoving_code < quantity_to_value(
        initial_condition['radius_perturbation_comoving'], code_units.length_unit
    )
    sim.fluid.rho_comoving_code = background_comoving * (
        1.0 + float(initial_condition['overdensity']) * inside
    ) * np.ones(grid_cells)
    gamma = 5.0 / 3.0
    temperature = quantity_to_value(
        initial_condition['temperature_proper'], code_units.temperature_unit
    )
    sim.fluid.temp_supercomoving_code = temperature * cosmology.scale_factor(cosmic_time)**2 * np.ones(grid_cells)
    sim.fluid.mu = np.ones(grid_cells) * float(initial_condition['mean_molecular_weight'])
    sim.fluid.vel_supercomoving_code = np.zeros(grid_cells)
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS,
        x_comoving_code=sim.mesh.x_comoving_code,
        boundary_comoving_code=sim.mesh.boundary_comoving_code,
        width_comoving_code=np.diff(sim.mesh.boundary_comoving_code),
        area_comoving_code=sim.mesh.area_comoving_code,
        volume_comoving_code=sim.mesh.volume_comoving_code,
    )
    sim.fluid.pre_supercomoving_code = sim.fluid.rho_comoving_code * sim.fluid.temp_supercomoving_code
    sim.fluid.tau_supercomoving_code = float(sim.par.simulation.tau_supercomoving_code[0])
    sim.fluid.runtime_fields = SUPERCOMOVING_RUNTIME_FIELDS
    sim.fluid.runtime_state = FluidRuntimeState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS, rho_comoving_code=sim.fluid.rho_comoving_code,
        vel_supercomoving_code=sim.fluid.vel_supercomoving_code,
        pre_supercomoving_code=sim.fluid.pre_supercomoving_code,
        temp_supercomoving_code=sim.fluid.temp_supercomoving_code,
        tau_supercomoving_code=sim.fluid.tau_supercomoving_code, mu_dimensionless=sim.fluid.mu,
    )

    return sim

def load_output_state(filename, config):
    result = Rsim(config['par'])
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    return result
