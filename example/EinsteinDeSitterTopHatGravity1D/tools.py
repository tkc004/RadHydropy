"""Initial conditions and analytic solution for the EdS top-hat test."""

import numpy as np
import unyt

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits, quantity_to_value
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, SUPERCOMOVING_RUNTIME_FIELDS


def spherical_cell_centers(boundary):
    inner = boundary[:-1]
    outer = boundary[1:]
    return 0.75 * (outer**4 - inner**4) / (outer**3 - inner**3)


def top_hat_acceleration(radius, top_hat_radius, overdensity, rho_background,
                         scale_factor, gravitational_constant):
    """Analytic supercomoving acceleration from a spherical density excess."""
    radius = np.asarray(radius, dtype=float)
    enclosed_radius = np.minimum(radius, float(top_hat_radius))
    acceleration = (
        -4.0 * np.pi / 3.0
        * gravitational_constant
        * scale_factor
        * float(overdensity) * float(rho_background)
        * enclosed_radius**3 / np.maximum(radius, 1.0e-300)**2
    )
    return np.where(radius > 0.0, acceleration, 0.0)


def build_initial_condition(config):
    code_units = config['_code_units']
    cosmology = config['_cosmology']
    initial_condition = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    sim = Rsim(config['par'])
    sim.par.mesh.grid_cells = grid_cells
    sim.par.mesh.ghost_cells = 0
    sim.par.simulation.coordinate_system = 'spherical'
    sim.par.simulation.box_size = np.ones(1) * quantity_to_value(
        initial_condition['boxsize'], code_units.length_unit
    )
    cosmic_time = float(initial_condition['cosmic_time'])
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

    sim.mesh.boundary = np.linspace(
        initial_condition['rmin'], initial_condition['rmax'], grid_cells + 1,
    )
    sim.mesh.coordinate = spherical_cell_centers(sim.mesh.boundary)
    sim.mesh.area = 4.0 * np.pi * sim.mesh.boundary[:-1]**2
    sim.mesh.vol = 4.0 * np.pi / 3.0 * (
        sim.mesh.boundary[1:]**3 - sim.mesh.boundary[:-1]**3
    )

    background = cosmology.background_density(cosmic_time)
    background_comoving = background * cosmology.scale_factor(cosmic_time)**3
    inside = sim.mesh.coordinate < float(initial_condition['top_hat_radius'])
    sim.fluid.rho_comoving_code = background_comoving * (
        1.0 + float(initial_condition['overdensity']) * inside
    ) * np.ones(grid_cells)
    gamma = 5.0 / 3.0
    temperature = quantity_to_value(
        initial_condition['tempini'], code_units.temperature_unit
    )
    sim.fluid.temp_supercomoving_code = temperature * cosmology.scale_factor(cosmic_time)**2 * np.ones(grid_cells)
    sim.fluid.mu = np.ones(grid_cells) * float(initial_condition['muini'])
    sim.fluid.vel_supercomoving_code = np.zeros(grid_cells)
    sim.mesh.boundary_comoving_code = sim.mesh.boundary
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS, coordinate=sim.mesh.coordinate,
        boundary=sim.mesh.boundary, width=np.diff(sim.mesh.boundary),
        area=sim.mesh.area, volume=sim.mesh.vol,
    )
    sim.fluid.pre_supercomoving_code = sim.fluid.rho_comoving_code * sim.fluid.temp_supercomoving_code
    sim.fluid.tau_supercomoving_code = float(sim.par.simulation.tau_supercomoving_code[0])
    sim.fluid.runtime_fields = SUPERCOMOVING_RUNTIME_FIELDS
    sim.fluid.runtime_state = FluidRuntimeState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS, density=sim.fluid.rho_comoving_code,
        velocity=sim.fluid.vel_supercomoving_code,
        pressure=sim.fluid.pre_supercomoving_code,
        temperature=sim.fluid.temp_supercomoving_code,
        time=sim.fluid.tau_supercomoving_code, mu=sim.fluid.mu,
    )

    return sim

def read_code_units(par_config):
    return CodeUnits.from_mapping(par_config['CodeUnits'])


def read_snapshot(filename, par_config):
    code_units = read_code_units(par_config)
    cosmology = EinsteinDeSitter.from_code_units(code_units)
    result = build_initial_condition({
        'par': {'mesh': {'grid_cells': 1}},
        'initial_condition': {
            'boxsize': 1.0 * code_units.length_unit,
            'rmin': 0.0 * code_units.length_unit,
            'rmax': 1.0 * code_units.length_unit,
            'cosmic_time': 1.0, 'top_hat_radius': 0.5,
            'overdensity': 0.0, 'tempini': 1.0 * code_units.temperature_unit,
            'muini': 1.0,
        },
        '_code_units': code_units, '_cosmology': cosmology,
    })
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    return result
