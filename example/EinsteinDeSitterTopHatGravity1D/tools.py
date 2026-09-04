"""Initial conditions and analytic solution for the EdS top-hat test."""

import numpy as np
from types import SimpleNamespace
import unyt

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.units import CodeUnits, quantity_to_value


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


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
    sim = SimpleNamespace()
    icparams = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    sim.par = Par()
    sim.mesh = Mesh()
    sim.fluid = Fluid()
    sim.par.CodeUnits = code_units
    sim.par.units = SimpleNamespace(CodeUnits=code_units)
    sim.par.unit_system = code_units.unit_system
    sim.par.nogrid = grid_cells
    sim.par.coordsys = 'spherical'
    sim.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=0)
    sim.par.hydrodynamics = SimpleNamespace(gamma=5.0 / 3.0)
    sim.par.boxsize = np.ones(1) * icparams['boxsize']
    cosmic_time = float(icparams['cosmic_time'])
    sim.par.simulation = SimpleNamespace(
        current_time=np.ones(1) * cosmology.supercomoving_time(cosmic_time),
        box_size=np.ones(1) * icparams['boxsize'],
        coordinate_system='spherical',
    )
    sim.par.time = np.ones(1) * cosmology.supercomoving_time(cosmic_time)
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
        icparams['rmin'], icparams['rmax'], sim.par.nogrid + 1,
    )
    sim.mesh.coordinate = spherical_cell_centers(sim.mesh.boundary)
    sim.mesh.area = 4.0 * np.pi * sim.mesh.boundary[:-1]**2
    sim.mesh.vol = 4.0 * np.pi / 3.0 * (
        sim.mesh.boundary[1:]**3 - sim.mesh.boundary[:-1]**3
    )

    background = cosmology.background_density(cosmic_time)
    background_comoving = background * cosmology.scale_factor(cosmic_time)**3
    inside = sim.mesh.coordinate < float(icparams['top_hat_radius'])
    sim.fluid.rho_code = background_comoving * (
        1.0 + float(icparams['overdensity']) * inside
    ) * np.ones(sim.par.nogrid)
    gamma = 5.0 / 3.0
    temperature = quantity_to_value(
        icparams['tempini'], code_units.temperature_unit
    )
    sim.fluid.temp_code = temperature * cosmology.scale_factor(cosmic_time)**2 * np.ones(sim.par.nogrid)
    sim.fluid.mu = np.ones(sim.par.nogrid) * float(icparams['muini'])
    sim.fluid.vel_code = np.zeros(sim.par.nogrid)


    return sim

def read_code_units(runparams):
    return CodeUnits.from_mapping(runparams['CodeUnits'])


def read_snapshot(filename, runparams):
    code_units = read_code_units(runparams)
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





