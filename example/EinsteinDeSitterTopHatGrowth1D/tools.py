"""Helpers for the Einstein--de Sitter linear-growth benchmark."""

import numpy as np
from types import SimpleNamespace
import unyt

from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.units import CodeUnits, quantity_to_value
import radhydropy.io as rio


class Par: pass
class Mesh: pass
class Fluid: pass


def spherical_cell_centers(boundary):
    inner, outer = boundary[:-1], boundary[1:]
    return 0.75 * (outer**4 - inner**4) / (outer**3 - inner**3)


def growing_mode_velocity(radius, overdensity, scale_factor, hubble):
    """Supercomoving peculiar velocity for the EdS growing mode."""
    return -(scale_factor**2 * hubble * overdensity / 3.0) * np.asarray(radius)


def enclosed_mass_radius(boundary, density, cell_volume, target_mass):
    """Interpolate the radius enclosing ``target_mass`` from cell masses."""
    cumulative = np.concatenate(([0.0], np.cumsum(np.asarray(density) * cell_volume)))
    target_mass = float(np.clip(target_mass, cumulative[0], cumulative[-1]))
    return float(np.interp(target_mass, cumulative, np.asarray(boundary)))


def linear_overdensity(delta_initial, scale_factor, initial_scale_factor):
    return float(delta_initial) * float(scale_factor) / float(initial_scale_factor)


def build_initial_condition(config):
    code_units = config['_code_units']
    cosmology = config['_cosmology']
    sim = SimpleNamespace()
    icparams = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    sim.par, sim.mesh, sim.fluid = Par(), Mesh(), Fluid()
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
    scale_factor = cosmology.scale_factor(cosmic_time)
    hubble = cosmology.hubble(cosmic_time)
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

    rho_background = cosmology.background_density(cosmic_time)
    rho_comoving = rho_background * scale_factor**3
    delta = float(icparams['overdensity'])
    inside = sim.mesh.coordinate < float(icparams['top_hat_radius'])
    sim.fluid.rho_code = rho_comoving * (1.0 + delta * inside) * np.ones(sim.par.nogrid)
    sim.fluid.vel_code = growing_mode_velocity(
        sim.mesh.coordinate, delta, scale_factor, hubble,
    )
    sim.fluid.temp_code = np.ones(sim.par.nogrid) * quantity_to_value(
        icparams['tempini'], code_units.temperature_unit,
    ) * scale_factor**2
    sim.fluid.mu = np.ones(sim.par.nogrid) * float(icparams['muini'])


    return sim

def read_snapshot(filename, runparams):
    units = CodeUnits.from_mapping(runparams['CodeUnits'])
    cosmology = EinsteinDeSitter.from_code_units(units)
    result = build_initial_condition({
        'par': {'mesh': {'grid_cells': 1}},
        'initial_condition': {
            'boxsize': units.length_unit, 'rmin': 0.0 * units.length_unit,
            'rmax': units.length_unit, 'cosmic_time': 1.0,
            'top_hat_radius': 0.5, 'overdensity': 0.0,
            'tempini': units.temperature_unit, 'muini': 1.0,
        },
        '_code_units': units, '_cosmology': cosmology,
    })
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    return result





