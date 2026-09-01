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


class Simwrap:
    """Build a supercomoving spherical top-hat IC."""

    def __init__(self, icparams, code_units, cosmology):
        self.par, self.mesh, self.fluid = Par(), Mesh(), Fluid()
        self.par.CodeUnits = code_units
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.unit_system = code_units.unit_system
        self.par.nogrid = int(icparams['nogrid'])
        self.par.coordsys = 'spherical'
        self.par.mesh = SimpleNamespace(grid_cells=int(icparams['nogrid']), ghost_cells=0)
        self.par.hydrodynamics = SimpleNamespace(gamma=5.0 / 3.0)
        self.par.boxsize = np.ones(1) * icparams['boxsize']
        cosmic_time = float(icparams['cosmic_time'])
        self.par.simulation = SimpleNamespace(
            current_time=np.ones(1) * cosmology.supercomoving_time(cosmic_time),
            box_size=np.ones(1) * icparams['boxsize'],
            coordinate_system='spherical',
        )
        scale_factor = cosmology.scale_factor(cosmic_time)
        hubble = cosmology.hubble(cosmic_time)
        self.par.time = np.ones(1) * cosmology.supercomoving_time(cosmic_time)
        self.par.cosmological_expansion = True
        self.par.supercomoving_coordinates = True
        self.par.cosmological_gravity = True
        self.par.selfgravity = True
        self.par.externalgravity = False
        self.par.cosmology = cosmology
        self.par.cosmology_type = cosmology.type_name
        self.par.cosmology_t_ref = cosmology.t_ref
        self.par.cosmology_a_ref = cosmology.a_ref
        self.par.coordinate_frame = 'comoving'
        self.par.time_coordinate = 'supercomoving'
        self.par.velocity_representation = 'supercomoving_peculiar'
        self.par.density_representation = 'comoving'
        self.par.pressure_representation = 'supercomoving'
        self.par.temperature_representation = 'supercomoving'

        self.mesh.boundary = np.linspace(
            icparams['rmin'], icparams['rmax'], self.par.nogrid + 1,
        )
        self.mesh.coordinate = spherical_cell_centers(self.mesh.boundary)
        self.mesh.area = 4.0 * np.pi * self.mesh.boundary[:-1]**2
        self.mesh.vol = 4.0 * np.pi / 3.0 * (
            self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3
        )

        rho_background = cosmology.background_density(cosmic_time)
        rho_comoving = rho_background * scale_factor**3
        delta = float(icparams['overdensity'])
        inside = self.mesh.coordinate < float(icparams['top_hat_radius'])
        self.fluid.rho = rho_comoving * (1.0 + delta * inside) * np.ones(self.par.nogrid)
        self.fluid.vel = growing_mode_velocity(
            self.mesh.coordinate, delta, scale_factor, hubble,
        )
        self.fluid.temp = np.ones(self.par.nogrid) * quantity_to_value(
            icparams['tempini'], code_units.temperature_unit,
        ) * scale_factor**2
        self.fluid.mu = np.ones(self.par.nogrid) * float(icparams['muini'])


def read_snapshot(filename, runparams):
    units = CodeUnits.from_mapping(runparams['CodeUnits'])
    cosmology = EinsteinDeSitter.from_code_units(units)
    result = Simwrap({
        'nogrid': 1, 'boxsize': units.length_unit,
        'rmin': 0.0 * units.length_unit, 'rmax': units.length_unit,
        'cosmic_time': 1.0, 'top_hat_radius': 0.5,
        'overdensity': 0.0, 'tempini': units.temperature_unit, 'muini': 1.0,
    }, units, cosmology)
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    return result
