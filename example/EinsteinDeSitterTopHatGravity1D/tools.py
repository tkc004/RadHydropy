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


class Simwrap:
    """Construct a supercomoving top-hat IC accepted by ``writehdf5``."""

    def __init__(self, icparams, code_units, cosmology):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
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

        background = cosmology.background_density(cosmic_time)
        background_comoving = background * cosmology.scale_factor(cosmic_time)**3
        inside = self.mesh.coordinate < float(icparams['top_hat_radius'])
        self.fluid.rho = background_comoving * (
            1.0 + float(icparams['overdensity']) * inside
        ) * np.ones(self.par.nogrid)
        gamma = 5.0 / 3.0
        temperature = quantity_to_value(
            icparams['tempini'], code_units.temperature_unit
        )
        self.fluid.temp = temperature * cosmology.scale_factor(cosmic_time)**2 * np.ones(self.par.nogrid)
        self.fluid.mu = np.ones(self.par.nogrid) * float(icparams['muini'])
        self.fluid.vel = np.zeros(self.par.nogrid)


def read_code_units(runparams):
    return CodeUnits.from_mapping(runparams['CodeUnits'])


def read_snapshot(filename, runparams):
    code_units = read_code_units(runparams)
    cosmology = EinsteinDeSitter.from_code_units(code_units)
    result = Simwrap({
        'nogrid': 1, 'boxsize': 1.0 * code_units.length_unit,
        'rmin': 0.0 * code_units.length_unit,
        'rmax': 1.0 * code_units.length_unit,
        'cosmic_time': 1.0, 'top_hat_radius': 0.5,
        'overdensity': 0.0, 'tempini': 1.0 * code_units.temperature_unit,
        'muini': 1.0,
    }, code_units, cosmology)
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    return result
