"""Initial-condition and dark-matter helpers for the coupled example."""

import numpy as np
import unyt
from types import SimpleNamespace

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def build_initial_condition(config):
    initial = config['initial_condition']
    runtime = config['par']
    code_units = config['_code_units']
    grid_cells = int(runtime['mesh']['grid_cells'])
    result = SimpleNamespace(par=Par(), mesh=Mesh(), fluid=Fluid())
    result.par.CodeUnits = code_units
    result.par.units = SimpleNamespace(CodeUnits=code_units)
    result.par.unit_system = code_units.unit_system
    result.par.nogrid = grid_cells
    result.par.coordsys = 'spherical'
    result.par.boxsize = np.ones(1) * initial['rmax']
    result.par.time = np.ones(1) * initial.get('current_time', 0.0 * unyt.s)
    result.par.simulation = SimpleNamespace(current_time=result.par.time, box_size=result.par.boxsize, coordinate_system='spherical')
    result.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=0)
    result.mesh.boundary = np.linspace(initial['rmin'], initial['rmax'], grid_cells + 1)
    result.mesh.coordinate = 0.75 * (result.mesh.boundary[1:]**4 - result.mesh.boundary[:-1]**4) / (result.mesh.boundary[1:]**3 - result.mesh.boundary[:-1]**3)
    result.mesh.area = 4.0 * np.pi * result.mesh.boundary[:-1]**2
    result.mesh.vol = 4.0 * np.pi / 3.0 * (result.mesh.boundary[1:]**3 - result.mesh.boundary[:-1]**3)
    result.fluid.rho_code = np.ones(grid_cells) * initial['gas_density']
    result.fluid.temp_code = np.ones(grid_cells) * initial['gas_temperature']
    result.fluid.mu = np.ones(grid_cells) * initial['mu']
    result.fluid.vel_code = np.zeros(grid_cells) * unyt.cm / unyt.s
    return result


def make_dark_matter(icparams, code_units):
    count = int(icparams['dark_matter_shells'])
    radius = np.linspace(0.05, 0.95, count)
    velocity = np.asarray(radius) * float(icparams['dark_matter_velocity_scale'])
    angular_momentum = np.full(count, float(icparams['dark_matter_angular_momentum']))
    return DarkMatterShells(
        radius=radius,
        velocity=velocity,
        mass=np.full(count, icparams['dark_matter_mass'] / count),
        angular_momentum=angular_momentum,
        softening=icparams['dark_matter_softening'],
        code_units=code_units,
    )


def load_units(runparams):
    return CodeUnits.from_mapping(runparams['units']['CodeUnits'])

