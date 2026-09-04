"""Helpers for the uniform-density spherical self-gravity diagnostic."""

import numpy as np
import unyt
from types import SimpleNamespace

from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS
import radhydropy.io as rio
from radhydropy.units import CodeUnits


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def spherical_cell_centers(boundary):
    """Return volume-weighted centers for spherical cells."""
    inner = boundary[:-1]
    outer = boundary[1:]
    denominator = outer**3 - inner**3
    return 0.75 * (outer**4 - inner**4) / denominator


def uniform_sphere_acceleration(radius, rho0):
    """Return the analytic interior field of a uniform-density sphere."""
    radius = radius.to(unyt.cm)
    rho0 = rho0.to(unyt.g / unyt.cm**3)
    return (
        -4.0 * np.pi / 3.0
        * (GRAVITATIONAL_CONSTANT_CGS * unyt.cm**3 / (unyt.g * unyt.s**2))
        * rho0
        * radius
    ).to(unyt.cm / unyt.s**2)


class Simwrap:
    """Build the uniform-density initial condition for ``writehdf5``."""

    def __init__(self, config, code_units):
        icparams = config['initial_condition']
        grid_cells = int(config['par']['mesh']['grid_cells'])
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.CodeUnits = code_units
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.unit_system = code_units.unit_system
        self.par.nogrid = grid_cells
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = np.ones(1) * icparams['boxsize']
        self.par.time = np.ones(1) * icparams['time']
        self.par.simulation = SimpleNamespace(current_time=self.par.time, box_size=self.par.boxsize, coordinate_system='spherical')
        self.par.mesh = SimpleNamespace(grid_cells=self.par.nogrid, ghost_cells=0)
        self.par.hydrodynamics = SimpleNamespace(gamma=5.0 / 3.0)

        self.mesh.boundary = np.linspace(
            icparams['rmin'],
            icparams['rmax'],
            self.par.nogrid + 1,
        )
        self.mesh.coordinate = spherical_cell_centers(self.mesh.boundary)
        self.mesh.area = 4.0 * np.pi * self.mesh.boundary[:-1]**2
        self.mesh.vol = 4.0 * np.pi / 3.0 * (
            self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3
        )

        self.fluid.rho_code = np.ones(self.par.nogrid) * icparams['rho0']
        self.fluid.temp_code = np.ones(self.par.nogrid) * icparams['tempini']
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['muini']
        self.fluid.vel_code = np.zeros(self.par.nogrid) * unyt.cm / unyt.s


def read_code_units(runparams):
    return CodeUnits.from_mapping(runparams['CodeUnits'])


def read_snapshot(filename, runparams):
    code_units = read_code_units(runparams)
    result = Simwrap(
        {
            'nogrid': 1,
            'coordsys': 'spherical',
            'boxsize': 1.0 * code_units.length_unit,
            'rmin': 0.0 * code_units.length_unit,
            'rmax': 1.0 * code_units.length_unit,
            'time': 0.0 * code_units.time_unit,
            'rho0': 1.0 * code_units.density_unit,
            'tempini': 1.0 * code_units.temperature_unit,
            'muini': 1.0,
        },
        code_units,
    )
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    return result
