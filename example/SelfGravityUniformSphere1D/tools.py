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


def build_initial_condition(config, code_units=None):
    if code_units is None:
        code_units = config['_code_units']
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
    sim.par.coordsys = icparams['coordsys']
    sim.par.boxsize = np.ones(1) * icparams['boxsize']
    sim.par.time = np.ones(1) * icparams['time']
    sim.par.simulation = SimpleNamespace(current_time=sim.par.time, box_size=sim.par.boxsize, coordinate_system='spherical')
    sim.par.mesh = SimpleNamespace(grid_cells=sim.par.nogrid, ghost_cells=0)
    sim.par.hydrodynamics = SimpleNamespace(gamma=5.0 / 3.0)

    sim.mesh.boundary = np.linspace(
        icparams['rmin'],
        icparams['rmax'],
        sim.par.nogrid + 1,
    )
    sim.mesh.coordinate = spherical_cell_centers(sim.mesh.boundary)
    sim.mesh.area = 4.0 * np.pi * sim.mesh.boundary[:-1]**2
    sim.mesh.vol = 4.0 * np.pi / 3.0 * (
        sim.mesh.boundary[1:]**3 - sim.mesh.boundary[:-1]**3
    )

    sim.fluid.rho_code = np.ones(sim.par.nogrid) * icparams['rho0']
    sim.fluid.temp_code = np.ones(sim.par.nogrid) * icparams['tempini']
    sim.fluid.mu = np.ones(sim.par.nogrid) * icparams['muini']
    sim.fluid.vel_code = np.zeros(sim.par.nogrid) * unyt.cm / unyt.s


    return sim

def read_code_units(runparams):
    return CodeUnits.from_mapping(runparams['CodeUnits'])


def read_snapshot(filename, runparams):
    code_units = read_code_units(runparams)
    result = build_initial_condition(
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





