"""Initial-condition and output helpers for the spherical shock example."""

from types import SimpleNamespace

import numpy as np

import radhydropy.io as rio
from radhydropy.eos import EOS
from radhydropy.units import CodeUnits, quantity_to_value


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
    result = SimpleNamespace(par=Par(), mesh=Mesh(), fluid=Fluid())
    result.par.CodeUnits = code_units
    result.par.units = SimpleNamespace(CodeUnits=code_units)
    result.par.unit_system = code_units.unit_system
    result.par.nogrid = int(runtime['mesh']['grid_cells'])
    result.par.coordsys = runtime['simulation']['coordinate_system']
    rmin = quantity_to_value(initial['rmin'], code_units.length_unit)
    rmax = quantity_to_value(initial['rmax'], code_units.length_unit)
    result.par.inner_radius = rmin
    result.par.boxsize = np.asarray([rmax])
    result.par.time = np.asarray([quantity_to_value(initial['current_time'], code_units.time_unit)])
    result.par.simulation = SimpleNamespace(current_time=result.par.time, box_size=result.par.boxsize, coordinate_system=result.par.coordsys)
    result.par.mesh = SimpleNamespace(grid_cells=result.par.nogrid, ghost_cells=0)
    result.par.hydrodynamics = SimpleNamespace(gamma=float(runtime['hydrodynamics']['gamma']))
    result.par.dual_energy = bool(runtime['hydrodynamics'].get('dual_energy', False))
    faces = np.linspace(result.par.inner_radius, result.par.boxsize[0], result.par.nogrid + 1)
    result.mesh.boundary = faces
    result.mesh.coordinate = 0.5 * (faces[1:] + faces[:-1])
    result.mesh.area = 4.0 * np.pi * faces[:-1] ** 2
    result.mesh.vol = 4.0 * np.pi / 3.0 * np.diff(faces ** 3)
    result.fluid.rho_code = np.full(result.par.nogrid, quantity_to_value(initial['initial_density'], code_units.density_unit))
    result.fluid.temp_code = np.full(result.par.nogrid, quantity_to_value(initial['temperature'], code_units.temperature_unit))
    result.fluid.mu = np.full(result.par.nogrid, float(initial['mean_molecular_weight']))
    result.fluid.vel_code = np.full(result.par.nogrid, quantity_to_value(initial['velocity'], code_units.velocity_unit))
    return result


def read_output(filename, runparams):
    """Read one output with the metadata needed by the HDF5 reader."""
    code_units = CodeUnits.from_mapping(runparams['units']['CodeUnits'])
    result = SimpleNamespace(par=Par(), mesh=Mesh(), fluid=Fluid())
    result.par.CodeUnits = code_units
    result.par.units = SimpleNamespace(CodeUnits=code_units)
    result.par.simulation = SimpleNamespace(coordinate_system='spherical')
    result.par.mesh = SimpleNamespace(
        grid_cells=int(runparams['mesh']['grid_cells']),
        ghost_cells=int(runparams['mesh']['ghost_cells']),
    )
    result.par.hydrodynamics = SimpleNamespace(
        gamma=float(runparams['hydrodynamics']['gamma']),
    )
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    result.fluid.eos = EOS(
        runparams['hydrodynamics']['eos_type'],
        result.par.hydrodynamics.gamma,
        code_units,
    )
    return result
