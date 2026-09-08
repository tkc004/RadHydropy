"""Initial-condition and output helpers for the spherical shock example."""

import numpy as np

import radhydropy.io as rio
from radhydropy.eos import EOS
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import CodeUnits, quantity_to_value


def build_initial_condition(config):
    initial = config['initial_condition']

    code_units = config['_code_units']
    result = Rsim(config["par"])
    result.par.mesh.grid_cells = int(config["par"]['mesh']['grid_cells'])
    result.par.mesh.ghost_cells = 0
    rmin = quantity_to_value(initial['rmin'], code_units.length_unit)
    rmax = quantity_to_value(initial['rmax'], code_units.length_unit)
    result.par.simulation.box_size = np.asarray([rmax])
    result.par.simulation.time_proper_code = quantity_to_value(initial['current_time'], code_units.time_unit)
    faces = np.linspace(rmin, rmax, result.par.mesh.grid_cells + 1)
    result.mesh.boundary_proper_code = faces
    result.mesh.x_proper_code = 0.5 * (faces[1:] + faces[:-1])
    result.mesh.width_proper_code = np.diff(faces)
    result.mesh.area_proper_code = 4.0 * np.pi * faces[:-1] ** 2
    result.mesh.volume_proper_code = 4.0 * np.pi / 3.0 * np.diff(faces ** 3)
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=result.mesh.x_proper_code,
        boundary_proper_code=faces, width_proper_code=result.mesh.width_proper_code,
        area_proper_code=result.mesh.area_proper_code, volume_proper_code=result.mesh.volume_proper_code,
    )
    result.fluid.rho_proper_code = np.full(result.par.mesh.grid_cells, quantity_to_value(initial['initial_density'], code_units.density_unit))
    result.fluid.temp_proper_code = np.full(result.par.mesh.grid_cells, quantity_to_value(initial['temperature'], code_units.temperature_unit))
    result.fluid.mu = np.full(result.par.mesh.grid_cells, float(initial['mean_molecular_weight']))
    result.fluid.vel_proper_code = np.full(result.par.mesh.grid_cells, quantity_to_value(initial['velocity'], code_units.velocity_unit))
    result.fluid.SetUpFluid(result.par, result.mesh)
    result.solver.SetConserved(result.mesh, result.fluid, verbose=0)
    return Rsim.FromComponents(result.par, result.mesh, result.fluid, result.solver)


def read_output(filename, config):
    """Read one output with the metadata needed by the HDF5 reader."""

    code_units = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])
    result = Rsim(config["par"])
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    result.fluid.eos = EOS(
        config["par"]['hydrodynamics']['eos_type'],
        result.par.hydrodynamics.gamma,
        code_units,
    )
    return result
