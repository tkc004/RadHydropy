"""Helpers for the uniform HM12 PIE cooling example."""

import numpy as np
import unyt

from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import quantity_to_value


def build_initial_condition(config):
    initial = config['initial_condition']
    par = config['par']
    grid_cells = int(par['mesh']['grid_cells'])
    result = Rsim(par)
    code_units = result.par.units.CodeUnits
    result.par.simulation.time_code = quantity_to_value(initial['time'], code_units.time_unit)
    result.par.simulation.box_size = quantity_to_value(initial['boxsize'], code_units.length_unit)
    result.par.simulation.coordinate_system = initial['coordsys']
    result.par.mesh.grid_cells = grid_cells
    result.par.mesh.ghost_cells = int(par['mesh'].get('ghost_cells', 0))
    boxsize = initial['boxsize']
    dx = boxsize / grid_cells
    boundary = as_named_array(quantity_to_value(
        np.linspace(dx, boxsize + dx, grid_cells + 1), code_units.length_unit
    ))
    width = np.diff(boundary)
    coordinate = 0.5 * (boundary[1:] + boundary[:-1])
    result.mesh.boundary_proper_code = boundary
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, coordinate=coordinate, boundary=boundary,
        width=width, area=np.ones(grid_cells), volume=width,
    )
    result.fluid.vel_proper_code = as_named_array(quantity_to_value(
        np.zeros(grid_cells) * initial['vini'], code_units.velocity_unit
    ))
    result.fluid.temp_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * initial['tempini'], code_units.temperature_unit
    ))
    rho = config['initial_condition']['hydrogen_density_cgs_cm3'] * float(initial['proton_mass_g']) / float(initial['hydrogen_mass_fraction'])
    result.fluid.rho_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * rho * unyt.g / unyt.cm**3, code_units.density_unit
    ))
    result.fluid.mu = np.ones(grid_cells) * initial['muini']
    result.fluid.time_proper_code = 0.0
    result.SetMesh()
    result.fluid.SetUpFluid(result.par, result.mesh)
    result.fluid.SetFluidTime(0.0)
    result.fluid.SetEnergyDensity()
    result.fluid._refresh_runtime_state()
    result.mesh._par = result.par
    result.solver.SetConserved(result.mesh, result.fluid, verbose=0)
    result.ConvertParametersToCodeUnits()
    return result
