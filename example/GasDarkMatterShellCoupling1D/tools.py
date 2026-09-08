"""Initial-condition and dark-matter helpers for the coupled example."""

import numpy as np
import unyt
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import CodeUnits


def build_initial_condition(config):
    initial = config['initial_condition']
    runtime = config['par']
    code_units = CodeUnits.from_mapping(runtime['units']['CodeUnits'])
    grid_cells = int(runtime['mesh']['grid_cells'])
    result = Rsim(runtime)
    result.par.simulation.coordinate_system = 'spherical'
    result.par.simulation.box_size = float(initial['rmax'].to_value(code_units.length_unit))
    result.par.mesh.ghost_cells = 1
    boundary = np.linspace(initial['rmin'].to_value(code_units.length_unit), result.par.simulation.box_size, grid_cells + 1)
    coordinate = 0.75 * (boundary[1:]**4 - boundary[:-1]**4) / (boundary[1:]**3 - boundary[:-1]**3)
    result.mesh.boundary_proper_code = as_named_array(boundary)
    result.fluid.rho_proper_code = as_named_array((np.ones(grid_cells) * initial['gas_density']).to_value(code_units.density_unit))
    result.fluid.temp_proper_code = as_named_array((np.ones(grid_cells) * initial['gas_temperature']).to_value(code_units.temperature_unit))
    result.fluid.vel_proper_code = as_named_array(np.zeros(grid_cells))
    result.fluid.mu = as_named_array(np.ones(grid_cells) * initial['mu'])
    result.SetMesh()
    result.fluid.SetUpFluid(result.par, result.mesh)
    first, last = 1, grid_cells + 1
    result.mesh.boundary_proper_code = as_named_array(result.mesh.boundary_proper_code[first:last + 1])
    for field in ('rho_proper_code', 'vel_proper_code', 'temp_proper_code', 'mu'):
        setattr(result.fluid, field, as_named_array(getattr(result.fluid, field)[first:last]))
    result.par.mesh.ghost_cells = 0
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=coordinate,
        boundary_proper_code=result.mesh.boundary_proper_code,
        width_proper_code=np.diff(result.mesh.boundary_proper_code),
        area_proper_code=4.0 * np.pi * result.mesh.boundary_proper_code[:-1]**2,
        volume_proper_code=4.0 * np.pi / 3.0 * (result.mesh.boundary_proper_code[1:]**3 - result.mesh.boundary_proper_code[:-1]**3),
    )
    result.fluid.SetPressure()
    result.fluid.SetEnergyDensity()
    return result


def make_dark_matter(config):
    """Build dark-matter shells from the complete nested configuration."""
    initial_condition = config['initial_condition']
    par_config = config['par']
    code_units = CodeUnits.from_mapping(par_config['units']['CodeUnits'])
    count = int(initial_condition['dark_matter_shells'])
    radius = np.linspace(0.05, 0.95, count)
    velocity = np.asarray(radius) * float(
        initial_condition['dark_matter_velocity_scale']
    )
    angular_momentum = np.full(
        count, float(initial_condition['dark_matter_angular_momentum'])
    )
    return DarkMatterShells(
        radius=radius,
        velocity=velocity,
        mass=np.full(count, initial_condition['dark_matter_mass'] / count),
        angular_momentum=angular_momentum,
        softening=initial_condition['dark_matter_softening'],
        code_units=code_units,
    )
