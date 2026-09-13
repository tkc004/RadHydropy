"""Initial-condition and dark-matter helpers for the coupled example."""

import numpy as np
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits, quantity_to_value
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.cosmology_context import CosmologyContext


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])
    grid_cells = int(config["par"]['mesh']['grid_cells'])
    radius_inner_proper_code = quantity_to_value(
        initial['radius_inner_proper'], code_units.length_unit
    )
    radius_outer_proper_code = quantity_to_value(
        initial['radius_outer_proper'], code_units.length_unit
    )
    boundary_proper_code = np.linspace(
        radius_inner_proper_code, radius_outer_proper_code, grid_cells + 1
    )
    x_proper_code = 0.75 * (boundary_proper_code[1:]**4 - boundary_proper_code[:-1]**4) / (boundary_proper_code[1:]**3 - boundary_proper_code[:-1]**3)
    boundary_proper_unyt = boundary_proper_code * code_units.length_unit
    writer = InitialConditionWriter(par_config=config['par'], code_units=code_units, ic_config=config["initial_condition"])
    writer.simulation.par.cosmology_context = CosmologyContext(
        gamma=1.000001, cosmology='proper'
    )
    writer.box_size = writer.radquantity(boundary_proper_unyt[-1])
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.mesh.x_radarray = writer.radarray(x_proper_code * code_units.length_unit)
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(grid_cells) * initial['rho_proper']
    )
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(grid_cells) * initial['temperature_proper']
    )
    writer.fluid.vel_radarray = writer.radarray(
        np.zeros(grid_cells) * code_units.velocity_unit
    )
    writer.simulation.fluid.mu = np.ones(grid_cells) * float(initial['mu'])
    return writer


def make_dark_matter(config):
    """Build dark-matter shells from the complete nested configuration."""
    initial_condition = config['initial_condition']

    code_units = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])
    count = int(initial_condition['dark_matter_shells'])
    radius_inner_proper_code = quantity_to_value(
        initial_condition['radius_inner_proper'], code_units.length_unit
    )
    radius_outer_proper_code = quantity_to_value(
        initial_condition['radius_outer_proper'], code_units.length_unit
    )
    radius_proper_code = np.linspace(
        radius_inner_proper_code, radius_outer_proper_code, count
    )
    vel_radial_proper_code = np.asarray(radius_proper_code) * float(
        initial_condition['dark_matter_velocity_scale_dimensionless']
    )
    angular_momentum = np.full(
        count, float(initial_condition['dark_matter_angular_momentum_dimensionless'])
    )
    mass_code = quantity_to_value(
        initial_condition['dark_matter_mass'], code_units.mass_unit
    )
    softening_proper_code = quantity_to_value(
        initial_condition['dark_matter_softening'], code_units.length_unit
    )
    return DarkMatterShells(
        radius_proper_code,
        vel_radial_proper_code,
        np.full(count, mass_code / count),
        angular_momentum=angular_momentum,
        softening=softening_proper_code,
        code_units=code_units,
    )
