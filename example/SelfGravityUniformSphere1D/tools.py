"""Helpers for the uniform-density spherical self-gravity diagnostic."""

import numpy as np
import unyt
from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS
import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, quantity_to_value


def spherical_cell_centers(boundary_proper_code):
    """Return volume-weighted centers for spherical cells."""
    inner = boundary_proper_code[:-1]
    outer = boundary_proper_code[1:]
    denominator = outer**3 - inner**3
    return 0.75 * (outer**4 - inner**4) / denominator


def uniform_sphere_acceleration(radius_proper_unyt, rho_proper_cgs_g_cm3_unyt):
    """Return the analytic interior field of a uniform-density sphere."""
    radius_proper_cgs_cm_unyt = radius_proper_unyt.to(unyt.cm)
    rho_proper_cgs_g_cm3_unyt = rho_proper_cgs_g_cm3_unyt.to(unyt.g / unyt.cm**3)
    return (
        -4.0 * np.pi / 3.0
        * (GRAVITATIONAL_CONSTANT_CGS * unyt.cm**3 / (unyt.g * unyt.s**2))
        * rho_proper_cgs_g_cm3_unyt
        * radius_proper_cgs_cm_unyt
    ).to(unyt.cm / unyt.s**2)


def build_initial_condition(config):
    code_unit_system = CodeUnits.from_mapping(
        config['par']['units']['CodeUnits']
    )
    initial_condition = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=code_unit_system,
        ic_config=initial_condition,
    )
    writer.simulation.par.simulation.coordinate_system = initial_condition['coordsys']
    boundary_proper_unyt = np.linspace(
        initial_condition['radius_inner_proper'],
        initial_condition['radius_outer_proper'],
        grid_cells + 1,
    )
    boundary_proper_code = quantity_to_value(
        boundary_proper_unyt, code_unit_system.length_unit
    )
    writer.box_size = writer.radquantity(initial_condition['box_size_proper'])
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.mesh.x_radarray = writer.radarray(
        spherical_cell_centers(boundary_proper_code) * code_unit_system.length_unit
    )
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(grid_cells) * initial_condition['rho_proper']
    )
    writer.fluid.vel_radarray = writer.radarray(
        np.zeros(grid_cells) * code_unit_system.velocity_unit
    )
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(grid_cells) * initial_condition['temperature_proper']
    )
    writer.simulation.fluid.mu = np.ones(grid_cells) * float(
        initial_condition['mean_molecular_weight']
    )
    return writer

def load_output_state(filename, config):
    return rio.loadhdf5(config, filename)
