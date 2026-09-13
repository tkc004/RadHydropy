"""Helpers for the uniform HM12 PIE cooling example."""

import numpy as np
import unyt

from radhydropy.initial_condition_writer import InitialConditionWriter


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    box_size_proper_unyt = initial['box_size_proper']
    dx_proper_unyt = box_size_proper_unyt / grid_cells
    boundary_proper_unyt = np.linspace(dx_proper_unyt, box_size_proper_unyt + dx_proper_unyt, grid_cells + 1)
    hydrogen_number_density_cgs_cm3_unyt = initial['hydrogen_number_density']
    hydrogen_mass_fraction = float(config['par']['thermochemistry']['hydrogen_mass_fraction'])
    rho_proper_unyt = np.ones(grid_cells) * hydrogen_number_density_cgs_cm3_unyt * unyt.mp / hydrogen_mass_fraction
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=code_units, ic_config=initial,
    )
    writer.box_size = writer.radquantity(boundary_proper_unyt[-1])
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(np.zeros(grid_cells) * code_units.velocity_unit)
    writer.fluid.temp_radarray = writer.radarray(np.ones(grid_cells) * initial['temperature_proper'])
    writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
    writer.fluid.mu = np.full(grid_cells, initial['mean_molecular_weight'])
    return writer
