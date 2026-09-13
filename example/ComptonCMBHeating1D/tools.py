"""Small IC helper for the fixed-density CMB Compton example."""

import numpy as np
import unyt

from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, quantity_to_value


def build_initial_condition(config):
    initial = config['initial_condition']

    code_units = config['_code_units']
    simulation = config["par"]['simulation']
    mesh = config["par"]['mesh']
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=code_units, ic_config=initial,
    )
    grid_cells = int(mesh['grid_cells'])
    boundary_proper_unyt = np.linspace(0.0, 1.0, grid_cells + 1) * initial['box_size_proper']
    density_proper_cgs_g_cm3_unyt = (
        np.ones(grid_cells) * initial['hydrogen_number_density'] * unyt.mp
    )
    writer.box_size = writer.radquantity(initial['box_size_proper'])
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(density_proper_cgs_g_cm3_unyt)
    writer.fluid.vel_radarray = writer.radarray(
        np.zeros(grid_cells) * code_units.velocity_unit
    )
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(grid_cells) * initial['temperature_proper']
    )
    writer.simulation.fluid.xHI = np.full(grid_cells, initial['xHI'])
    writer.fluid.mu = np.full(
        grid_cells, initial['mean_molecular_weight']
    )
    return writer
