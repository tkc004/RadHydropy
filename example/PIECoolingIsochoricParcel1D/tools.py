"""Initial-condition helper for the isochoric PIE parcel benchmark."""

import numpy as np
import unyt
from radhydropy.initial_condition_writer import InitialConditionWriter


def build_initial_condition(config):
    initial = config['initial_condition']
    thermochemistry = config['par']['thermochemistry']
    code_units = config['_code_units']
    hydrogen_number_density_unyt = initial.get(
        'hydrogen_number_density', 1.0 / unyt.cm**3
    )
    temperature_proper_unyt = initial['temperature_proper']
    hydrogen_mass_fraction = float(thermochemistry['hydrogen_mass_fraction'])
    grid_cells = int(config['par']['mesh']['grid_cells'])
    box_size_proper_unyt = initial['box_size_proper']
    boundary_proper_unyt = np.linspace(
        0.0, 1.0, grid_cells + 1
    ) * box_size_proper_unyt
    writer = InitialConditionWriter(par_config=config['par'], code_units=code_units)
    writer.box_size = writer.radquantity(box_size_proper_unyt)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(np.zeros(grid_cells) * code_units.velocity_unit)
    writer.fluid.temp_radarray = writer.radarray(np.ones(grid_cells) * temperature_proper_unyt)
    rho_proper_unyt = np.ones(grid_cells) * hydrogen_number_density_unyt * unyt.mp / hydrogen_mass_fraction
    writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
    writer.simulation.fluid.mu = np.full(grid_cells, initial['mean_molecular_weight'])
    writer.simulation.par.simulation.time_proper_code = initial['time_proper'].to_value(code_units.time_unit)
    return writer
