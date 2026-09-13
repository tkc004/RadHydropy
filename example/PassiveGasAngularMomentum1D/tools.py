"""Initial-condition builder for the passive gas angular-momentum example."""

import numpy as np

from radhydropy.initial_condition_writer import InitialConditionWriter


def build_initial_condition(config):
    """Build the active proper-coordinate state from nested configuration."""

    initial = config["initial_condition"]
    code_units = config["_code_units"]
    grid_cells = int(config["par"]["mesh"]["grid_cells"])
    box_size_proper_unyt = initial["box_size_proper"]
    boundary_proper_unyt = np.linspace(0.0, 1.0, grid_cells + 1) * box_size_proper_unyt
    coordinate_proper_unyt = 0.5 * (boundary_proper_unyt[:-1] + boundary_proper_unyt[1:])
    writer = InitialConditionWriter(
        par_config=config["par"], code_units=code_units,
        ic_config=initial,
    )
    writer.box_size = writer.radquantity(box_size_proper_unyt)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(np.ones(grid_cells) * initial["rho_proper"])
    writer.fluid.vel_radarray = writer.radarray(np.ones(grid_cells) * initial["vel_proper"])
    writer.fluid.temp_radarray = writer.radarray(np.ones(grid_cells) * initial["temperature_proper"])
    writer.fluid.mu = np.full(grid_cells, float(initial["mean_molecular_weight"]))
    if initial.get("include_angular_momentum", True):
        angular_momentum_unit = code_units.length_unit * code_units.velocity_unit
        coordinate_dimensionless = coordinate_proper_unyt / box_size_proper_unyt
        angular_momentum_unyt = (
            initial["angular_momentum_offset"]
            + initial["angular_momentum_amplitude"]
            * np.sin(2.0 * np.pi * coordinate_dimensionless)
        )
        writer.fluid.specific_angular_momentum_radarray = writer.radarray(
            angular_momentum_unyt,
        )
    return writer
