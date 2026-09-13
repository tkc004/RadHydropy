"""Initial-condition and output helpers for the spherical shock example."""

import numpy as np

import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter


def build_initial_condition(config):
    initial = config["initial_condition"]
    writer = InitialConditionWriter(
        par_config=config["par"],
        code_units=config["_code_units"],
        ic_config=initial,
    )
    grid_cells = int(initial["grid_cells"])
    radius_inner_proper_unyt = initial["radius_inner_proper"]
    radius_outer_proper_unyt = initial["radius_outer_proper"]
    boundary_proper_unyt = np.linspace(0.0, 1.0, grid_cells + 1)
    boundary_proper_unyt = (
        radius_inner_proper_unyt
        + boundary_proper_unyt * (radius_outer_proper_unyt - radius_inner_proper_unyt)
    )
    writer.box_size = writer.radquantity(radius_outer_proper_unyt)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(grid_cells) * initial["rho_proper"]
    )
    writer.fluid.vel_radarray = writer.radarray(
        np.ones(grid_cells) * initial["vel_proper"]
    )
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(grid_cells) * initial["temperature_proper"]
    )
    writer.fluid.mu = np.full(
        grid_cells, float(initial["mean_molecular_weight"])
    )
    return writer


def read_output(filename, config):
    """Read one output with the metadata needed by the HDF5 reader."""

    return rio.loadhdf5(config, filename)
