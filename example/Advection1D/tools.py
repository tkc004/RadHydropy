"""Helper utilities for the cartesian advection example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import quantity_to_value


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    grid_cells = int(initial['grid_cells'])
    box_size_proper_unyt = initial['box_size_proper']
    boundary_proper_unyt = np.linspace(0.0, 1.0, grid_cells + 1) * box_size_proper_unyt
    coordinate_proper_unyt = 0.5 * (boundary_proper_unyt[1:] + boundary_proper_unyt[:-1])
    rho_proper_unyt = np.ones(grid_cells) * initial['rho_proper']
    rho_proper_unyt[
        np.logical_or(
            coordinate_proper_unyt < 0.25 * box_size_proper_unyt,
            coordinate_proper_unyt > 0.75 * box_size_proper_unyt,
        )
    ] *= 0.5
    writer = InitialConditionWriter(par_config=config['par'], code_units=code_units)
    writer.box_size = writer.radquantity(box_size_proper_unyt)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(np.ones(grid_cells) * initial['vel_proper'])
    writer.fluid.temp_radarray = writer.radarray(np.ones(grid_cells) * initial['temperature_proper'])
    writer.simulation.fluid.mu = np.full(grid_cells, float(initial['mean_molecular_weight']))
    return writer

def plot_snapshot(outfilename, config, **kwargs):
    initial = config['initial_condition']
    rout = rio.loadhdf5(config, outfilename)
    code_units_obj = config['_code_units']
    first = int(rout.par.mesh.ghost_cells)
    last = first + int(rout.par.mesh.grid_cells)
    x_proper_code = 0.5 * (rout.mesh.boundary_radarray.to_value(code_units_obj.length_unit)[:-1] + rout.mesh.boundary_radarray.to_value(code_units_obj.length_unit)[1:])
    x_active_proper_code = x_proper_code[first:last]
    box_size_proper_code = quantity_to_value(initial['box_size_proper'], code_units_obj.length_unit)
    vel_proper_code = quantity_to_value(initial['vel_proper'], code_units_obj.velocity_unit)
    time_proper_code = float(np.asarray(rout.fluid.time_proper_code).flat[0])
    launch_proper_code = np.mod(
        x_active_proper_code - vel_proper_code * time_proper_code,
        box_size_proper_code,
    )
    rho_high_proper_code = quantity_to_value(initial['rho_proper'], code_units_obj.density_unit)
    rho_analytic_proper_code = np.where(
        (launch_proper_code >= 0.25 * box_size_proper_code)
        & (launch_proper_code <= 0.75 * box_size_proper_code),
        rho_high_proper_code,
        0.5 * rho_high_proper_code,
    )
    rho_snapshot_proper_code = rout.fluid.rho_radarray.to_value(code_units_obj.density_unit)
    plt.plot(x_proper_code[first:last] * code_units_obj.length_unit,
             rho_snapshot_proper_code[first:last] * code_units_obj.density_unit,
             **kwargs)
    plt.plot(
        x_active_proper_code * code_units_obj.length_unit,
        rho_analytic_proper_code * code_units_obj.density_unit,
        color=kwargs.get('color'),
        linestyle='--',
        label='analytic',
    )
