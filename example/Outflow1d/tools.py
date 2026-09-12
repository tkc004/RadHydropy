"""Helper utilities for the cartesian outflow example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits


def build_initial_condition(config):
    initial = config['initial_condition']
    grid_cells = int(initial['grid_cells'])
    boundary_proper_unyt = np.linspace(0.0, 1.0, grid_cells + 1) * initial['box_size_proper']
    coordinate_proper_unyt = 0.5 * (
        boundary_proper_unyt[:-1] + boundary_proper_unyt[1:]
    )
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    writer = InitialConditionWriter(par_config=config['par'], code_units=code_units)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.mesh.x_radarray = writer.radarray(coordinate_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(
        np.ones(grid_cells) * initial['vel_proper']
    )
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(grid_cells) * initial['temperature_proper']
    )
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(grid_cells) * initial['rho_proper']
    )
    writer.simulation.fluid.mu = np.full(
        grid_cells, float(initial['mean_molecular_weight'])
    )
    return writer

def plot_snapshot(outfilename, config, **kwargs):

    rout = rio.loadhdf5(config, outfilename)
    code_units_obj = config['_code_units']
    first = int(rout.par.mesh.ghost_cells)
    last = first + int(rout.par.mesh.grid_cells)
    boundary_proper_code = rout.mesh.boundary_radarray.to(code_units_obj.length_unit).value
    x_proper_code = 0.5 * (boundary_proper_code[:-1] + boundary_proper_code[1:])
    rho_proper_code = rout.fluid.rho_radarray.to(code_units_obj.density_unit).value
    plt.plot(x_proper_code[first:last] * code_units_obj.length_unit,
             rho_proper_code[first:last] * code_units_obj.density_unit,
             **kwargs)
    plt.axvline(
        x=(rout.fluid.time_proper_code * code_units_obj.time_unit)
        * config["par"]['boundary']['vel_outflow_proper'],
        color=kwargs['color'],
        ls='dashed',
    )
