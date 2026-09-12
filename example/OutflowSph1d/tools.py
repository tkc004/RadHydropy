"""Helper utilities for the spherical outflow example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits
from example.OutflowSph1d import outflow_sph_analytic as oa


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])
    grid_cells = int(initial['grid_cells'])
    boundary_proper_unyt = np.linspace(
        0.0, 1.0, grid_cells + 1
    ) * initial['box_size_proper'] + initial['radius_injection_proper']
    inner_proper_unyt = boundary_proper_unyt[:-1]
    outer_proper_unyt = boundary_proper_unyt[1:]
    coordinate_proper_unyt = 0.75 * (
        outer_proper_unyt**4 - inner_proper_unyt**4
    ) / (outer_proper_unyt**3 - inner_proper_unyt**3)
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=code_units
    )
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
    initial = config['initial_condition']

    rout = rio.loadhdf5(config, outfilename)
    code_units_obj = config['_code_units']
    boundary_proper_code = rout.mesh.boundary_radarray.to(
        code_units_obj.length_unit
    ).value
    first = int(rout.par.mesh.ghost_cells)
    last = first + int(rout.par.mesh.grid_cells)
    radius_proper_unyt = (
        0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])[first:last]
        * code_units_obj.length_unit
    )
    rho_num = rout.fluid.rho_radarray.to(
        config["par"]['boundary']['rho_outflow_proper'].units
    )[first:last]
    rho_ana_proper_unyt = oa.density_profile_proper_unyt(
        radius_proper_unyt,
        config["par"]['boundary']['rho_outflow_proper'],
        initial['radius_injection_proper'],
    )
    front_radius_proper_unyt = oa.front_position_proper_unyt(
        rout.fluid.time_proper_code * code_units_obj.time_unit,
        config["par"]['boundary']['vel_outflow_proper'],
    )
    radius_values = radius_proper_unyt.to_value(initial['box_size_proper'].units)
    rho_values = np.asarray(rho_num.to_value(config["par"]['boundary']['rho_outflow_proper'].units), dtype=float)
    rho_ana_values = np.asarray(rho_ana_proper_unyt.to_value(config["par"]['boundary']['rho_outflow_proper'].units), dtype=float)
    plt.plot(radius_values, rho_values, **kwargs)
    plt.plot(
        radius_values,
        rho_ana_values,
        ls='dashed',
        color='k',
    )
    plt.axvline(
        x=front_radius_proper_unyt.to_value(initial['box_size_proper'].units),
        color=kwargs['color'],
        ls='dashed',
    )
    plt.xlim(xmin=float(np.min(radius_values)), xmax=float(np.max(radius_values)))
    plt.yscale('log')
    plt.xlabel(r'Radius [cm]')
    plt.ylabel(r'$\rho$ [g/cm$^3$]')
