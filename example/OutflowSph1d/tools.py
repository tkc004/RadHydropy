"""Helper utilities for the spherical outflow example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

import radhydropy.io as rio
from radhydropy.units import CodeUnits, code_quantity_to_cgs
import outflow_sph_analytic as oa


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def build_initial_condition(config):
    icparams = config['initial_condition']
    code_units = config['_code_units']
    sim = SimpleNamespace()
    sim.par = Par()
    sim.mesh = Mesh()
    sim.fluid = Fluid()
    sim.par.units = SimpleNamespace(CodeUnits=code_units)
    if code_units is not None:
        sim.par.unit_system = code_units.unit_system

    grid_cells = icparams['grid_cells']
    box_size = icparams['box_size'] * np.ones(1)
    sim.par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=grid_cells)
    sim.par.simulation = SimpleNamespace(
        coordinate_system=icparams['coordinate_system'],
        current_time=icparams['current_time'] * np.ones(1),
        box_size=box_size,
    )

    sim.mesh.boundary = np.linspace(
        icparams['injection_radius'],
        icparams['injection_radius'] + box_size[0],
        grid_cells + 1,
    )
    sim.fluid.vel_code = icparams['initial_velocity'] * np.ones(grid_cells)
    sim.fluid.temp_code = icparams['initial_temperature'] * np.ones(grid_cells)
    sim.fluid.rho_code = icparams['initial_density'] * np.ones(grid_cells)
    sim.fluid.mu = icparams['mean_molecular_weight'] * np.ones(grid_cells)


    return sim

def ReadandPlot(outfilename, config, **kwargs):
    icparams = config['initial_condition']
    runparams = config['par']
    rout = build_initial_condition(config)
    code_units_obj = config['_code_units']
    rout.par.units.CodeUnits = code_units_obj
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    boundary = np.asarray(rout.mesh.boundary, dtype=float)
    x_center = 0.5 * (boundary[1:] + boundary[:-1]) * code_units_obj.length_unit
    rho_num = code_quantity_to_cgs(rout.fluid.rho_code, code_units_obj, 'density_cgs_g_cm3')
    rho_num = rho_num * (1.0 * runparams['boundary']['outflow_density'].units)
    rho_ana = oa.density_profile(
        x_center,
        runparams['boundary']['outflow_density'],
        icparams['injection_radius'],
    )
    front = oa.front_position(
        rout.par.simulation.current_time * code_units_obj.time_unit,
        runparams['boundary']['outflow_velocity'],
    )
    x_values = x_center.to_value(icparams['box_size'].units)
    rho_values = np.asarray(rho_num.to_value(runparams['boundary']['outflow_density'].units), dtype=float)
    rho_ana_values = np.asarray(rho_ana.to_value(runparams['boundary']['outflow_density'].units), dtype=float)
    plt.plot(x_values, rho_values, **kwargs)
    plt.plot(
        x_values,
        rho_ana_values,
        ls='dashed',
        color='k',
    )
    plt.axvline(
        x=front.to_value(icparams['box_size'].units),
        color=kwargs['color'],
        ls='dashed',
    )
    plt.xlim(
        xmin=float(np.min(x_values)),
        xmax=float(np.max(x_values)),
    )
    plt.yscale('log')
    plt.xlabel(r'Radius [cm]')
    plt.ylabel(r'$\rho$ [g/cm$^3$]')





