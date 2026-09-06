"""Helper utilities for the spherical outflow example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import code_quantity_to_cgs
import outflow_sph_analytic as oa


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    sim = Rsim(config['par'])
    grid_cells = int(initial['grid_cells'])
    sim.par.mesh.grid_cells = grid_cells
    start = initial['injection_radius'].to_value(code_units.length_unit)
    width = initial['box_size'].to_value(code_units.length_unit)
    sim.mesh.boundary_proper_code = as_named_array(np.linspace(start, start + width, grid_cells + 1))
    sim.fluid.vel_proper_code = as_named_array(np.full(
        grid_cells, initial['initial_velocity'].to_value(code_units.velocity_unit)
    ))
    sim.fluid.temp_proper_code = as_named_array(np.full(
        grid_cells, initial['initial_temperature'].to_value(code_units.temperature_unit)
    ))
    sim.fluid.rho_proper_code = as_named_array(np.full(
        grid_cells, initial['initial_density'].to_value(code_units.density_unit)
    ))
    sim.fluid.mu = as_named_array(np.full(grid_cells, initial['mean_molecular_weight']))
    sim.SetMesh()
    sim.fluid.SetUpFluid(sim.par, sim.mesh)
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
    first = int(sim.par.mesh.ghost_cells)
    last = first + grid_cells
    sim.mesh.boundary_proper_code = as_named_array(sim.mesh.boundary_proper_code[first:last + 1])
    for field in ('rho_proper_code', 'vel_proper_code', 'temp_proper_code', 'mu', 'Energy_code', 'InternalEnergy_code'):
        if hasattr(sim.fluid, field):
            setattr(sim.fluid, field, as_named_array(getattr(sim.fluid, field)[first:last]))
    sim.par.mesh.ghost_cells = 0
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        coordinate=sim.mesh.x_proper_code[first:last],
        boundary=sim.mesh.boundary_proper_code,
        width=sim.mesh.width_proper_code[first:last],
        area=sim.mesh.area_proper_code[first:last],
        volume=sim.mesh.volume_proper_code[first:last],
    )
    sim.fluid._refresh_runtime_state()
    return sim

def ReadandPlot(outfilename, config, **kwargs):
    initial = config['initial_condition']
    par_config = config['par']
    rout = Rsim(par_config)
    code_units_obj = config['_code_units']
    rout.par.units.CodeUnits = code_units_obj
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    boundary = np.asarray(rout.mesh.boundary_proper_code, dtype=float)
    first = int(rout.par.mesh.ghost_cells)
    last = first + int(rout.par.mesh.grid_cells)
    x_center = 0.5 * (boundary[1:] + boundary[:-1])[first:last] * code_units_obj.length_unit
    rho_num = code_quantity_to_cgs(rout.fluid.rho_proper_code[first:last], code_units_obj, 'density_cgs_g_cm3')
    rho_num = rho_num * (1.0 * par_config['boundary']['outflow_density'].units)
    rho_ana = oa.density_profile(
        x_center,
        par_config['boundary']['outflow_density'],
        initial['injection_radius'],
    )
    front = oa.front_position(
        rout.fluid.time_proper_code * code_units_obj.time_unit,
        par_config['boundary']['outflow_velocity'],
    )
    x_values = x_center.to_value(initial['box_size'].units)
    rho_values = np.asarray(rho_num.to_value(par_config['boundary']['outflow_density'].units), dtype=float)
    rho_ana_values = np.asarray(rho_ana.to_value(par_config['boundary']['outflow_density'].units), dtype=float)
    plt.plot(x_values, rho_values, **kwargs)
    plt.plot(
        x_values,
        rho_ana_values,
        ls='dashed',
        color='k',
    )
    plt.axvline(
        x=front.to_value(initial['box_size'].units),
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

