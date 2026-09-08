"""Helper utilities for the spherical inflow example."""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import CodeUnits
import inflow_sph_analytic as ia


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    sim = Rsim(config['par'])
    grid_cells = int(initial['grid_cells'])
    sim.par.mesh.grid_cells = grid_cells
    box_size_code = float(initial['box_size_proper'].to_value(code_units.length_unit))
    sim.mesh.boundary_proper_code = as_named_array(np.linspace(
        0.0, box_size_code, grid_cells + 1
    ))
    sim.fluid.vel_proper_code = as_named_array(np.full(
        grid_cells, initial['vel_proper'].to_value(code_units.velocity_unit)
    ))
    sim.fluid.temp_proper_code = as_named_array(np.full(
        grid_cells, initial['temperature_proper'].to_value(code_units.temperature_unit)
    ))
    sim.fluid.rho_proper_code = as_named_array(np.full(
        grid_cells, initial['rho_proper'].to_value(code_units.density_unit)
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
        x_proper_code=sim.mesh.x_proper_code[first:last],
        boundary_proper_code=sim.mesh.boundary_proper_code,
        width_proper_code=sim.mesh.width_proper_code[first:last],
        area_proper_code=sim.mesh.area_proper_code[first:last],
        volume_proper_code=sim.mesh.volume_proper_code[first:last],
    )
    sim.fluid._refresh_runtime_state()
    return sim

def ReadandPlot(outfilename, config, **kwargs):
    initial = config['initial_condition']

    rout = Rsim(config["par"])
    code_units_obj = config['_code_units']
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    time = rout.fluid.time_proper_code * code_units_obj.time_unit
    first = int(rout.par.mesh.ghost_cells)
    last = first + int(rout.par.mesh.grid_cells)
    boundary_proper_code = rout.mesh.boundary_proper_code
    x_proper_code = 0.5 * (boundary_proper_code[:-1] + boundary_proper_code[1:])
    plt.plot(x_proper_code[first:last] * code_units_obj.length_unit,
             rout.fluid.rho_proper_code[first:last] * code_units_obj.density_unit,
             **kwargs)
    plt.ylim(ymax=10.1)
    plt.axvline(
        x=ia.front_position(
            initial['box_size_proper'],
            time,
            config["par"]['boundary']['inflow_velocity'],
        ),
        color=kwargs['color'],
        ls='dashed',
    )
    rhoana = ia.density_profile(
        x_proper_code[first:last] * code_units_obj.length_unit,
        config["par"]['boundary']['inflow_density'],
        initial['box_size_proper'],
    )
    plt.plot(x_proper_code[first:last] * code_units_obj.length_unit, rhoana, ls='dashed', color='k')
