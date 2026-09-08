"""Helper utilities for the cartesian outflow example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import quantity_to_value


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    sim = Rsim(config['par'])
    grid_cells = int(initial['grid_cells'])
    sim.par.mesh.grid_cells = grid_cells
    sim.mesh.boundary_proper_code = as_named_array(np.linspace(
        0.0, quantity_to_value(initial['box_size'], code_units.length_unit), grid_cells + 1
    ))
    sim.fluid.vel_proper_code = as_named_array(np.full(
        grid_cells, quantity_to_value(initial['initial_velocity'], code_units.velocity_unit)
    ))
    sim.fluid.temp_proper_code = as_named_array(np.full(
        grid_cells, quantity_to_value(initial['initial_temperature'], code_units.temperature_unit)
    ))
    sim.fluid.rho_proper_code = as_named_array(np.full(
        grid_cells, quantity_to_value(initial['initial_density'], code_units.density_unit)
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
    par_config = config['par']
    rout = Rsim(par_config)
    code_units_obj = config['_code_units']
    rout.par.units.CodeUnits = code_units_obj
    rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    first = int(rout.par.mesh.ghost_cells)
    last = first + int(rout.par.mesh.grid_cells)
    boundary_proper_code = rout.mesh.boundary_proper_code
    x_proper_code = 0.5 * (boundary_proper_code[:-1] + boundary_proper_code[1:])
    plt.plot(x_proper_code[first:last] * code_units_obj.length_unit,
             rout.fluid.rho_proper_code[first:last] * code_units_obj.density_unit,
             **kwargs)
    plt.axvline(
        x=(rout.fluid.time_proper_code * code_units_obj.time_unit)
        * par_config['boundary']['outflow_velocity'],
        color=kwargs['color'],
        ls='dashed',
    )


