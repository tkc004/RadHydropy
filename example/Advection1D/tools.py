"""Helper utilities for the cartesian advection example."""

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
    box_size_code = quantity_to_value(initial['box_size_proper'], code_units.length_unit)
    boundary_proper_code = np.linspace(0.0, box_size_code, grid_cells + 1)
    coordinate_proper_code = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])

    rho_proper_code = np.full(grid_cells, quantity_to_value(initial['initial_density'], code_units.density_unit))
    sim.fluid.vel_proper_code = as_named_array(np.full(grid_cells, quantity_to_value(initial['initial_velocity'], code_units.velocity_unit)))
    sim.fluid.temp_proper_code = as_named_array(np.full(grid_cells, quantity_to_value(initial['initial_temperature'], code_units.temperature_unit)))
    rho_proper_code[
        np.logical_or(
            coordinate_proper_code < 0.25 * box_size_code,
            coordinate_proper_code > 0.75 * box_size_code,
        )
    ] *= 0.5
    sim.fluid.rho_proper_code = as_named_array(rho_proper_code)
    sim.fluid.mu = as_named_array(np.full(grid_cells, initial['mean_molecular_weight']))
    sim.mesh.boundary_proper_code = as_named_array(boundary_proper_code)
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
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(PROPER_RUNTIME_FIELDS,
        x_proper_code=sim.mesh.x_proper_code[first:last], boundary_proper_code=sim.mesh.boundary_proper_code,
        width_proper_code=sim.mesh.width_proper_code[first:last], area_proper_code=sim.mesh.area_proper_code[first:last],
        volume_proper_code=sim.mesh.volume_proper_code[first:last])
    sim.fluid._refresh_runtime_state()


    return sim

def ReadandPlot(outfilename, config, **kwargs):
    initial = config['initial_condition']
    rout = Rsim(config['par'])
    code_units_obj = config['_code_units']
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    first = int(rout.par.mesh.ghost_cells)
    last = first + int(rout.par.mesh.grid_cells)
    x_proper_code = 0.5 * (rout.mesh.boundary_proper_code[:-1] + rout.mesh.boundary_proper_code[1:])
    x_physical = x_proper_code[first:last]
    box_size = quantity_to_value(initial['box_size_proper'], code_units_obj.length_unit)
    velocity = quantity_to_value(initial['initial_velocity'], code_units_obj.velocity_unit)
    time_proper_code = float(np.asarray(rout.fluid.time_proper_code).flat[0])
    launch = np.mod(x_physical - velocity * time_proper_code, box_size)
    high_density = quantity_to_value(initial['initial_density'], code_units_obj.density_unit)
    analytic_density = np.where(
        (launch >= 0.25 * box_size) & (launch <= 0.75 * box_size),
        high_density,
        0.5 * high_density,
    )
    plt.plot(x_proper_code[first:last] * code_units_obj.length_unit,
             rout.fluid.rho_proper_code[first:last] * code_units_obj.density_unit,
             **kwargs)
    plt.plot(
        x_physical * code_units_obj.length_unit,
        analytic_density * code_units_obj.density_unit,
        color=kwargs.get('color'),
        linestyle='--',
        label='analytic',
    )
