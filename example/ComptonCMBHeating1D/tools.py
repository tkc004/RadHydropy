"""Small IC helper for the fixed-density CMB Compton example."""

import numpy as np
import unyt

from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import CodeUnits


def build_initial_condition(config):
    initial = config['initial_condition']

    simulation = config["par"]['simulation']
    mesh = config["par"]['mesh']
    code_units = config['_code_units']
    result = Rsim(config['par'])
    result.par.mesh.grid_cells = int(mesh['grid_cells'])
    result.par.simulation.coordinate_system = simulation['coordinate_system']
    result.par.simulation.time_proper_code = initial['time_proper']
    result.par.simulation.box_size_proper_code = initial['box_size_proper']
    result.mesh.boundary_proper_code = np.linspace(
        0.0, 1.0, result.par.mesh.grid_cells + 1
    ) * initial['box_size_proper']
    density_proper_cgs_g_cm3_unyt = (
        np.ones(result.par.mesh.grid_cells) * initial['hydrogen_number_density'] * unyt.mp
    )
    result.fluid.rho_proper_code = density_proper_cgs_g_cm3_unyt
    result.fluid.vel_proper_code = np.zeros(result.par.mesh.grid_cells, dtype=float)
    result.fluid.temp_proper_code = np.ones(result.par.mesh.grid_cells) * initial['temperature_proper']
    result.fluid.xHI = np.ones(result.par.mesh.grid_cells) * initial['xHI']
    result.fluid.mu = np.ones(result.par.mesh.grid_cells) * initial['mean_molecular_weight']
    result.SetMesh()
    result.fluid.SetUpFluid(result.par, result.mesh)
    result.solver.SetConserved(result.mesh, result.fluid, verbose=0)
    first = int(result.par.mesh.ghost_cells)
    last = first + result.par.mesh.grid_cells
    result.mesh.boundary_proper_code = as_named_array(
        result.mesh.boundary_proper_code[first:last + 1]
    )
    for field in (
        'rho_proper_code', 'vel_proper_code', 'temp_proper_code', 'xHI', 'mu',
        'Mass_code', 'Mom_code', 'Energy_code', 'InternalEnergy_code',
    ):
        if hasattr(result.fluid, field):
            setattr(result.fluid, field, as_named_array(
                getattr(result.fluid, field)[first:last]
            ))
    result.par.mesh.ghost_cells = 0
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=result.mesh.x_proper_code[first:last],
        boundary_proper_code=result.mesh.boundary_proper_code,
        width_proper_code=result.mesh.width_proper_code[first:last],
        area_proper_code=result.mesh.area_proper_code[first:last],
        volume_proper_code=result.mesh.volume_proper_code[first:last],
    )
    result.fluid._refresh_runtime_state()
    return Rsim.FromComponents(result.par, result.mesh, result.fluid, result.solver)
