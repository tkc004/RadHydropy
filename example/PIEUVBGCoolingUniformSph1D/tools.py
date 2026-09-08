"""Helpers for the uniform HM12 PIE cooling example."""

import numpy as np
import unyt

from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import quantity_to_value
from basic_hydro_utils import finalize_initial_condition


def build_initial_condition(config):
    initial = config['initial_condition']
    par = config['par']
    grid_cells = int(par['mesh']['grid_cells'])
    result = Rsim(config["par"])
    code_units = result.par.units.CodeUnits
    result.par.simulation.time_proper_code = quantity_to_value(initial['time_proper'], code_units.time_unit)
    result.par.simulation.box_size_proper_code = quantity_to_value(initial['box_size_proper'], code_units.length_unit)
    result.par.simulation.coordinate_system = initial['coordsys']
    result.par.mesh.grid_cells = grid_cells
    result.par.mesh.ghost_cells = int(par['mesh'].get('ghost_cells', 0))
    box_size_proper_unyt = initial['box_size_proper']
    dx = box_size_proper_unyt / grid_cells
    boundary_proper_code = as_named_array(quantity_to_value(
        np.linspace(dx, box_size_proper_unyt + dx, grid_cells + 1), code_units.length_unit
    ))
    width = np.diff(boundary_proper_code)
    x_proper_code = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
    result.mesh.boundary_proper_code = boundary_proper_code
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=x_proper_code, boundary_proper_code=boundary_proper_code,
        width_proper_code=width, area_proper_code=np.ones(grid_cells), volume_proper_code=width,
    )
    vel_proper = initial.get('vini', 0.0 * unyt.cm / unyt.s)
    result.fluid.vel_proper_code = as_named_array(quantity_to_value(
        np.zeros(grid_cells) * vel_proper, code_units.velocity_unit
    ))
    result.fluid.temp_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * initial['temperature_proper'], code_units.temperature_unit
    ))
    hydrogen_number_density_cgs_cm3 = initial['hydrogen_number_density']
    proton_mass_g = float(initial.get('proton_mass_g', unyt.mp.to_value(unyt.g)))
    hydrogen_mass_fraction = float(initial.get(
        'hydrogen_mass_fraction', par['thermochemistry']['hydrogen_mass_fraction']
    ))
    if hasattr(hydrogen_number_density_cgs_cm3, 'to_value'):
        hydrogen_number_density_cgs_cm3 = hydrogen_number_density_cgs_cm3.to_value(1 / unyt.cm**3)
    rho_proper = float(hydrogen_number_density_cgs_cm3) * proton_mass_g / hydrogen_mass_fraction
    result.fluid.rho_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * rho_proper * unyt.g / unyt.cm**3, code_units.density_unit
    ))
    result.fluid.mu = np.ones(grid_cells) * initial['muini']
    result.fluid.time_proper_code = 0.0
    result.SetMesh()
    result.fluid.SetUpFluid(result.par, result.mesh)
    result.fluid.SetFluidTime(0.0)
    result.fluid.SetEnergyDensity()
    result.fluid._refresh_runtime_state()
    result.mesh._par = result.par
    result.solver.SetConserved(result.mesh, result.fluid, verbose=0)
    finalize_initial_condition(result, grid_cells)
    result.ConvertParametersToCodeUnits()
    return result
