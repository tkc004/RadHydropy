"""Initial-condition helper for the isochoric PIE parcel benchmark."""

import numpy as np
import unyt
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import quantity_to_value
from basic_hydro_utils import finalize_initial_condition


def build_initial_condition(config):
    initial = config['initial_condition']
    thermochemistry = config['par']['thermochemistry']
    code_units = config['_code_units']
    density_cgs_cm3 = initial.get('hydrogen_density_cgs_cm3', 1.0)
    temperature_unyt = initial.get('temperature_unyt', initial['initial_temperature'])
    hydrogen_mass_fraction = float(thermochemistry['hydrogen_mass_fraction'])
    grid_cells = int(config['par']['mesh']['grid_cells'])
    result = Rsim(config['par'])
    result.par.simulation.coordinate_system = initial['coordinate_system']
    result.par.simulation.time_proper_code = quantity_to_value(initial['current_time'], code_units.time_unit)
    result.par.simulation.box_size = quantity_to_value(initial['box_size'], code_units.length_unit)
    boxsize_code = result.par.simulation.box_size
    dx_code = boxsize_code / grid_cells
    boundary_code = as_named_array(np.linspace(dx_code, boxsize_code + dx_code, grid_cells + 1))
    width_code = np.diff(boundary_code)
    volume_code = 4.0 * np.pi / 3.0 * (boundary_code[1:] ** 3 - boundary_code[:-1] ** 3)
    coordinate_code = 0.75 * (boundary_code[1:] ** 4 - boundary_code[:-1] ** 4) / (boundary_code[1:] ** 3 - boundary_code[:-1] ** 3)
    area_code = 4.0 * np.pi * boundary_code[:-1] ** 2
    result.mesh.boundary_proper_code = boundary_code
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, coordinate=coordinate_code, boundary=boundary_code,
        width=width_code, area=area_code, volume=volume_code,
    )
    result.fluid.vel_proper_code = as_named_array(np.zeros(grid_cells, dtype=float))
    result.fluid.temp_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * temperature_unyt, code_units.temperature_unit
    ))
    rho_cgs_g_cm3 = density_cgs_cm3 * unyt.mp.to_value(unyt.g) / hydrogen_mass_fraction
    result.fluid.rho_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * rho_cgs_g_cm3 * unyt.g / unyt.cm**3,
        code_units.density_unit,
    ))
    result.fluid.mu = np.ones(grid_cells) * initial['mean_molecular_weight']
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
