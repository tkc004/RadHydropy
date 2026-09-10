"""Helpers for the HM12 PIE photoionization-timescale example."""

from pathlib import Path

import numpy as np
import unyt

from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
import radhydropy.io as rio
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import quantity_to_value
from basic_hydro_utils import finalize_initial_condition


def build_initial_condition(config):
    initial = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    result = Rsim(config['par'])
    code_units = result.par.units.CodeUnits
    result.par.simulation.coordinate_system = initial['coordsys']
    result.par.simulation.time_proper_code = quantity_to_value(initial['time_proper'], code_units.time_unit)
    result.par.simulation.box_size_proper_code = quantity_to_value(initial['box_size_proper'], code_units.length_unit)
    result.par.mesh.grid_cells = grid_cells
    boundary_proper_code = as_named_array(quantity_to_value(
        np.linspace(0.0 * initial['box_size_proper'], initial['box_size_proper'], grid_cells + 1),
        code_units.length_unit,
    ))
    result.mesh.boundary_proper_code = boundary_proper_code
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1]),
        boundary_proper_code=boundary_proper_code, width_proper_code=np.diff(boundary_proper_code), area_proper_code=np.ones(grid_cells),
        volume_proper_code=np.diff(boundary_proper_code),
    )
    vel_proper_unyt = np.zeros(grid_cells) * initial['vel_proper']
    result.fluid.vel_proper_code = as_named_array(quantity_to_value(
        vel_proper_unyt, code_units.velocity_unit
    ))
    result.fluid.temp_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * initial['temperature_proper'], code_units.temperature_unit
    ))
    rho_proper_cgs_g_cm3 = (
        float(initial['hydrogen_number_density'].to_value(1 / unyt.cm**3))
        * float(initial['proton_mass'].to_value(unyt.g))
        / float(initial['hydrogen_mass_fraction'])
    )
    result.fluid.rho_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * rho_proper_cgs_g_cm3 * unyt.g / unyt.cm**3,
        code_units.density_unit,
    ))
    result.fluid.mu = np.ones(grid_cells) * initial['mean_molecular_weight']
    result.fluid.time_proper_code = 0.0
    result.SetMesh()
    result.fluid.SetUpFluid(result.par, result.mesh)
    result.fluid.SetFluidTime(0.0)
    result.fluid.SetEnergyDensity()
    result.mesh._par = result.par
    result.solver.SetConserved(result.mesh, result.fluid, verbose=0)
    finalize_initial_condition(result, grid_cells, extra_fields=('xHI', 'ngamma_code'))
    result.ConvertParametersToCodeUnits()
    return result

def clean_outputs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in output_dir.glob("Output_*.hdf5"):
        filename.unlink()
    initial = output_dir / "InitialCondition.hdf5"
    if initial.exists():
        initial.unlink()


def load_history(output_dir, config):
    history = []
    for filename in sorted(output_dir.glob("Output_*.hdf5")):
        snapshot = Rsim(config["par"])
        rio.readhdf5(snapshot.par, snapshot.mesh, snapshot.fluid, str(filename))
        first = int(snapshot.par.mesh.ghost_cells)
        last = first + int(snapshot.par.mesh.grid_cells)
        code_units = snapshot.par.units.CodeUnits
        history.append(
            {
                "filename": Path(filename),
                "time_proper_cgs_s": float(np.asarray(snapshot.fluid.time_proper_code).flat[0])
                * float(code_units.time_unit.to_value(unyt.s)),
                "temperature_proper_cgs_K": float(np.mean(
                    snapshot.fluid.temp_proper_code[first:last]
                )) * float(code_units.temperature_unit.to_value(unyt.K)),
                "rho_proper_cgs_g_cm3": float(np.mean(
                    snapshot.fluid.rho_proper_code[first:last]
                )) * float(code_units.density_unit.to_value(unyt.g / unyt.cm**3)),
            }
        )
    return history
