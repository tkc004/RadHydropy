"""Helpers for the HM12 PIE photoionization-timescale example."""

from pathlib import Path

import h5py
import numpy as np
import unyt

from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import quantity_to_value


def build_initial_condition(config):
    initial = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    result = Rsim(config['par'])
    code_units = result.par.units.CodeUnits
    result.par.simulation.coordinate_system = initial['coordsys']
    result.par.simulation.time_code = quantity_to_value(initial['time'], code_units.time_unit)
    result.par.simulation.box_size = quantity_to_value(initial['boxsize'], code_units.length_unit)
    result.par.mesh.grid_cells = grid_cells
    boundary = as_named_array(quantity_to_value(
        np.linspace(0.0 * initial['boxsize'], initial['boxsize'], grid_cells + 1),
        code_units.length_unit,
    ))
    result.mesh.boundary_proper_code = boundary
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, coordinate=0.5 * (boundary[1:] + boundary[:-1]),
        boundary=boundary, width=np.diff(boundary), area=np.ones(grid_cells),
        volume=np.diff(boundary),
    )
    result.fluid.vel_proper_code = as_named_array(quantity_to_value(
        np.zeros(grid_cells) * initial['vini'], code_units.velocity_unit
    ))
    result.fluid.temp_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * initial['tempini'], code_units.temperature_unit
    ))
    rho = float(initial['nHini']) * float(initial['proton_mass_g']) / float(initial['hydrogen_mass_fraction'])
    result.fluid.rho_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * rho * unyt.g / unyt.cm**3, code_units.density_unit
    ))
    result.fluid.mu = np.ones(grid_cells) * initial['muini']
    result.fluid.time_proper_code = 0.0
    result.SetMesh()
    result.fluid.SetUpFluid(result.par, result.mesh)
    result.fluid.SetFluidTime(0.0)
    result.fluid.SetEnergyDensity()
    result.mesh._par = result.par
    result.solver.SetConserved(result.mesh, result.fluid, verbose=0)
    result.ConvertParametersToCodeUnits()
    return result

def clean_outputs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in output_dir.glob("Output_*.hdf5"):
        filename.unlink()
    initial = output_dir / "InitialCondition.hdf5"
    if initial.exists():
        initial.unlink()


def load_history(output_dir):
    history = []
    for filename in sorted(output_dir.glob("Output_*.hdf5")):
        with h5py.File(filename, "r") as handle:
            header = handle["Header"]
            data = handle["Data"]
            noghost = int(header.attrs.get("GhostCells", 0))
            nogrid = int(header.attrs["GridCells"])
            interior = slice(noghost, noghost + nogrid)
            history.append(
                {
                    "filename": Path(filename),
                    "time_s": float(header.attrs.get("Time", 0.0)),
                    "temperature_cgs_K": float(np.mean(data["temp_proper_code"][interior])),
                    "density_cgs_g_cm3": float(np.mean(data["rho_proper_code"][interior])),
                }
            )
    return history
