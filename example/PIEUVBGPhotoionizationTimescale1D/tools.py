"""Helpers for the HM12 PIE photoionization-timescale example."""

from pathlib import Path

import numpy as np
import unyt

import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter


def build_initial_condition(config):
    initial = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    code_units = config['_code_units']
    boundary_proper_unyt = np.linspace(0.0, 1.0, grid_cells + 1) * initial['box_size_proper']
    rho_proper_unyt = np.ones(grid_cells) * initial['hydrogen_number_density'] * initial['proton_mass'] / initial['hydrogen_mass_fraction']
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=code_units, ic_config=initial,
    )
    writer.box_size = writer.radquantity(initial['box_size_proper'])
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(np.zeros(grid_cells) * initial['vel_proper'])
    writer.fluid.temp_radarray = writer.radarray(np.ones(grid_cells) * initial['temperature_proper'])
    writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
    writer.fluid.ngamma_radarray = writer.radarray(
        np.zeros(grid_cells) / unyt.cm**3
    )
    writer.simulation.fluid.xHI = np.ones(grid_cells)
    writer.simulation.fluid.mu = np.full(grid_cells, initial['mean_molecular_weight'])
    return writer

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
        snapshot = rio.loadhdf5(config, str(filename))
        first = int(snapshot.par.mesh.ghost_cells)
        last = first + int(snapshot.par.mesh.grid_cells)
        code_units = config['_code_units']
        history.append(
            {
                "filename": Path(filename),
                "time_proper_cgs_s": float(np.asarray(snapshot.fluid.time_proper_code).flat[0])
                * float(code_units.time_unit.to_value(unyt.s)),
                "temperature_proper_cgs_K": float(np.mean(
                    snapshot.fluid.temp_radarray.to(unyt.K).value[first:last]
                )),
                "rho_proper_cgs_g_cm3": float(np.mean(
                    snapshot.fluid.rho_radarray.to(unyt.g / unyt.cm**3).value[first:last]
                )),
            }
        )
    return history
