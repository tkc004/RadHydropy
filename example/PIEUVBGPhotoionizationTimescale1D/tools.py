"""Helpers for the HM12 PIE photoionization-timescale example."""

from pathlib import Path

import h5py
import numpy as np
from types import SimpleNamespace


def build_initial_condition(config):
    initial = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    result = SimpleNamespace()
    result.par = SimpleNamespace(
        time=np.ones(1) * initial['time'],
        units=SimpleNamespace(CodeUnits=config['_code_units']),
        simulation=SimpleNamespace(
            current_time=initial['time'], box_size=initial['boxsize'],
            coordinate_system=initial['coordsys'],
        ),
        mesh=SimpleNamespace(grid_cells=grid_cells, ghost_cells=0),
    )
    result.mesh = SimpleNamespace()
    result.fluid = SimpleNamespace()
    result.mesh.boundary = np.linspace(0.0, initial['boxsize'], grid_cells + 1)
    result.fluid.vel_code = np.zeros(grid_cells) * initial['vini']
    result.fluid.temp_code = np.ones(grid_cells) * initial['tempini']
    rho = float(initial['nHini']) * float(initial['proton_mass_g']) / float(initial['hydrogen_mass_fraction'])
    result.fluid.rho_code = np.ones(grid_cells) * rho
    result.fluid.mu = np.ones(grid_cells) * initial['muini']
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
                    "temperature_cgs_K": float(np.mean(data["Temperature"][interior])),
                    "density_cgs_g_cm3": float(np.mean(data["Density"][interior])),
                }
            )
    return history

