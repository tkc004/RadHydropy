"""Helpers for the HM12 PIE photoionization-timescale example."""

from pathlib import Path

import h5py
import numpy as np


class Simwrap:
    """Build the uniform IC object consumed by ``radhydropy.io``."""

    def __init__(self, config, code_units):
        from types import SimpleNamespace
        initial_mapping = config['initial_condition']
        grid_cells = int(config['par']['mesh']['grid_cells'])

        boxsize = np.ones(1) * initial_mapping["boxsize"]
        self.par = SimpleNamespace(time=np.ones(1) * initial_mapping["time"])
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.simulation = SimpleNamespace(
            current_time=initial_mapping["time"], box_size=initial_mapping["boxsize"],
            coordinate_system=initial_mapping["coordsys"],
        )
        self.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=0)
        self.mesh = SimpleNamespace()
        self.fluid = SimpleNamespace()
        self.mesh.boundary = np.linspace(
            0.0 * boxsize[0], boxsize[0], grid_cells + 1,
        )
        self.fluid.vel_code = np.zeros(grid_cells) * initial_mapping["vini"]
        self.fluid.temp_code = np.ones(grid_cells) * initial_mapping["tempini"]
        rho = (
            float(initial_mapping["nHini"])
            * float(initial_mapping["proton_mass_g"])
            / float(initial_mapping["hydrogen_mass_fraction"])
        )
        self.fluid.rho_code = np.ones(grid_cells) * rho
        self.fluid.mu = np.ones(grid_cells) * initial_mapping["muini"]


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
