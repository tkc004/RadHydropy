"""Helpers for the HM12 PIE photoionization-timescale example."""

from pathlib import Path

import h5py
import numpy as np


class Simwrap:
    """Build the uniform IC object consumed by ``radhydropy.io``."""

    def __init__(self, icparams, code_units):
        from types import SimpleNamespace

        grid_cells = int(icparams["nogrid"])
        boxsize = np.ones(1) * icparams["boxsize"]
        self.par = SimpleNamespace(time=np.ones(1) * icparams["time"])
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        self.par.simulation = SimpleNamespace(
            current_time=icparams["time"], box_size=icparams["boxsize"],
            coordinate_system=icparams["coordsys"],
        )
        self.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=0)
        self.mesh = SimpleNamespace()
        self.fluid = SimpleNamespace()
        self.mesh.boundary = np.linspace(
            0.0 * boxsize[0], boxsize[0], grid_cells + 1,
        )
        self.fluid.vel_code = np.zeros(grid_cells) * icparams["vini"]
        self.fluid.temp_code = np.ones(grid_cells) * icparams["tempini"]
        rho = (
            float(icparams["nHini"])
            * float(icparams["proton_mass_g"])
            / float(icparams["hydrogen_mass_fraction"])
        )
        self.fluid.rho_code = np.ones(grid_cells) * rho
        self.fluid.mu = np.ones(grid_cells) * icparams["muini"]


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
