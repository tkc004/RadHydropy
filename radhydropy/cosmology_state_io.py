"""Canonical HDF5 boundary for typed supercomoving cosmological states."""

import h5py
import numpy as np

from radhydropy.cosmology_state import (
    SupercomovingHdf5State,
    SupercomovingMeshState,
    SupercomovingState,
)
from radhydropy.units import code_unit_scales


def _dataset(group, name, value, units, **metadata):
    dataset = group.create_dataset(name, data=np.asarray(value, dtype=float))
    dataset.attrs["units"] = units
    for key, item in metadata.items():
        dataset.attrs[key] = item
    return dataset


def write_supercomoving_state_hdf5(
    filename,
    *,
    state,
    boundary_comoving_code,
    width_comoving_code,
    box_size_comoving_code,
    code_units,
):
    """Write a typed state using only canonical cosmological dataset names."""
    if not isinstance(state, SupercomovingState):
        raise TypeError("state must be a SupercomovingState")
    scales = code_unit_scales(code_units)
    with h5py.File(filename, "w") as handle:
        handle.attrs["coordinate_frame"] = "comoving"
        handle.attrs["time_coordinate"] = "supercomoving"
        handle.attrs["representation"] = "supercomoving_peculiar"
        header = handle.create_group("Header")
        _dataset(
            header,
            "tau_supercomoving_code",
            state.tau_supercomoving_code,
            "code_tau",
            coordinate_frame="comoving",
            time_coordinate="supercomoving",
            representation="supercomoving",
        )
        _dataset(
            header,
            "box_size_comoving_code",
            box_size_comoving_code,
            "code_length",
            coordinate_frame="comoving",
            representation="comoving",
        )
        data = handle.create_group("Data")
        _dataset(
            data,
            "x_comoving_code",
            state.x_comoving_code,
            "code_length",
            coordinate_frame="comoving",
            representation="comoving",
        )
        _dataset(
            data,
            "boundary_comoving_code",
            boundary_comoving_code,
            "code_length",
            coordinate_frame="comoving",
            representation="comoving",
        )
        _dataset(
            data,
            "width_comoving_code",
            width_comoving_code,
            "code_length",
            coordinate_frame="comoving",
            representation="comoving",
        )
        _dataset(
            data,
            "rho_comoving_code",
            state.rho_comoving_code,
            "code_density",
            coordinate_frame="comoving",
            representation="comoving",
            cgs_scale=scales["density_cgs_g_cm3"],
        )
        _dataset(
            data,
            "vel_supercomoving_code",
            state.vel_supercomoving_code,
            "code_velocity",
            coordinate_frame="comoving",
            representation="supercomoving_peculiar",
            cgs_scale=scales["velocity_cgs_cm_s"],
        )
        _dataset(
            data,
            "pre_supercomoving_code",
            state.pre_supercomoving_code,
            "code_pressure",
            coordinate_frame="comoving",
            representation="supercomoving",
            cgs_scale=scales["pressure_cgs_erg_cm3"],
        )
        _dataset(
            data,
            "temp_supercomoving_code",
            state.temp_supercomoving_code,
            "code_temperature",
            coordinate_frame="comoving",
            representation="supercomoving",
            cgs_scale=scales["temperature_cgs_K"],
        )


def read_supercomoving_state_hdf5(filename):
    """Read and validate the canonical typed cosmological HDF5 state."""
    with h5py.File(filename, "r") as handle:
        header = handle["Header"]
        data = handle["Data"]
        required = {
            "Header": ("tau_supercomoving_code", "box_size_comoving_code"),
            "Data": (
                "x_comoving_code",
                "boundary_comoving_code",
                "width_comoving_code",
                "rho_comoving_code",
                "vel_supercomoving_code",
                "pre_supercomoving_code",
                "temp_supercomoving_code",
            ),
        }
        for group_name, names in required.items():
            group = header if group_name == "Header" else data
            missing = [name for name in names if name not in group]
            if missing:
                raise ValueError(
                    f"canonical cosmological HDF5 state is missing {group_name}/"
                    + ", ".join(missing)
                )
        if "time_code" in header or "vel_code" in data:
            raise ValueError("legacy generic cosmological HDF5 names are forbidden")
        return SupercomovingHdf5State(
            state=SupercomovingState(
                x_comoving_code=data["x_comoving_code"][()],
                rho_comoving_code=data["rho_comoving_code"][()],
                vel_supercomoving_code=data["vel_supercomoving_code"][()],
                pre_supercomoving_code=data["pre_supercomoving_code"][()],
                temp_supercomoving_code=data["temp_supercomoving_code"][()],
                tau_supercomoving_code=header["tau_supercomoving_code"][()],
            ),
            mesh=SupercomovingMeshState(
                x_comoving_code=data["x_comoving_code"][()],
                boundary_comoving_code=data["boundary_comoving_code"][()],
                width_comoving_code=data["width_comoving_code"][()],
            ),
            box_size_comoving_code=np.asarray(
                header["box_size_comoving_code"][()], dtype=float
            ),
        )
