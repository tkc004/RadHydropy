# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Mesh construction utilities for one-dimensional simulations."""

from typing import Any

import numpy as np

from radhydropy.arrays import as_named_array
from radhydropy.runtime_fields import MeshGeometryState, runtime_fields
from radhydropy.units import _code_units, quantity_to_value


# set up the underlying mesh for fluid
class Mesh:
    """Store cell faces, cell centers, areas, and volumes.

    A ``Mesh`` instance expects its ``boundary`` attribute to be populated with
    physical cell-face locations before :meth:`SetUpMesh` is called.
    """

    boundary_comoving_code: Any
    width_comoving_code: Any
    coordinate_inverse_comoving_code: Any
    x_comoving_code: Any
    area_comoving_code: Any
    volume_comoving_code: Any
    boundary_proper_code: Any
    width_proper_code: Any
    coordinate_inverse_proper_code: Any
    x_proper_code: Any
    area_proper_code: Any
    volume_proper_code: Any

    def __init__(self) -> None:
        self.runtime_fields: Any = None
        self.geometry_state: Any = None
        self.par: Any = None

    def SetUpMesh(self, par: Any) -> None:  # noqa: N802
        """Build ghost cells and geometric factors from run parameters.

        Parameters
        ----------
        par : object
            Parameter object with ``nogrid``, ``noghost``, and ``coordsys``.
            Cartesian meshes also require ``area_proper``.

        Raises
        ------
        AttributeError
            If required mesh or parameter attributes are missing.
        ValueError
            If the mesh dimensions or coordinate system are invalid.

        """
        if not getattr(par, "supercomoving_coordinates", False):
            return self._set_up_proper_mesh(par)
        return self._set_up_supercomoving_mesh(par)

    def _set_up_supercomoving_mesh(self, par: Any) -> None:
        """Initialize a comoving mesh using supercomoving runtime names."""
        code_units = _code_units(par)
        if code_units is None:
            raise ValueError("SetUpMesh requires configured code units")
        nogrid = par.mesh.grid_cells
        noghost = par.mesh.ghost_cells
        self.CodeUnits = code_units
        self.coordsys = par.simulation.coordinate_system
        self.runtime_fields = runtime_fields(par)
        if not hasattr(self, "boundary_comoving_code"):
            raise AttributeError("mesh.boundary_comoving_code is required")
        for attr in ("grid_cells", "ghost_cells"):
            if not hasattr(par.mesh, attr):
                raise AttributeError(f"mesh.{attr} does not exist in params; quitting.")
        if nogrid < 1:
            raise ValueError("nogrid has to be at least 1")
        if noghost < 1:
            raise ValueError("noghost has to be at least 1")
        boundary_comoving_code = np.asarray(
            quantity_to_value(self.boundary_comoving_code, code_units.length_unit),
            dtype=float,
        )
        if len(boundary_comoving_code) != nogrid + 1:
            raise ValueError("boundary point and nogrid are inconsistent")
        # note that we use first (0) and final (nogrid+1) cells as ghost cells
        # to set boundary conditions

        # add ghost cells:
        dx = boundary_comoving_code[1] - boundary_comoving_code[0]
        start = boundary_comoving_code[0] - dx * noghost
        end = boundary_comoving_code[-1] + dx * noghost
        ghost_front = np.linspace(start, boundary_comoving_code[0] - dx, noghost)
        ghost_back = np.linspace(boundary_comoving_code[-1] + dx, end, noghost)
        self.boundary_comoving_code = as_named_array(
            np.concatenate((ghost_front, boundary_comoving_code, ghost_back)),
        )

        # mesh size
        self.width_comoving_code = as_named_array(
            self.boundary_comoving_code[1:] - self.boundary_comoving_code[:-1],
        )
        self.coordinate_inverse_comoving_code = as_named_array(
            1.0 / self.width_comoving_code,
        )
        self._set_up_supercomoving_geometry(par, code_units, nogrid, noghost)

        self.geometry_state = MeshGeometryState.from_arrays(
            self.runtime_fields,
            **{
                self.runtime_fields.coordinate: self.x_comoving_code,
                self.runtime_fields.boundary: self.boundary_comoving_code,
                self.runtime_fields.width: self.width_comoving_code,
                self.runtime_fields.area: self.area_comoving_code,
                self.runtime_fields.volume: self.volume_comoving_code,
            },
        )

        if np.any(self.volume_comoving_code == 0.0) or np.any(np.isnan(self.volume_comoving_code)):
            raise ValueError("volume vanished")

    def _set_up_supercomoving_geometry(
        self, par: Any, code_units: Any, nogrid: int, noghost: int,
    ) -> None:
        if self.coordsys == "cartesian":
            if not hasattr(par.mesh, "area_proper"):
                raise AttributeError("par.mesh.area_proper is required for a cartesian mesh")
            self.x_comoving_code = as_named_array(
                0.5 * (self.boundary_comoving_code[1:] + self.boundary_comoving_code[:-1]),
            )
            area_value = quantity_to_value(par.mesh.area_proper, code_units.area_unit)
            self.area_comoving_code = as_named_array(
                np.ones(nogrid + noghost * 2, dtype=float) * np.asarray(area_value, dtype=float),
            )
            self.volume_comoving_code = as_named_array(
                self.width_comoving_code * self.area_comoving_code,
            )
            return
        if self.coordsys != "spherical":
            raise ValueError(f"coordinate system unknown: {self.coordsys}")
        boundary = self.boundary_comoving_code
        self.area_comoving_code = as_named_array(boundary[:-1] ** 2 * 4.0 * np.pi)
        self.volume_comoving_code = as_named_array(
            np.abs(boundary[1:] ** 3 - boundary[:-1] ** 3) * 4.0 * np.pi / 3.0,
        )
        volume_denominator = boundary[1:] ** 3 - boundary[:-1] ** 3
        self.x_comoving_code = as_named_array(0.5 * (boundary[1:] + boundary[:-1]))
        nonzero = volume_denominator != 0.0
        self.x_comoving_code[nonzero] = (
            0.75
            * (boundary[1:][nonzero] ** 4 - boundary[:-1][nonzero] ** 4)
            / volume_denominator[nonzero]
        )
        for index in range(len(self.volume_comoving_code)):
            if boundary[index] < 0.0 < boundary[index + 1]:
                self.volume_comoving_code[index] = boundary[index + 1] ** 3 * 4.0 * np.pi / 3.0
                self.x_comoving_code[index] = 0.75 * boundary[index + 1]
                self.area_comoving_code[index] = 0.0

    def _set_up_proper_mesh(self, par: Any) -> None:
        """Initialize a proper-code mesh from explicit proper fields."""
        code_units = _code_units(par)
        if code_units is None:
            raise ValueError("SetUpMesh requires configured code units")
        nogrid = int(par.mesh.grid_cells)
        noghost = int(par.mesh.ghost_cells)
        if nogrid < 1 or noghost < 1:
            raise ValueError("proper-code mesh requires positive grid and ghost counts")
        if not hasattr(self, "boundary_proper_code"):
            raise AttributeError("mesh.boundary_proper_code is required")
        self.coordsys = par.simulation.coordinate_system
        boundary_proper_code = np.asarray(
            quantity_to_value(self.boundary_proper_code, code_units.length_unit),
            dtype=float,
        )
        if boundary_proper_code.size != nogrid + 1:
            raise ValueError("boundary point and grid_cells are inconsistent")
        width_initial_proper_code = boundary_proper_code[1] - boundary_proper_code[0]
        ghost_front_proper_code = np.linspace(
            boundary_proper_code[0] - noghost * width_initial_proper_code,
            boundary_proper_code[0] - width_initial_proper_code,
            noghost,
        )
        ghost_back_proper_code = np.linspace(
            boundary_proper_code[-1] + width_initial_proper_code,
            boundary_proper_code[-1] + noghost * width_initial_proper_code,
            noghost,
        )
        boundary_proper_code = as_named_array(
            np.concatenate(
                (ghost_front_proper_code, boundary_proper_code, ghost_back_proper_code),
            ),
        )
        width_proper_code = as_named_array(
            np.diff(boundary_proper_code),
        )
        self.boundary_proper_code = boundary_proper_code
        self.width_proper_code = width_proper_code
        self.coordinate_inverse_proper_code = as_named_array(1.0 / width_proper_code)
        self._set_up_proper_geometry(
            par,
            code_units,
            boundary_proper_code,
            width_proper_code,
        )
        self.runtime_fields = runtime_fields(par)
        self.geometry_state = MeshGeometryState.from_arrays(
            self.runtime_fields,
            **{
                self.runtime_fields.coordinate: self.x_proper_code,
                self.runtime_fields.boundary: self.boundary_proper_code,
                self.runtime_fields.width: self.width_proper_code,
                self.runtime_fields.area: self.area_proper_code,
                self.runtime_fields.volume: self.volume_proper_code,
            },
        )
        if np.any(self.volume_proper_code == 0.0) or np.any(
            np.isnan(self.volume_proper_code),
        ):
            raise ValueError("volume vanished")

    def _set_up_proper_geometry(
        self,
        par: Any,
        code_units: Any,
        boundary_proper_code: Any,
        width_proper_code: Any,
    ) -> None:
        if par.simulation.coordinate_system == "cartesian":
            if not hasattr(par.mesh, "area_proper"):
                raise AttributeError("par.mesh.area_proper is required for a cartesian mesh")
            self.x_proper_code = as_named_array(
                0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1]),
            )
            area_proper_code = quantity_to_value(par.mesh.area_proper, code_units.area_unit)
            self.area_proper_code = as_named_array(
                np.ones_like(width_proper_code) * float(np.asarray(area_proper_code)),
            )
            self.volume_proper_code = as_named_array(
                width_proper_code * self.area_proper_code,
            )
        elif par.simulation.coordinate_system == "spherical":
            self.area_proper_code = as_named_array(
                4.0 * np.pi * boundary_proper_code[:-1] ** 2,
            )
            self.volume_proper_code = as_named_array(
                np.abs(boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3)
                * 4.0
                * np.pi
                / 3.0,
            )
            volume_denominator_proper_code = (
                boundary_proper_code[1:] ** 3 - boundary_proper_code[:-1] ** 3
            )
            self.x_proper_code = as_named_array(
                0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1]),
            )
            valid = volume_denominator_proper_code != 0.0
            self.x_proper_code[valid] = (
                0.75
                * (boundary_proper_code[1:][valid] ** 4 - boundary_proper_code[:-1][valid] ** 4)
                / volume_denominator_proper_code[valid]
            )
            for index in range(self.volume_proper_code.size):
                if boundary_proper_code[index] < 0.0 < boundary_proper_code[index + 1]:
                    self.volume_proper_code[index] = (
                        boundary_proper_code[index + 1] ** 3 * 4.0 * np.pi / 3.0
                    )
                    self.x_proper_code[index] = 0.75 * boundary_proper_code[index + 1]
                    self.area_proper_code[index] = 0.0
        else:
            raise ValueError(
                f"coordinate system unknown: {par.simulation.coordinate_system}",
            )
