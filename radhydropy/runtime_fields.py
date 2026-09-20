# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Canonical runtime field names selected by the simulation representation.

The solver operates on one explicit representation at a time.  This module
contains the naming contract used while migrating solver consumers; it does
not provide legacy aliases or fallback lookups.
"""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from radhydropy.arrays import as_named_array

RuntimeArray = np.ndarray[Any, np.dtype[np.float64]]


def _validated_runtime_array(
    name: str,
    value: Any,
    kind: str = "runtime",
) -> RuntimeArray:
    """Return a finite, one-dimensional, unitless runtime array."""
    if hasattr(value, "units") or hasattr(value, "to_value"):
        raise TypeError(f"{name} must be a unitless numeric {kind} code value")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a numeric {kind} code array",
        ) from exc
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; received {array.ndim} dimensions",
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast("RuntimeArray", array)


def _validated_runtime_scalar(
    name: str,
    value: Any,
    kind: str = "runtime",
) -> float:
    """Return a finite, unitless scalar runtime value."""
    if hasattr(value, "units") or hasattr(value, "to_value"):
        raise TypeError(f"{name} must be a unitless numeric {kind} code value")
    try:
        scalar = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric {kind} code scalar") from exc
    if scalar.ndim != 0:
        raise ValueError(
            f"{name} must be scalar; received {scalar.ndim} dimensions",
        )
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must contain a finite value")
    return float(scalar)


def _named_runtime_array(
    name: str,
    value: Any,
    kind: str = "runtime",
) -> RuntimeArray:
    """Validate and preserve a mutable named array at the solver boundary."""
    return cast(
        "RuntimeArray",
        as_named_array(_validated_runtime_array(name, value, kind)),
    )


@dataclass(frozen=True)
class RuntimeFieldNames:
    """Names of the mutable runtime arrays for one representation."""

    coordinate: str
    boundary: str
    width: str
    area: str
    volume: str
    density: str
    velocity: str
    pressure: str
    temperature: str
    time: str


@dataclass
class MeshGeometryState:
    """Mutable mesh geometry with one explicitly named representation."""

    x_comoving_code: RuntimeArray | None = None
    boundary_comoving_code: RuntimeArray | None = None
    width_comoving_code: RuntimeArray | None = None
    area_comoving_code: RuntimeArray | None = None
    volume_comoving_code: RuntimeArray | None = None
    x_proper_code: RuntimeArray | None = None
    boundary_proper_code: RuntimeArray | None = None
    width_proper_code: RuntimeArray | None = None
    area_proper_code: RuntimeArray | None = None
    volume_proper_code: RuntimeArray | None = None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value is not None:
                setattr(
                    self,
                    name,
                    as_named_array(_validated_runtime_array(name, value, "mesh")),
                )
        _validate_mesh_state_lengths(self)

    @classmethod
    def from_arrays(cls, fields: RuntimeFieldNames, **arrays: Any) -> "MeshGeometryState":
        required = (
            fields.coordinate,
            fields.boundary,
            fields.width,
            fields.area,
            fields.volume,
        )
        missing = [name for name in required if name not in arrays]
        if missing:
            raise TypeError(
                "missing canonical mesh runtime arrays: " + ", ".join(missing),
            )
        values: dict[str, Any] = {
            name: _named_runtime_array(name, arrays[name], kind="mesh") for name in required
        }
        return cls(
            **values,
        )


@dataclass
class FluidRuntimeState:
    """Mutable primitive fluid arrays with representation-specific names."""

    rho_comoving_code: RuntimeArray | None = None
    vel_supercomoving_code: RuntimeArray | None = None
    pre_supercomoving_code: RuntimeArray | None = None
    temp_supercomoving_code: RuntimeArray | None = None
    tau_supercomoving_code: float | None = None
    rho_proper_code: RuntimeArray | None = None
    vel_proper_code: RuntimeArray | None = None
    pre_proper_code: RuntimeArray | None = None
    temp_proper_code: RuntimeArray | None = None
    time_proper_code: float | None = None
    mu_dimensionless: RuntimeArray | None = None
    xHI_dimensionless: RuntimeArray | None = None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value is None:
                continue
            if name in ("tau_supercomoving_code", "time_proper_code"):
                setattr(self, name, _validated_runtime_scalar(name, value, "fluid"))
            else:
                setattr(
                    self,
                    name,
                    as_named_array(
                        _validated_runtime_array(name, value, "fluid"),
                    ),
                )
        _validate_fluid_state_lengths(self)

    @classmethod
    def from_arrays(
        cls,
        fields: RuntimeFieldNames,
        *,
        mu_dimensionless: Any = None,
        xHI_dimensionless: Any = None,
        **arrays: Any,
    ) -> "FluidRuntimeState":
        required = (
            fields.density,
            fields.velocity,
            fields.pressure,
            fields.temperature,
            fields.time,
        )
        missing = [name for name in required if name not in arrays]
        if missing:
            raise TypeError(
                "missing canonical fluid runtime arrays: " + ", ".join(missing),
            )
        uninitialized = [name for name in required if arrays[name] is None]
        if uninitialized:
            raise ValueError(
                "canonical fluid runtime fields are uninitialized: " + ", ".join(uninitialized),
            )
        values = {
            name: _named_runtime_array(name, arrays[name], kind="fluid")
            if name != fields.time
            else arrays[name]
            for name in required
        }
        values.update(
            mu_dimensionless=mu_dimensionless,
            xHI_dimensionless=xHI_dimensionless,
        )
        return cls(**cast("Any", values))


def _validate_mesh_state_lengths(state: MeshGeometryState) -> None:
    """Validate lengths for the mesh fields that are present."""
    cell_arrays = (
        "x_comoving_code",
        "width_comoving_code",
        "area_comoving_code",
        "volume_comoving_code",
        "x_proper_code",
        "width_proper_code",
        "area_proper_code",
        "volume_proper_code",
    )
    lengths = {
        name: len(getattr(state, name)) for name in cell_arrays if getattr(state, name) is not None
    }
    if len(set(lengths.values())) > 1:
        raise ValueError(
            "mesh cell-centered runtime arrays must have matching lengths: "
            + ", ".join(f"{name}={length}" for name, length in lengths.items()),
        )
    for name in ("boundary_comoving_code", "boundary_proper_code"):
        boundary = getattr(state, name)
        if boundary is not None and lengths:
            cell_count = next(iter(lengths.values()))
            if len(boundary) != cell_count + 1:
                raise ValueError(
                    f"{name} must have one more entry than mesh cell arrays",
                )


def _validate_fluid_state_lengths(state: FluidRuntimeState) -> None:
    """Validate lengths for fluid primitive and auxiliary fields."""
    array_names = (
        "rho_comoving_code",
        "vel_supercomoving_code",
        "pre_supercomoving_code",
        "temp_supercomoving_code",
        "rho_proper_code",
        "vel_proper_code",
        "pre_proper_code",
        "temp_proper_code",
    )
    lengths = {
        name: len(getattr(state, name)) for name in array_names if getattr(state, name) is not None
    }
    if len(set(lengths.values())) > 1:
        raise ValueError(
            "fluid runtime arrays must have matching lengths: "
            + ", ".join(f"{name}={length}" for name, length in lengths.items()),
        )


def validate_runtime_state_shapes(
    mesh_state: MeshGeometryState,
    fluid_state: FluidRuntimeState,
    fields: RuntimeFieldNames,
) -> None:
    """Validate that selected mesh and fluid states describe the same cells."""
    required_mesh = (
        fields.coordinate,
        fields.boundary,
        fields.width,
        fields.area,
        fields.volume,
    )
    required_fluid = (
        fields.density,
        fields.velocity,
        fields.pressure,
        fields.temperature,
    )
    missing_mesh = [name for name in required_mesh if getattr(mesh_state, name, None) is None]
    missing_fluid = [name for name in required_fluid if getattr(fluid_state, name, None) is None]
    if missing_mesh or missing_fluid:
        details = []
        if missing_mesh:
            details.append("mesh: " + ", ".join(missing_mesh))
        if missing_fluid:
            details.append("fluid: " + ", ".join(missing_fluid))
        raise ValueError("runtime state is uninitialized (" + "; ".join(details) + ")")
    mesh_arrays = tuple(
        getattr(mesh_state, name)
        for name in (
            fields.coordinate,
            fields.boundary,
            fields.width,
            fields.area,
            fields.volume,
        )
    )
    fluid_arrays = tuple(
        getattr(fluid_state, name)
        for name in (
            fields.density,
            fields.velocity,
            fields.pressure,
            fields.temperature,
        )
    )
    mesh_cell_count = len(mesh_arrays[0])
    if len(mesh_arrays[1]) != mesh_cell_count + 1:
        raise ValueError(
            f"{fields.boundary} must have one more entry than mesh cells",
        )
    fluid_cell_count = len(fluid_arrays[0])
    if mesh_cell_count != fluid_cell_count:
        raise ValueError(
            "mesh and fluid runtime cell counts disagree: "
            f"mesh={mesh_cell_count}, fluid={fluid_cell_count}",
        )


PROPER_RUNTIME_FIELDS = RuntimeFieldNames(
    coordinate="x_proper_code",
    boundary="boundary_proper_code",
    width="width_proper_code",
    area="area_proper_code",
    volume="volume_proper_code",
    density="rho_proper_code",
    velocity="vel_proper_code",
    pressure="pre_proper_code",
    temperature="temp_proper_code",
    time="time_proper_code",
)


SUPERCOMOVING_RUNTIME_FIELDS = RuntimeFieldNames(
    coordinate="x_comoving_code",
    boundary="boundary_comoving_code",
    width="width_comoving_code",
    area="area_comoving_code",
    volume="volume_comoving_code",
    density="rho_comoving_code",
    velocity="vel_supercomoving_code",
    pressure="pre_supercomoving_code",
    temperature="temp_supercomoving_code",
    time="tau_supercomoving_code",
)


def select_fluid_primitive_arrays(
    runtime_state: FluidRuntimeState,
    par: Any,
) -> tuple[Any, ...]:
    """Return primitive arrays for the configured runtime representation."""
    fields = runtime_fields(par)
    if fields is PROPER_RUNTIME_FIELDS:
        return (
            runtime_state.rho_proper_code,
            runtime_state.vel_proper_code,
            runtime_state.pre_proper_code,
            runtime_state.temp_proper_code,
            runtime_state.time_proper_code,
        )
    if fields is SUPERCOMOVING_RUNTIME_FIELDS:
        return (
            runtime_state.rho_comoving_code,
            runtime_state.vel_supercomoving_code,
            runtime_state.pre_supercomoving_code,
            runtime_state.temp_supercomoving_code,
            runtime_state.tau_supercomoving_code,
        )
    raise ValueError("unsupported runtime field representation")


def select_mesh_geometry_arrays(
    geometry: MeshGeometryState,
    par: Any,
) -> tuple[Any, ...]:
    """Return mesh geometry arrays for the configured representation."""
    fields = runtime_fields(par)
    if fields is PROPER_RUNTIME_FIELDS:
        return (
            geometry.x_proper_code,
            geometry.boundary_proper_code,
            geometry.width_proper_code,
            geometry.area_proper_code,
            geometry.volume_proper_code,
        )
    if fields is SUPERCOMOVING_RUNTIME_FIELDS:
        return (
            geometry.x_comoving_code,
            geometry.boundary_comoving_code,
            geometry.width_comoving_code,
            geometry.area_comoving_code,
            geometry.volume_comoving_code,
        )
    raise ValueError("unsupported runtime field representation")


def runtime_fields(par: Any) -> RuntimeFieldNames:
    """Return the canonical field contract for ``par``.

    Cosmological runs must explicitly opt into the comoving/supercomoving
    representation.  All other runs use proper-code names.
    """
    cosmological = bool(
        getattr(par, "cosmological_expansion", False)
        or getattr(par, "supercomoving_coordinates", False),
    )
    if cosmological:
        if getattr(par, "coordinate_frame", None) != "comoving":
            raise ValueError(
                "cosmological runtime requires coordinate_frame='comoving'",
            )
        if getattr(par, "time_coordinate", None) != "supercomoving":
            raise ValueError(
                "cosmological runtime requires time_coordinate='supercomoving'",
            )
        if getattr(par, "velocity_representation", None) != "supercomoving_peculiar":
            raise ValueError(
                "cosmological runtime requires velocity_representation='supercomoving_peculiar'",
            )
        return SUPERCOMOVING_RUNTIME_FIELDS
    return PROPER_RUNTIME_FIELDS


def require_runtime_fields(
    owner: Any,
    fields: RuntimeFieldNames,
    owner_name: str,
) -> None:
    """Reject an object that has not been initialized with canonical fields."""
    missing = [
        name
        for name in fields.__dict__.values()
        if not hasattr(owner, name) or getattr(owner, name) is None
    ]
    if missing:
        raise ValueError(
            f"{owner_name} is missing canonical runtime fields: " + ", ".join(missing),
        )
