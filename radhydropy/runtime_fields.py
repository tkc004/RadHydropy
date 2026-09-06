"""Canonical runtime field names selected by the simulation representation.

The solver operates on one explicit representation at a time.  This module
contains the naming contract used while migrating solver consumers; it does
not provide legacy aliases or fallback lookups.
"""

from dataclasses import dataclass

import numpy as np


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

    x_comoving_code: np.ndarray | None = None
    boundary_comoving_code: np.ndarray | None = None
    width_comoving_code: np.ndarray | None = None
    area_comoving_code: np.ndarray | None = None
    volume_comoving_code: np.ndarray | None = None
    x_proper_code: np.ndarray | None = None
    boundary_proper_code: np.ndarray | None = None
    width_proper_code: np.ndarray | None = None
    area_proper_code: np.ndarray | None = None
    volume_proper_code: np.ndarray | None = None

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if value is not None and (
                hasattr(value, "units") or hasattr(value, "to_value")
            ):
                raise TypeError(
                    f"{name} must be a unitless numeric mesh code array"
                )

    @classmethod
    def from_arrays(cls, fields, *, coordinate, boundary, width, area, volume):
        values = {
            "coordinate": np.asarray(coordinate, dtype=float).copy(),
            "boundary": np.asarray(boundary, dtype=float).copy(),
            "width": np.asarray(width, dtype=float).copy(),
            "area": np.asarray(area, dtype=float).copy(),
            "volume": np.asarray(volume, dtype=float).copy(),
        }
        return cls(
            **{
                fields.coordinate: values["coordinate"],
                fields.boundary: values["boundary"],
                fields.width: values["width"],
                fields.area: values["area"],
                fields.volume: values["volume"],
            }
        )


@dataclass
class FluidRuntimeState:
    """Mutable primitive fluid arrays with representation-specific names."""

    rho_comoving_code: np.ndarray | None = None
    vel_supercomoving_code: np.ndarray | None = None
    pre_supercomoving_code: np.ndarray | None = None
    temp_supercomoving_code: np.ndarray | None = None
    tau_supercomoving_code: float | None = None
    rho_proper_code: np.ndarray | None = None
    vel_proper_code: np.ndarray | None = None
    pre_proper_code: np.ndarray | None = None
    temp_proper_code: np.ndarray | None = None
    time_proper_code: float | None = None
    mu_dimensionless: np.ndarray | None = None
    xHI_dimensionless: np.ndarray | None = None

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if value is not None and (
                hasattr(value, "units") or hasattr(value, "to_value")
            ):
                raise TypeError(
                    f"{name} must be a unitless numeric fluid code value"
                )

    @classmethod
    def from_arrays(
        cls, fields, *, density, velocity, pressure, temperature, time,
        mu=None, xHI=None,
    ):
        return cls(
            **{
                # Keep the runtime arrays live: solver updates must be visible
                # through the typed state without a synchronization alias.
                fields.density: density,
                fields.velocity: velocity,
                fields.pressure: pressure,
                fields.temperature: temperature,
                # Runtime clocks are numeric values in the active code-time
                # coordinate; physical time conversion happens at boundaries.
                fields.time: time,
                "mu_dimensionless": mu,
                "xHI_dimensionless": xHI,
            }
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


def runtime_fields(par):
    """Return the canonical field contract for ``par``.

    Cosmological runs must explicitly opt into the comoving/supercomoving
    representation.  All other runs use proper-code names.
    """
    cosmological = bool(
        getattr(par, "cosmological_expansion", False)
        or getattr(par, "supercomoving_coordinates", False)
    )
    if cosmological:
        if getattr(par, "coordinate_frame", None) != "comoving":
            raise ValueError(
                "cosmological runtime requires coordinate_frame='comoving'"
            )
        if getattr(par, "time_coordinate", None) != "supercomoving":
            raise ValueError(
                "cosmological runtime requires time_coordinate='supercomoving'"
            )
        if getattr(par, "velocity_representation", None) != "supercomoving_peculiar":
            raise ValueError(
                "cosmological runtime requires velocity_representation="
                "'supercomoving_peculiar'"
            )
        return SUPERCOMOVING_RUNTIME_FIELDS
    return PROPER_RUNTIME_FIELDS


def require_runtime_fields(owner, fields, owner_name):
    """Reject an object that has not been initialized with canonical fields."""
    missing = [name for name in fields.__dict__.values() if not hasattr(owner, name)]
    if missing:
        raise ValueError(
            f"{owner_name} is missing canonical runtime fields: "
            + ", ".join(missing)
        )
