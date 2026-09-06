"""Typed conversions between comoving and proper cosmological states.

The solver-side state is expressed in comoving/supercomoving code variables.
Physics source terms cross this module once and receive proper CGS values.
Representation and unit names are deliberately part of every public field.
"""

from dataclasses import dataclass

import numpy as np

from radhydropy.units import code_unit_scales


def validate_supercomoving_contract(par, mesh, fluid):
    """Validate the explicit cosmological runtime representation.

    This is intentionally strict once a state has adopted the new schema;
    callers still using the legacy shared field names fail with a diagnostic
    rather than silently mixing cosmic and supercomoving time.
    """
    expected = {
        "coordinate_frame": "comoving",
        "time_coordinate": "supercomoving",
        "velocity_representation": "supercomoving_peculiar",
    }
    for name, value in expected.items():
        if getattr(par, name, None) != value:
            raise ValueError(
                f"cosmological runtime requires par.{name}={value!r}"
            )
    required_mesh = (
        "x_comoving_code",
        "boundary_comoving_code",
        "width_comoving_code",
    )
    required_fluid = (
        "rho_comoving_code",
        "vel_supercomoving_code",
        "pre_supercomoving_code",
        "temp_supercomoving_code",
        "tau_supercomoving_code",
    )
    missing = [
        f"mesh.{name}" for name in required_mesh if not hasattr(mesh, name)
    ]
    missing.extend(
        f"fluid.{name}" for name in required_fluid if not hasattr(fluid, name)
    )
    missing.extend(
        f"par.{name}" for name in ("time_cosmic_code", "tau_supercomoving_code")
        if not hasattr(par, name)
    )
    if missing:
        raise ValueError(
            "incomplete supercomoving runtime state; missing "
            + ", ".join(missing)
        )
    if hasattr(fluid, "vel_code"):
        raise ValueError("legacy fluid.vel_code is forbidden in cosmological state")


def _array(name, value):
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result.copy()


@dataclass(frozen=True)
class SupercomovingState:
    """Numeric solver state in comoving/supercomoving code variables."""

    x_comoving_code: np.ndarray
    rho_comoving_code: np.ndarray
    vel_supercomoving_code: np.ndarray
    pre_supercomoving_code: np.ndarray
    temp_supercomoving_code: np.ndarray
    tau_supercomoving_code: float

    def __post_init__(self):
        for name in (
            "x_comoving_code",
            "rho_comoving_code",
            "vel_supercomoving_code",
            "pre_supercomoving_code",
            "temp_supercomoving_code",
        ):
            object.__setattr__(self, name, _array(name, getattr(self, name)))
        tau = float(self.tau_supercomoving_code)
        if not np.isfinite(tau):
            raise ValueError("tau_supercomoving_code must be finite")
        object.__setattr__(self, "tau_supercomoving_code", tau)


@dataclass(frozen=True)
class ProperCgsState:
    """Numeric proper physical state in CGS units."""

    x_proper_cgs_cm: np.ndarray
    rho_proper_cgs_g_cm3: np.ndarray
    vel_peculiar_proper_cgs_cm_s: np.ndarray
    pre_proper_cgs_erg_cm3: np.ndarray
    temp_proper_cgs_K: np.ndarray
    time_cosmic_cgs_s: float

    def __post_init__(self):
        for name in (
            "x_proper_cgs_cm",
            "rho_proper_cgs_g_cm3",
            "vel_peculiar_proper_cgs_cm_s",
            "pre_proper_cgs_erg_cm3",
            "temp_proper_cgs_K",
        ):
            object.__setattr__(self, name, _array(name, getattr(self, name)))
        time = float(self.time_cosmic_cgs_s)
        if not np.isfinite(time) or time <= 0.0:
            raise ValueError("time_cosmic_cgs_s must be positive and finite")
        object.__setattr__(self, "time_cosmic_cgs_s", time)


@dataclass(frozen=True)
class SupercomovingMeshState:
    """Mesh geometry in comoving code coordinates."""

    x_comoving_code: np.ndarray
    boundary_comoving_code: np.ndarray
    width_comoving_code: np.ndarray

    def __post_init__(self):
        for name in (
            "x_comoving_code",
            "boundary_comoving_code",
            "width_comoving_code",
        ):
            object.__setattr__(self, name, _array(name, getattr(self, name)))


@dataclass(frozen=True)
class SupercomovingHdf5State:
    """Complete canonical HDF5 payload without mutable dictionaries."""

    state: SupercomovingState
    mesh: SupercomovingMeshState
    box_size_comoving_code: np.ndarray


def supercomoving_to_cosmic_time(cosmology, tau_supercomoving_code):
    """Convert supercomoving code time to cosmic code time."""
    return np.asarray(
        cosmology.cosmic_time_from_supercomoving(
            np.asarray(tau_supercomoving_code, dtype=float)
        ),
        dtype=float,
    )


def proper_to_supercomoving_time(cosmology, time_cosmic_code):
    """Convert cosmic code time to supercomoving code time."""
    return np.asarray(
        cosmology.supercomoving_time(np.asarray(time_cosmic_code, dtype=float)),
        dtype=float,
    )


def to_proper_state(state, cosmology, code_units, gamma):
    """Convert a typed supercomoving code state to proper CGS values."""
    scales = code_unit_scales(code_units)
    tau = float(state.tau_supercomoving_code)
    cosmic_time, scale_factor, hubble = cosmology.background_state_from_supercomoving(tau)
    x_code = np.asarray(state.x_comoving_code, dtype=float)
    rho_code = np.asarray(state.rho_comoving_code, dtype=float)
    velocity_code = np.asarray(state.vel_supercomoving_code, dtype=float)
    pressure_code = np.asarray(state.pre_supercomoving_code, dtype=float)
    temperature_code = np.asarray(state.temp_supercomoving_code, dtype=float)
    x_proper = scale_factor * x_code * scales["length_cgs_cm"]
    rho_proper = rho_code / scale_factor**3 * scales["density_cgs_g_cm3"]
    peculiar_velocity = velocity_code / scale_factor * scales["velocity_cgs_cm_s"]
    pressure = (
        pressure_code / scale_factor ** (3.0 * float(gamma))
        * scales["pressure_cgs_erg_cm3"]
    )
    temperature = (
        temperature_code / scale_factor ** (3.0 * (float(gamma) - 1.0))
        * scales["temperature_cgs_K"]
    )
    return ProperCgsState(
        x_proper_cgs_cm=x_proper,
        rho_proper_cgs_g_cm3=rho_proper,
        vel_peculiar_proper_cgs_cm_s=peculiar_velocity,
        pre_proper_cgs_erg_cm3=pressure,
        temp_proper_cgs_K=temperature,
        time_cosmic_cgs_s=float(cosmic_time) * scales["time_s"],
    )


def to_supercomoving_state(
    state, cosmology, code_units, tau_supercomoving_code, gamma
):
    """Convert a proper CGS state to a typed supercomoving code state."""
    scales = code_unit_scales(code_units)
    tau = float(tau_supercomoving_code)
    _, scale_factor, _ = cosmology.background_state_from_supercomoving(tau)
    return SupercomovingState(
        x_comoving_code=(
            np.asarray(state.x_proper_cgs_cm, dtype=float)
            / scales["length_cgs_cm"] / scale_factor
        ),
        rho_comoving_code=(
            np.asarray(state.rho_proper_cgs_g_cm3, dtype=float)
            / scales["density_cgs_g_cm3"] * scale_factor**3
        ),
        vel_supercomoving_code=(
            np.asarray(state.vel_peculiar_proper_cgs_cm_s, dtype=float)
            / scales["velocity_cgs_cm_s"] * scale_factor
        ),
        pre_supercomoving_code=(
            np.asarray(state.pre_proper_cgs_erg_cm3, dtype=float)
            / scales["pressure_cgs_erg_cm3"]
            * scale_factor ** (3.0 * float(gamma))
        ),
        temp_supercomoving_code=(
            np.asarray(state.temp_proper_cgs_K, dtype=float)
            / scales["temperature_cgs_K"]
            * scale_factor ** (3.0 * (float(gamma) - 1.0))
        ),
        tau_supercomoving_code=tau,
    )
