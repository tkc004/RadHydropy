"""Typed boundaries between physical quantities and runtime arrays.

This module is intentionally independent of the solver.  It establishes the
unit contract that solver and source-process callers can adopt incrementally:
``CodeFluidState`` contains numeric code-unit arrays, while
``CgsSourceState`` contains numeric cgs arrays.  Physical inputs must cross
the boundary as real ``unyt`` quantities.
"""

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from radhydropy.units import CodeUnits, code_unit_scales


class UnitBoundaryError(ValueError):
    """Raised when a value crosses a unit boundary with the wrong type."""


def _plain_array(name, value):
    if value is None:
        return None
    if hasattr(value, "units") or hasattr(value, "to_value"):
        raise UnitBoundaryError(
            f"{name} must be a unitless numeric code-unit value; "
            "convert physical quantities before constructing CodeFluidState"
        )
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise UnitBoundaryError(f"{name} contains non-finite values")
    return array.copy()


def _cgs_array(name, value):
    if value is None:
        return None
    if hasattr(value, "units") or hasattr(value, "to_value"):
        raise UnitBoundaryError(
            f"{name} must be a unitless numeric cgs value; "
            "strip units only at the typed cgs boundary"
        )
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise UnitBoundaryError(f"{name} contains non-finite values")
    return array.copy()


def _physical_value(name, value, unit):
    if value is None:
        return None
    if not hasattr(value, "to_value"):
        raise UnitBoundaryError(
            f"{name} requires a physical unyt quantity with units compatible "
            f"with {unit}"
        )
    try:
        result = np.asarray(value.to_value(unit), dtype=float)
    except Exception as exc:  # unyt raises several conversion exceptions
        raise UnitBoundaryError(f"{name} has incompatible units; expected {unit}") from exc
    if not np.all(np.isfinite(result)):
        raise UnitBoundaryError(f"{name} contains non-finite values")
    return result.copy()


def _validate_fields(instance, converter):
    for field in fields(instance):
        value = getattr(instance, field.name)
        if value is not None:
            setattr(instance, field.name, converter(field.name, value))


@dataclass
class CodeFluidState:
    """Numeric runtime fluid and conserved arrays in code units."""

    rho_code: np.ndarray
    vel_code: np.ndarray
    temp_code: np.ndarray
    pre_code: Optional[np.ndarray] = None
    specific_energy_code: Optional[np.ndarray] = None
    Mass_code: Optional[np.ndarray] = None
    Mom_code: Optional[np.ndarray] = None
    Energy_code: Optional[np.ndarray] = None
    ngamma_code: Optional[np.ndarray] = None
    mu_dimensionless: Optional[np.ndarray] = None
    xHI_dimensionless: Optional[np.ndarray] = None
    time_code: Optional[float] = None

    def __post_init__(self):
        _validate_fields(self, _plain_array)


@dataclass
class CgsSourceState:
    """Numeric cgs source state consumed by physics kernels."""

    boundary_cgs_cm: np.ndarray
    volume_cgs_cm3: np.ndarray
    rho_cgs_g_cm3: np.ndarray
    velocity_cgs_cm_s: np.ndarray
    temperature_cgs_K: np.ndarray
    specific_energy_cgs_erg_g: np.ndarray
    pressure_cgs_erg_cm3: Optional[np.ndarray] = None
    ngamma_cgs_cm3: Optional[np.ndarray] = None
    xHI_dimensionless: Optional[np.ndarray] = None
    mu_dimensionless: Optional[np.ndarray] = None
    time_cgs_s: Optional[float] = None

    def __post_init__(self):
        _validate_fields(self, _cgs_array)


def code_fluid_state_from_physical(
    *,
    code_units: CodeUnits,
    rho_unyt,
    vel_unyt,
    temp_unyt,
    pre_unyt=None,
    specific_energy_unyt=None,
    Mass_unyt=None,
    Mom_unyt=None,
    Energy_unyt=None,
    ngamma_unyt=None,
    mu_dimensionless=None,
    xHI_dimensionless=None,
    time_unyt=None,
):
    """Convert physical ``unyt`` quantities into strict code arrays."""
    if not isinstance(code_units, CodeUnits):
        raise UnitBoundaryError("code_units must be a CodeUnits instance")
    values = {
        "rho_code": _physical_value("rho_unyt", rho_unyt, code_units.density_unit),
        "vel_code": _physical_value("vel_unyt", vel_unyt, code_units.velocity_unit),
        "temp_code": _physical_value("temp_unyt", temp_unyt, code_units.temperature_unit),
        "pre_code": _physical_value("pre_unyt", pre_unyt, code_units.pressure_unit),
        "specific_energy_code": _physical_value(
            "specific_energy_unyt", specific_energy_unyt, code_units.specific_energy_unit
        ),
        "Mass_code": _physical_value("Mass_unyt", Mass_unyt, code_units.mass_unit),
        "Mom_code": _physical_value("Mom_unyt", Mom_unyt, code_units.momentum_unit),
        "Energy_code": _physical_value("Energy_unyt", Energy_unyt, code_units.energy_unit),
        "ngamma_code": _physical_value("ngamma_unyt", ngamma_unyt, code_units.number_density_unit),
        "mu_dimensionless": _plain_array("mu_dimensionless", mu_dimensionless),
        "xHI_dimensionless": _plain_array("xHI_dimensionless", xHI_dimensionless),
        "time_code": _physical_value("time_unyt", time_unyt, code_units.time_unit),
    }
    return CodeFluidState(**values)


def cgs_source_state_from_code(
    *,
    code_units: CodeUnits,
    fluid: CodeFluidState,
    boundary_code,
    volume_code,
):
    """Build a typed cgs source state from numeric runtime arrays."""
    if not isinstance(fluid, CodeFluidState):
        raise UnitBoundaryError("fluid must be a CodeFluidState instance")
    scales = code_unit_scales(code_units)
    if scales is None:
        raise UnitBoundaryError("cgs conversion requires code_units")
    converted = {
        "boundary_cgs_cm": _plain_array("boundary_code", boundary_code)
        * scales["length_cgs_cm"],
        "volume_cgs_cm3": _plain_array("volume_code", volume_code)
        * scales["volume_cgs_cm3"],
        "rho_cgs_g_cm3": fluid.rho_code * scales["density_cgs_g_cm3"],
        "velocity_cgs_cm_s": fluid.vel_code * scales["velocity_cgs_cm_s"],
        "temperature_cgs_K": fluid.temp_code * scales["temperature_cgs_K"],
        "specific_energy_cgs_erg_g": None,
        "pressure_cgs_erg_cm3": None if fluid.pre_code is None else fluid.pre_code * scales["pressure_cgs_erg_cm3"],
        "ngamma_cgs_cm3": None if fluid.ngamma_code is None else fluid.ngamma_code * scales["number_density_cgs_cm3"],
        "xHI_dimensionless": fluid.xHI_dimensionless,
        "mu_dimensionless": fluid.mu_dimensionless,
        "time_cgs_s": None if fluid.time_code is None else float(fluid.time_code) * scales["time_s"],
    }
    if fluid.Energy_code is not None and fluid.Mass_code is not None:
        converted["specific_energy_cgs_erg_g"] = (
            fluid.Energy_code / np.maximum(fluid.Mass_code, np.finfo(float).tiny)
            * scales["specific_energy_cgs_erg_g"]
        )
    elif fluid.specific_energy_code is not None:
        converted["specific_energy_cgs_erg_g"] = (
            fluid.specific_energy_code * scales["specific_energy_cgs_erg_g"]
        )
    else:
        raise UnitBoundaryError(
            "fluid requires specific_energy_code or both Energy_code and Mass_code"
        )
    return CgsSourceState(**converted)


def cgs_source_state_to_code(
    *, code_units: CodeUnits, source: CgsSourceState
) -> CodeFluidState:
    """Convert the representable fields of a cgs source state to code arrays."""
    if not isinstance(source, CgsSourceState):
        raise UnitBoundaryError("source must be a CgsSourceState instance")
    scales = code_unit_scales(code_units)
    return CodeFluidState(
        rho_code=source.rho_cgs_g_cm3 / scales["density_cgs_g_cm3"],
        vel_code=source.velocity_cgs_cm_s / scales["velocity_cgs_cm_s"],
        temp_code=source.temperature_cgs_K / scales["temperature_cgs_K"],
        pre_code=None if source.pressure_cgs_erg_cm3 is None else source.pressure_cgs_erg_cm3 / scales["pressure_cgs_erg_cm3"],
        specific_energy_code=source.specific_energy_cgs_erg_g / scales["specific_energy_cgs_erg_g"],
        ngamma_code=None if source.ngamma_cgs_cm3 is None else source.ngamma_cgs_cm3 / scales["number_density_cgs_cm3"],
        mu_dimensionless=source.mu_dimensionless,
        xHI_dimensionless=source.xHI_dimensionless,
        time_code=None if source.time_cgs_s is None else source.time_cgs_s / scales["time_s"],
    )
