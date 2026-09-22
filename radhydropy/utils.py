# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Numerical and thermodynamic helper functions."""

from typing import Any

import numpy as np
import unyt

from radhydropy.arrays import as_named_array


def periodic_roll(values: Any, shift: int) -> Any:
    """Return a 1D periodic shift without calling ``np.roll``."""
    out = np.empty_like(values)
    if out.size == 0:
        return out

    shift = int(shift) % out.shape[0]
    if shift == 0:
        out[...] = values
    else:
        out[:shift] = values[-shift:]
        out[shift:] = values[:-shift]
    return out


def SafeDivide(numerator: Any, denominator: Any) -> Any:  # noqa: N802
    """Divide two ``unyt`` quantities and return zero where the denominator is zero."""
    if hasattr(numerator, "units") or hasattr(denominator, "units"):
        numerator_value, denominator_value = np.broadcast_arrays(
            np.asarray(getattr(numerator, "value", numerator), dtype=float),
            np.asarray(getattr(denominator, "value", denominator), dtype=float),
        )
        quotient = np.zeros_like(denominator_value, dtype=float)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            np.divide(
                numerator_value,
                denominator_value,
                out=quotient,
                where=denominator_value != 0.0,
            )
        numerator_units = getattr(numerator, "units", 1.0)
        denominator_units = getattr(denominator, "units", 1.0)
        return quotient * (numerator_units / denominator_units)
    numerator_value, denominator_value = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
    )
    quotient = np.zeros_like(denominator_value, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        np.divide(
            numerator_value,
            denominator_value,
            out=quotient,
            where=denominator_value != 0.0,
        )
    return as_named_array(quotient)


def CalPressure(rho: Any, temp: Any, mu: Any) -> Any:  # noqa: N802
    """Calculate ideal-gas pressure from density, temperature, and molecular weight."""
    if hasattr(rho, "units") or hasattr(temp, "units"):
        return rho / (mu * unyt.mp) * unyt.kb * temp
    return (
        np.asarray(rho, dtype=float) * np.asarray(temp, dtype=float) / np.asarray(mu, dtype=float)
    )


def CalTemperature(rho: Any, pressure: Any, mu: Any) -> Any:  # noqa: N802
    """Calculate ideal-gas temperature from density, pressure, and molecular weight."""
    if not (hasattr(rho, "units") or hasattr(pressure, "units")):
        return (
            np.asarray(pressure, dtype=float)
            / np.asarray(rho, dtype=float)
            * np.asarray(mu, dtype=float)
        )
    pressure_over_rho = SafeDivide(pressure, rho)
    return (pressure_over_rho * (mu * unyt.mp) / unyt.kb).to(unyt.K)


def CalEnergyDensity(pressure: Any, gamma: Any) -> Any:  # noqa: N802
    """Calculate thermal energy density for a polytropic gas."""
    return pressure / (gamma - 1.0)


def CalSoundSpeed(pressure: Any, rho: Any, gamma: Any) -> Any:  # noqa: N802
    """Calculate adiabatic sound speed and zero invalid values."""
    if not (hasattr(pressure, "units") or hasattr(rho, "units")):
        pressure_over_rho = SafeDivide(pressure, rho)
        soundspeed = np.sqrt(gamma * pressure_over_rho)
        soundspeed[np.isnan(soundspeed)] = 0.0
        return soundspeed
    pressure_over_rho = SafeDivide(pressure, rho)
    soundspeed = np.sqrt(gamma * pressure_over_rho).to(unyt.cm / unyt.s)
    soundspeed[np.isnan(soundspeed)] = 0.0 * unyt.cm / unyt.s
    return soundspeed


def CheckParamDimen(params: Any) -> bool | str:  # noqa: N802
    """Validate known dimensional parameters.

    Returns ``True`` when all recognized parameters have compatible dimensions;
    otherwise returns the first key with incompatible units.
    """
    unitdir = {
        "boxsize": 1.0 * unyt.pc,
        "tini": 1.0 * unyt.yr,
        "vini": 1.0 * unyt.pc / unyt.yr,
        "rhoini": 1.0 * unyt.g / unyt.cm**3,
        "tempini": 1.0 * unyt.K,
        "gamma": 1.0,
    }
    for key, expected_unit in unitdir.items():
        if key in params:
            try:
                CheckDimension(params[key], expected_unit)
            except unyt.exceptions.UnitOperationError:
                return key
    return True


def CheckDimension(a: Any, dimcheck: Any) -> None:  # noqa: N802
    """Raise a ``unyt`` error if ``a`` is not dimensionally compatible."""
    if not hasattr(a, "units"):
        return
    # The addition is the validation operation: unyt raises when dimensions
    # are incompatible.  Keep the result local because this helper is only a
    # predicate/exception boundary.
    _validated = a + dimcheck
    del _validated


def gaussian(x: Any, mu: Any, sig: Any) -> Any:
    """Evaluate a normalized one-dimensional Gaussian profile."""
    return np.exp(-0.5 * np.power(x - mu, 2.0) / np.power(sig, 2.0)) / (np.sqrt(2.0 * np.pi) * sig)


def gaussiansph(r: Any, sig: Any) -> Any:
    """Evaluate a normalized spherical Gaussian profile."""
    return np.exp(-0.5 * np.power(r, 2.0) / np.power(sig, 2.0)) / (np.sqrt(2.0 * np.pi) * sig) ** 3


def CalGradient(quan: Any, width_runtime_code: Any) -> Any:  # noqa: N802
    """Calculate a centered periodic gradient."""
    # only work for periodic boundary condition!
    return (periodic_roll(quan, -1) - periodic_roll(quan, 1)) / (2.0 * width_runtime_code)


def CalInterFaceFluxGLF(flux_L: float, flux_R: float, q_L: float, q_R: float, cmax: float) -> float:  # noqa: N802, N803
    """Calculate a Lax-Friedrichs interface flux."""
    # Global Lax Friedrich function
    # F_(l+1/2) = 0.5*(F_L+F_R)+0.5*cmax*(q_L-q_R)
    InterFaceFlux = 0.5 * (flux_L + flux_R)
    # apply artifical diffusion +0.5*cmax*(q_L-q_R)
    InterFaceFlux += 0.5 * cmax * (q_L - q_R)
    return InterFaceFlux


def CalFluxLimiter(rlim: Any, limiter: str = "minmod") -> Any:  # noqa: N802
    """Calculate a slope limiter from the ratio of neighboring gradients."""
    if limiter == "minmod":
        firststep = np.minimum(np.ones(len(rlim)), rlim)
        philim = np.maximum(np.zeros(len(rlim)), firststep)
    elif limiter in ("MC", "monotonized_central"):
        # Monotonized-central is less compressive than minmod while keeping
        # the TVD bound.  It is a useful default for resolved rarefactions.
        philim = np.maximum(
            0.0,
            np.minimum(
                np.minimum(2.0 * rlim, 0.5 * (1.0 + rlim)),
                2.0,
            ),
        )
    elif limiter == "vanLeer":
        # is it correct when rlim -> inf, philim -> 2?
        philim = (rlim + np.absolute(rlim)) / (1.0 + np.absolute(rlim))
    else:
        raise ValueError(f"flux limiter unknown: {limiter}")
    return philim


def extrapolateToFace(  # noqa: N802
    fluxarray: Any, xb: Any, fgrad: Any, order: int = 1,
) -> tuple[Any, Any]:
    """Extrapolate cell-centered values to left and right faces."""
    # numpy roll Rroll, put the right value to this cell
    if order == 0:
        flux_R = fluxarray
        flux_L = periodic_roll(fluxarray, 1)
    elif order == 1:
        xdhalf = 0.5 * (xb[1:] - xb[:-1])
        flux_R = fluxarray - fgrad * xdhalf
        # the following is correct in the first order case
        flux_L = periodic_roll(fluxarray + fgrad * xdhalf, 1)
    else:
        raise ValueError(f"order unknown: {order}")
    return flux_L, flux_R


def GetFQ(rho: Any, vel: Any, pre: Any, gamma: Any) -> tuple[Any, ...]:  # noqa: N802
    """Return Euler fluxes and conserved densities for mass, momentum, and energy."""
    Fmass = rho * vel
    qmass = rho
    Fmom = rho * vel * vel
    Fmom[np.logical_or(vel == 0.0, np.isnan(vel))] = 0.0 * rho[0] * vel[0] ** 2
    Fmom += pre
    qmom = rho * vel
    FEn = vel * (gamma * pre / (gamma - 1.0) + 0.5 * rho * vel**2)
    qEn = pre / (gamma - 1.0) + rho * vel**2 * 0.5
    return Fmass, qmass, Fmom, qmom, FEn, qEn


def CalFluxFromLR(  # noqa: N802, N803
    rho_L: Any,
    rho_R: Any,
    u_L: Any,
    u_R: Any,
    p_L: Any,
    p_R: Any,
    gamma: Any,
    cmax: Any,
) -> tuple[Any, Any, Any]:
    """Calculate Rusanov/GLF fluxes from left and right primitive states."""
    Fmass_L, qmass_L, Fmom_L, qmom_L, FEn_L, qEn_L = GetFQ(rho_L, u_L, p_L, gamma)
    Fmass_R, qmass_R, Fmom_R, qmom_R, FEn_R, qEn_R = GetFQ(rho_R, u_R, p_R, gamma)

    Mass_flux = CalInterFaceFluxGLF(Fmass_L, Fmass_R, qmass_L, qmass_R, cmax)
    Mom_flux = CalInterFaceFluxGLF(Fmom_L, Fmom_R, qmom_L, qmom_R, cmax)
    Energy_flux = CalInterFaceFluxGLF(FEn_L, FEn_R, qEn_L, qEn_R, cmax)
    return Mass_flux, Mom_flux, Energy_flux


def ApplyFluxLimiter(  # noqa: N802
    q: Any, flux_1: Any, flux_0: Any, limiter: str = "minmod",
) -> tuple[Any, Any]:
    """Blend first-order and second-order fluxes using a slope limiter."""
    # numpy roll Rroll, put the right value to this cell
    q_l1 = periodic_roll(q, 1)
    q_l2 = periodic_roll(q, 2)
    bottom = q - q_l1
    top = q_l1 - q_l2
    rlim = np.ones(len(q)) * 1000.0
    nonzero = bottom != 0.0
    rlim[nonzero] = np.asarray(top[nonzero] / bottom[nonzero])
    rlim[np.isnan(rlim)] = 0.0
    philim = CalFluxLimiter(rlim, limiter=limiter)
    return flux_0 - philim * (flux_0 - flux_1), philim
