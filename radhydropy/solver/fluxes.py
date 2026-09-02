"""Interface-state and Riemann flux calculations."""

import numpy as np

import radhydropy.utils as ru
from radhydropy.arrays import as_named_array


def vacuum_safe_primitive_state(rho, vel, pre):
    """Return a finite, positive primitive state for a face Riemann solve.

    This operates on temporary face states only.  It does not alter the
    cell-centered density or conserved variables, so a vacuum cell can be
    populated by a later hydrodynamic flux update.
    """
    rho_value = np.asarray(rho, dtype=float)
    vel_value = np.asarray(vel, dtype=float)
    pre_value = np.asarray(pre, dtype=float)
    active = np.isfinite(rho_value) & (rho_value > 0.0)
    finite_velocity = np.isfinite(vel_value)
    finite_pressure = np.isfinite(pre_value) & (pre_value > 0.0)
    rho_safe = np.where(active, rho_value, 0.0)
    vel_safe = np.where(active & finite_velocity, vel_value, 0.0)
    pre_safe = np.where(active & finite_pressure, pre_value, 0.0)

    def restore_units(values, original):
        units = getattr(original, 'units', None)
        return values * units if units is not None else as_named_array(values)

    return (
        restore_units(rho_safe, rho),
        restore_units(vel_safe, vel),
        restore_units(pre_safe, pre),
    )


def hllc_flux(rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, gamma):
    """Return an HLLC Euler flux for positive, non-vacuum states.

    The caller supplies the Rusanov flux for vacuum, non-finite, or
    degenerate states.  Keeping that fallback explicit is important for
    the vacuum examples: HLLC's star-state formula is undefined when one
    side has zero density.
    """
    rho_L = np.asarray(rho_L, dtype=float)
    vel_L = np.asarray(vel_L, dtype=float)
    pre_L = np.asarray(pre_L, dtype=float)
    rho_R = np.asarray(rho_R, dtype=float)
    vel_R = np.asarray(vel_R, dtype=float)
    pre_R = np.asarray(pre_R, dtype=float)
    valid = (
        np.isfinite(rho_L) & np.isfinite(vel_L) & np.isfinite(pre_L)
        & np.isfinite(rho_R) & np.isfinite(vel_R) & np.isfinite(pre_R)
        & (rho_L > 0.0) & (rho_R > 0.0)
        & (pre_L > 0.0) & (pre_R > 0.0)
    )
    sound_L = np.zeros_like(rho_L)
    sound_R = np.zeros_like(rho_R)
    with np.errstate(divide='ignore', invalid='ignore'):
        sound_L = np.sqrt(gamma * pre_L / rho_L)
        sound_R = np.sqrt(gamma * pre_R / rho_R)
    valid &= np.isfinite(sound_L) & np.isfinite(sound_R)

    energy_L = pre_L / (gamma - 1.0) + 0.5 * rho_L * vel_L**2
    energy_R = pre_R / (gamma - 1.0) + 0.5 * rho_R * vel_R**2
    flux_L = np.stack((rho_L * vel_L,
                       rho_L * vel_L**2 + pre_L,
                       vel_L * (gamma * pre_L / (gamma - 1.0)
                                + 0.5 * rho_L * vel_L**2)))
    flux_R = np.stack((rho_R * vel_R,
                       rho_R * vel_R**2 + pre_R,
                       vel_R * (gamma * pre_R / (gamma - 1.0)
                                + 0.5 * rho_R * vel_R**2)))
    result = 0.5 * (flux_L + flux_R)
    with np.errstate(divide='ignore', invalid='ignore'):
        wave_L = np.minimum(vel_L - sound_L, vel_R - sound_R)
        wave_R = np.maximum(vel_L + sound_L, vel_R + sound_R)
        wave_M = (
            pre_R - pre_L
            + rho_L * vel_L * (wave_L - vel_L)
            - rho_R * vel_R * (wave_R - vel_R)
        ) / (rho_L * (wave_L - vel_L) - rho_R * (wave_R - vel_R))
        pressure_M = pre_L + rho_L * (wave_L - vel_L) * (wave_M - vel_L)
        rho_star_L = rho_L * (wave_L - vel_L) / (wave_L - wave_M)
        rho_star_R = rho_R * (wave_R - vel_R) / (wave_R - wave_M)
        energy_star_L = (
            (wave_L - vel_L) * energy_L - pre_L * vel_L
            + pressure_M * wave_M
        ) / (wave_L - wave_M)
        energy_star_R = (
            (wave_R - vel_R) * energy_R - pre_R * vel_R
            + pressure_M * wave_M
        ) / (wave_R - wave_M)
    star_L = np.stack((rho_star_L, rho_star_L * wave_M, energy_star_L))
    star_R = np.stack((rho_star_R, rho_star_R * wave_M, energy_star_R))
    flux_star_L = flux_L + wave_L * (star_L - np.stack((rho_L, rho_L * vel_L, energy_L)))
    flux_star_R = flux_R + wave_R * (star_R - np.stack((rho_R, rho_R * vel_R, energy_R)))
    left = wave_L >= 0.0
    left_star = (wave_L < 0.0) & (wave_M >= 0.0)
    right_star = (wave_M < 0.0) & (wave_R > 0.0)
    right = wave_R <= 0.0
    result = np.where(left[None, :], flux_L, result)
    result = np.where(left_star[None, :], flux_star_L, result)
    result = np.where(right_star[None, :], flux_star_R, result)
    result = np.where(right[None, :], flux_R, result)
    valid &= np.isfinite(result).all(axis=0)
    return result, valid

def interface_fluxes(fluid, rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, method):
    states = fluid.eos.fluxes(rho_L, vel_L, pre_L)
    states_R = fluid.eos.fluxes(rho_R, vel_R, pre_R)
    if method != 'HLLC' or not getattr(fluid.eos, 'is_polytropic', False):
        return tuple(
            ru.CalInterFaceFluxGLF(left, right, qleft, qright, fluid.cmax)
            for left, right, qleft, qright in (
                (states[0], states_R[0], states[1], states_R[1]),
                (states[2], states_R[2], states[3], states_R[3]),
                (states[4], states_R[4], states[5], states_R[5]),
            )
        )
    hllc, valid = hllc_flux(
        rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, fluid.eos.gamma
    )
    rusanov = np.stack(tuple(
        ru.CalInterFaceFluxGLF(left, right, qleft, qright, fluid.cmax)
        for left, right, qleft, qright in (
            (states[0], states_R[0], states[1], states_R[1]),
            (states[2], states_R[2], states[3], states_R[3]),
            (states[4], states_R[4], states[5], states_R[5]),
        )
    ))
    flux = np.where(valid[None, :], hllc, rusanov)
    return tuple(flux[index] for index in range(3))
