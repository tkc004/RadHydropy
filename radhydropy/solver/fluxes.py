# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
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

    return (
        as_named_array(rho_safe),
        as_named_array(vel_safe),
        as_named_array(pre_safe),
    )


def hllc_flux(rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, gamma):  # noqa: N803
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
        np.isfinite(rho_L)
        & np.isfinite(vel_L)
        & np.isfinite(pre_L)
        & np.isfinite(rho_R)
        & np.isfinite(vel_R)
        & np.isfinite(pre_R)
        & (rho_L > 0.0)
        & (rho_R > 0.0)
        & (pre_L > 0.0)
        & (pre_R > 0.0)
    )
    sound_L = np.zeros_like(rho_L)
    sound_R = np.zeros_like(rho_R)
    with np.errstate(divide="ignore", invalid="ignore"):
        sound_L = np.sqrt(gamma * pre_L / rho_L)
        sound_R = np.sqrt(gamma * pre_R / rho_R)
    valid &= np.isfinite(sound_L) & np.isfinite(sound_R)

    energy_L = pre_L / (gamma - 1.0) + 0.5 * rho_L * vel_L**2
    energy_R = pre_R / (gamma - 1.0) + 0.5 * rho_R * vel_R**2
    flux_L = np.stack(
        (
            rho_L * vel_L,
            rho_L * vel_L**2 + pre_L,
            vel_L * (gamma * pre_L / (gamma - 1.0) + 0.5 * rho_L * vel_L**2),
        ),
    )
    flux_R = np.stack(
        (
            rho_R * vel_R,
            rho_R * vel_R**2 + pre_R,
            vel_R * (gamma * pre_R / (gamma - 1.0) + 0.5 * rho_R * vel_R**2),
        ),
    )
    result = 0.5 * (flux_L + flux_R)
    with np.errstate(divide="ignore", invalid="ignore"):
        wave_L = np.minimum(vel_L - sound_L, vel_R - sound_R)
        wave_R = np.maximum(vel_L + sound_L, vel_R + sound_R)
        wave_M = (
            pre_R - pre_L + rho_L * vel_L * (wave_L - vel_L) - rho_R * vel_R * (wave_R - vel_R)
        ) / (rho_L * (wave_L - vel_L) - rho_R * (wave_R - vel_R))
        pressure_M = pre_L + rho_L * (wave_L - vel_L) * (wave_M - vel_L)
        rho_star_L = rho_L * (wave_L - vel_L) / (wave_L - wave_M)
        rho_star_R = rho_R * (wave_R - vel_R) / (wave_R - wave_M)
        energy_star_L = ((wave_L - vel_L) * energy_L - pre_L * vel_L + pressure_M * wave_M) / (
            wave_L - wave_M
        )
        energy_star_R = ((wave_R - vel_R) * energy_R - pre_R * vel_R + pressure_M * wave_M) / (
            wave_R - wave_M
        )
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


def interface_fluxes(fluid, rho_L, vel_L, pre_L, rho_R, vel_R, pre_R, method):  # noqa: N803
    states = fluid.eos.fluxes(rho_L, vel_L, pre_L)
    states_R = fluid.eos.fluxes(rho_R, vel_R, pre_R)
    if method != "HLLC" or not getattr(fluid.eos, "is_polytropic", False):
        return tuple(
            ru.CalInterFaceFluxGLF(left, right, qleft, qright, fluid.cmax)
            for left, right, qleft, qright in (
                (states[0], states_R[0], states[1], states_R[1]),
                (states[2], states_R[2], states[3], states_R[3]),
                (states[4], states_R[4], states[5], states_R[5]),
            )
        )
    hllc, valid = hllc_flux(
        rho_L,
        vel_L,
        pre_L,
        rho_R,
        vel_R,
        pre_R,
        fluid.eos.gamma,
    )
    rusanov = np.stack(
        tuple(
            ru.CalInterFaceFluxGLF(left, right, qleft, qright, fluid.cmax)
            for left, right, qleft, qright in (
                (states[0], states_R[0], states[1], states_R[1]),
                (states[2], states_R[2], states[3], states_R[3]),
                (states[4], states_R[4], states[5], states_R[5]),
            )
        ),
    )
    flux = np.where(valid[None, :], hllc, rusanov)
    return tuple(flux[index] for index in range(3))


def set_flux_on_face(solver, fluid, par=None, order=0, method="Rusanov"):
    """Assemble limited mass, momentum, and energy face fluxes."""
    density_code, velocity_code, pressure_code, _ = solver.active_primitive_arrays(fluid, par)
    rho_L, vel_L, pre_L = vacuum_safe_primitive_state(
        density_code.L,
        velocity_code.L,
        pressure_code.L,
    )
    rho_R, vel_R, pre_R = vacuum_safe_primitive_state(
        density_code.R,
        velocity_code.R,
        pressure_code.R,
    )
    Mass_flux_0, Mom_flux_0, Energy_flux_0 = interface_fluxes(
        fluid,
        rho_L,
        vel_L,
        pre_L,
        rho_R,
        vel_R,
        pre_R,
        method,
    )
    if order == 0:
        fluid.Mass_code.flux = Mass_flux_0
        fluid.Mom_code.flux = Mom_flux_0
        fluid.Energy_code.flux = Energy_flux_0
        fluid.angular_momentum_mass_flux_low = as_named_array(Mass_flux_0.copy())
        fluid.angular_momentum_mom_flux_low = as_named_array(Mom_flux_0.copy())
        fluid.angular_momentum_energy_flux_low = as_named_array(Energy_flux_0.copy())
    elif order == 1:
        rho_L, vel_L, pre_L = vacuum_safe_primitive_state(
            density_code.L.first,
            velocity_code.L.first,
            pressure_code.L.first,
        )
        rho_R, vel_R, pre_R = vacuum_safe_primitive_state(
            density_code.R.first,
            velocity_code.R.first,
            pressure_code.R.first,
        )
        Mass_flux_1, Mom_flux_1, Energy_flux_1 = interface_fluxes(
            fluid,
            rho_L,
            vel_L,
            pre_L,
            rho_R,
            vel_R,
            pre_R,
            method,
        )
        solver.SetConservedDensityFlux(fluid, par=par)
        limiter = getattr(par, "flux_limiter", "minmod") if par is not None else "minmod"
        fluid.Mass_code.flux, fluid.philim_Mass_code = ru.ApplyFluxLimiter(
            fluid.Mass_code.q,
            Mass_flux_1,
            Mass_flux_0,
            limiter=limiter,
        )
        fluid.Mom_code.flux, fluid.philim_Mom_code = ru.ApplyFluxLimiter(
            fluid.Mom_code.q,
            Mom_flux_1,
            Mom_flux_0,
            limiter=limiter,
        )
        fluid.Energy_code.flux, fluid.philim_Energy_code = ru.ApplyFluxLimiter(
            fluid.Energy_code.q,
            Energy_flux_1,
            Energy_flux_0,
            limiter=limiter,
        )
        fluid.angular_momentum_mass_flux_low = as_named_array(Mass_flux_0.copy())
        fluid.angular_momentum_mom_flux_low = as_named_array(Mom_flux_0.copy())
        fluid.angular_momentum_energy_flux_low = as_named_array(Energy_flux_0.copy())
        # A MUSCL reconstruction is not valid across a vacuum jump.  Retain
        # the first-order flux for the complete local stencil.
        floor = solver.cfl_density_floor(par)
        reconstructed_density = (
            np.asarray(density_code.L.first, dtype=float),
            np.asarray(density_code.R.first, dtype=float),
        )
        reconstructed_pressure = (
            np.asarray(pressure_code.L.first, dtype=float),
            np.asarray(pressure_code.R.first, dtype=float),
        )
        vacuum_face = np.zeros_like(reconstructed_density[0], dtype=bool)
        for state_density, state_pressure in zip(
            reconstructed_density,
            reconstructed_pressure,
            strict=False,
        ):
            vacuum_face |= (
                ~np.isfinite(state_density)
                | (state_density <= floor)
                | ~np.isfinite(state_pressure)
                | (state_pressure <= 0.0)
            )
        vacuum_face |= np.roll(vacuum_face, -1)
        vacuum_face |= np.roll(vacuum_face, 1)
        fluid.Mass_code.flux[vacuum_face] = Mass_flux_0[vacuum_face]
        fluid.Mom_code.flux[vacuum_face] = Mom_flux_0[vacuum_face]
        fluid.Energy_code.flux[vacuum_face] = Energy_flux_0[vacuum_face]
    else:
        raise ValueError(f"order unknown: {order}")


def set_face_lr(solver, mesh, fluid, order=0):
    """Construct left and right primitive states at cell faces."""
    par = getattr(mesh, "par", getattr(mesh, "_par", None))
    geometry = solver.geometry_state(mesh, par)
    density_code, velocity_code, pressure_code, _ = solver.active_primitive_arrays(fluid, par)
    if order not in (0, 1):
        raise ValueError(f"order unknown: {order}")

    density_code.R = as_named_array(np.asarray(density_code, dtype=float).copy())
    density_code.L = ru.periodic_roll(density_code, 1)
    velocity_code.R = as_named_array(np.asarray(velocity_code, dtype=float).copy())
    velocity_code.L = ru.periodic_roll(velocity_code, 1)
    pressure_code.R = as_named_array(np.asarray(pressure_code, dtype=float).copy())
    pressure_code.L = ru.periodic_roll(pressure_code, 1)
    if hasattr(fluid, "specific_angular_momentum_code"):
        fluid.specific_angular_momentum_code.R = as_named_array(
            np.asarray(fluid.specific_angular_momentum_code, dtype=float).copy(),
        )
        fluid.specific_angular_momentum_code.L = ru.periodic_roll(
            fluid.specific_angular_momentum_code,
            1,
        )
    if order == 1:
        solver.SetGradient(mesh, fluid)
        density_code.R.first, density_code.L.first = ru.extrapolateToFace(
            density_code,
            geometry.boundary_runtime_code,
            density_code.grad,
            order=1,
        )
        velocity_code.R.first, velocity_code.L.first = ru.extrapolateToFace(
            velocity_code,
            geometry.boundary_runtime_code,
            velocity_code.grad,
            order=1,
        )
        pressure_code.R.first, pressure_code.L.first = ru.extrapolateToFace(
            pressure_code,
            geometry.boundary_runtime_code,
            pressure_code.grad,
            order=1,
        )
        if hasattr(fluid, "specific_angular_momentum_code"):
            (
                fluid.specific_angular_momentum_code.R.first,
                fluid.specific_angular_momentum_code.L.first,
            ) = ru.extrapolateToFace(
                fluid.specific_angular_momentum_code,
                geometry.boundary_runtime_code,
                fluid.specific_angular_momentum_code.grad,
                order=1,
            )
            j_right_cell = np.asarray(
                fluid.specific_angular_momentum_code.R,
                dtype=float,
            )
            j_left_cell = np.asarray(
                fluid.specific_angular_momentum_code.L,
                dtype=float,
            )
            j_min = np.minimum(j_left_cell, j_right_cell)
            j_max = np.maximum(j_left_cell, j_right_cell)
            fluid.specific_angular_momentum_code.R.first = as_named_array(
                np.clip(
                    np.asarray(fluid.specific_angular_momentum_code.R.first, dtype=float),
                    j_min,
                    j_max,
                ),
            )
            fluid.specific_angular_momentum_code.L.first = as_named_array(
                np.clip(
                    np.asarray(fluid.specific_angular_momentum_code.L.first, dtype=float),
                    j_min,
                    j_max,
                ),
            )
    solver.apply_low_density_face_mask(fluid, par, order)
    solver.apply_cosmological_background_boundary_face(mesh, fluid, order)
