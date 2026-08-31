"""Numerical solver subsystem helpers."""

import numpy as np
from types import SimpleNamespace
import unyt

import radhydropy.utils as ru
import radhydropy.chemistry_species.hydrogen as rh
import radhydropy.radiative_transfer as rrt
import radhydropy.thermo_chemistry as rtc
import radhydropy.gravity as rg
from radhydropy.constants import DEFAULT_SIGMA_GAMMA, SPEED_OF_LIGHT_CGS
from radhydropy.units import (
    CGS_AREA_UNIT, CGS_MASS_DENSITY_UNIT, CGS_NUMBER_DENSITY_UNIT,
    CGS_PHOTON_FLUX_UNIT, CGS_RATE_UNIT, CGS_VOLUME_UNIT,
    code_unit_scales, _as_cgs_float, _code_units, code_quantity_to_cgs,
    photon_number_density,
)
from radhydropy.arrays import as_named_array


def _set_angular_momentum_flux(fluid, order=0):
    """Build a mass-consistent flux for optional gas angular momentum."""
    if not (
        hasattr(fluid, 'specific_angular_momentum')
        and hasattr(fluid, 'AngularMomentum')
    ):
        return None
    mass_flux = np.asarray(fluid.Mass.flux, dtype=float)
    j_left = np.asarray(fluid.specific_angular_momentum.L, dtype=float)
    j_right = np.asarray(fluid.specific_angular_momentum.R, dtype=float)
    j_donor = np.where(mass_flux >= 0.0, j_left, j_right)
    if order == 1 and hasattr(fluid.specific_angular_momentum.R, 'first'):
        j_left_high = np.asarray(
            fluid.specific_angular_momentum.L.first, dtype=float
        )
        j_right_high = np.asarray(
            fluid.specific_angular_momentum.R.first, dtype=float
        )
    else:
        j_left_high, j_right_high = j_left, j_right
    j_high = np.where(mass_flux >= 0.0, j_left_high, j_right_high)
    # The high-order candidate is bounded at reconstruction time.  The
    # face-level FCT limiter in AddFluxes decides how much of its
    # antidiffusive correction is admissible for the two cells.
    fluid.angular_momentum_flux_low = as_named_array(
        mass_flux * j_donor
    )
    fluid.angular_momentum_flux_high = as_named_array(
        mass_flux * j_high
    )
    fluid.AngularMomentum.flux = as_named_array(
        fluid.angular_momentum_flux_high.copy()
    )
    # Rotational energy must use the same donor state as J.  Keep this as
    # a scratch field rather than reconstructing j a second time after
    # the mass flux has been limited.
    fluid.angular_momentum_face = as_named_array(j_high)
    fluid.angular_momentum_face_low = as_named_array(j_donor)
    return j_high

def _limit_angular_momentum_flux(solver, dt, mesh, fluid, par):
    """Apply a local FCT limiter to the angular-momentum flux.

    The donor flux is the bound-preserving low-order base.  A MUSCL
    correction is recovered face by face only while both adjacent cells
    retain locally bounded J/M.  This preserves the shared mass-flux
    relation and avoids globally reducing angular-momentum accuracy.
    """
    if not (
        hasattr(fluid, 'AngularMomentum')
        and hasattr(fluid, 'angular_momentum_flux_low')
        and hasattr(fluid, 'angular_momentum_flux_high')
    ):
        return
    low = np.asarray(fluid.angular_momentum_flux_low, dtype=float)
    high = np.asarray(fluid.angular_momentum_flux_high, dtype=float)
    correction = high - low
    factors = np.ones_like(low)
    scheme = str(getattr(par, 'angular_momentum_flux_scheme', 'fct')).lower()
    if scheme not in ('fct', 'donor'):
        raise ValueError(
            "Unknown angular_momentum_flux_scheme %r; valid options are fct, donor"
            % scheme
        )
    if scheme == 'donor':
        factors[...] = 0.0
    if not np.any(correction):
        fluid.AngularMomentum.flux = as_named_array(low.copy())
        fluid.angular_momentum_face = as_named_array(
            np.divide(low, np.asarray(fluid.Mass.flux, dtype=float),
                      out=np.zeros_like(low),
                      where=np.asarray(fluid.Mass.flux, dtype=float) != 0.0)
        )
        return

    mass = np.asarray(fluid.Mass, dtype=float)
    angular = np.asarray(fluid.AngularMomentum, dtype=float)
    area = np.asarray(mesh.area, dtype=float)
    first = int(getattr(par, 'noghost', 0))
    last = min(first + int(getattr(par, 'nogrid', len(mass) - first)), len(mass))
    physical = np.zeros(len(mass), dtype=bool)
    physical[first:last] = True
    mass_flux_area = np.asarray(fluid.Mass.flux, dtype=float) * area
    mass_new = mass + dt * (
        mass_flux_area - ru.periodic_roll(mass_flux_area, -1)
    )
    momentum_flux_area = np.asarray(fluid.Mom.flux, dtype=float) * area
    mom_new = np.asarray(fluid.Mom, dtype=float) + dt * (
        momentum_flux_area - ru.periodic_roll(momentum_flux_area, -1)
    )
    low_area = low * area
    trial_angular = angular + dt * (
        low_area - ru.periodic_roll(low_area, -1)
    )
    specific = np.divide(
        angular, mass, out=np.zeros_like(angular), where=mass > 0.0
    )
    lower = np.minimum.reduce((specific, ru.periodic_roll(specific, 1),
                               ru.periodic_roll(specific, -1)))
    upper = np.maximum.reduce((specific, ru.periodic_roll(specific, 1),
                               ru.periodic_roll(specific, -1)))
    correction_area = correction * area
    radius_face = np.abs(np.asarray(mesh.boundary[:-1], dtype=float))
    mass_flux = np.asarray(fluid.Mass.flux, dtype=float)
    rotational_low_flux = np.zeros_like(mass_flux)
    rotational_high_flux = np.zeros_like(mass_flux)
    valid_face_radius = (radius_face > 0.0) & np.isfinite(radius_face)
    rotational_low_flux[valid_face_radius] = (
        0.5 * np.asarray(fluid.angular_momentum_face_low, dtype=float)[valid_face_radius]**2
        / radius_face[valid_face_radius]**2 * mass_flux[valid_face_radius]
    )
    rotational_high_flux[valid_face_radius] = (
        0.5 * np.asarray(fluid.angular_momentum_face, dtype=float)[valid_face_radius]**2
        / radius_face[valid_face_radius]**2 * mass_flux[valid_face_radius]
    )
    rotational_correction_area = (
        rotational_high_flux - rotational_low_flux
    ) * area
    base_energy_flux = np.asarray(fluid.Energy.flux, dtype=float)
    if hasattr(fluid, 'rotational_energy_flux'):
        base_energy_flux -= np.asarray(fluid.rotational_energy_flux, dtype=float)
    base_energy_area = base_energy_flux * area
    base_energy = np.asarray(fluid.Energy, dtype=float) + dt * (
        base_energy_area - ru.periodic_roll(base_energy_area, -1)
    )
    rotational_low_area = rotational_low_flux * area
    trial_energy = base_energy + dt * (
        rotational_low_area - ru.periodic_roll(rotational_low_area, -1)
    )
    thermal = np.asarray(fluid.Energy, dtype=float).copy()
    kinetic = np.zeros_like(mass)
    np.divide(
        0.5 * np.asarray(fluid.Mom, dtype=float)**2,
        mass, out=kinetic, where=mass > 0.0
    )
    radius = np.abs(np.asarray(mesh.coordinate, dtype=float))
    rotational = np.zeros_like(mass)
    valid_radius = (mass > 0.0) & (radius > 0.0)
    rotational[valid_radius] = (
        0.5 * angular[valid_radius]**2
        / (mass[valid_radius] * radius[valid_radius]**2)
    )
    thermal -= kinetic + rotational
    thermal_fraction = np.divide(
        thermal, np.maximum(np.abs(np.asarray(fluid.Energy, dtype=float)), 1.0e-300),
        out=np.full_like(thermal, -np.inf),
        where=np.isfinite(np.asarray(fluid.Energy, dtype=float)),
    )
    margin = max(0.0, float(getattr(
        par, 'angular_momentum_energy_margin_fraction', 1.0e-4
    )))
    energy_problematic = physical & (thermal_fraction <= margin)
    factors[energy_problematic | np.roll(energy_problematic, -1)] = 0.0

    def valid_cell(index, value, energy_value):
        if not physical[index] or mass_new[index] <= 0.0:
            return True
        candidate = value / mass_new[index]
        tolerance = 1.0e-12 * max(1.0, abs(lower[index]), abs(upper[index]))
        angular_ok = (
            np.isfinite(candidate)
            and candidate >= lower[index] - tolerance
            and candidate <= upper[index] + tolerance
        )
        kinetic_new = 0.5 * mom_new[index]**2 / mass_new[index]
        radius_value = abs(float(np.asarray(mesh.coordinate, dtype=float)[index]))
        rotational_new = (
            0.5 * value**2 / (mass_new[index] * radius_value**2)
            if radius_value > 0.0 else 0.0
        )
        energy_ok = energy_value >= kinetic_new + rotational_new
        return angular_ok and energy_ok

    # Start from the donor update and recover as much MUSCL correction
    # as each face can support.  Each accepted face changes only its two
    # neighboring cells, so the limiter remains local.
    for face in range(len(factors)):
        if scheme == 'donor':
            continue
        if factors[face] == 0.0:
            continue
        left = (face - 1) % len(mass)
        right = face
        if not (physical[left] or physical[right]):
            continue
        increment = dt * correction_area[face]

        def trial_valid(alpha):
            return (
                valid_cell(
                    left,
                    trial_angular[left] - alpha * increment,
                    trial_energy[left] - alpha * dt * rotational_correction_area[face],
                )
                and valid_cell(
                    right,
                    trial_angular[right] + alpha * increment,
                    trial_energy[right] + alpha * dt * rotational_correction_area[face],
                )
            )

        if trial_valid(1.0):
            factors[face] = 1.0
            trial_angular[left] -= increment
            trial_angular[right] += increment
            trial_energy[left] -= dt * rotational_correction_area[face]
            trial_energy[right] += dt * rotational_correction_area[face]
            continue
        if not trial_valid(0.0):
            factors[face] = 0.0
            continue
        lo, hi = 0.0, 1.0
        for _ in range(48):
            middle = 0.5 * (lo + hi)
            if trial_valid(middle):
                lo = middle
            else:
                hi = middle
        factors[face] = lo
        trial_angular[left] -= lo * increment
        trial_angular[right] += lo * increment
        trial_energy[left] -= lo * dt * rotational_correction_area[face]
        trial_energy[right] += lo * dt * rotational_correction_area[face]

    limited = low + factors * correction
    fluid.AngularMomentum.flux = as_named_array(limited)
    mass_flux = np.asarray(fluid.Mass.flux, dtype=float)
    fluid.angular_momentum_face = as_named_array(
        np.divide(limited, mass_flux, out=np.zeros_like(limited),
                  where=mass_flux != 0.0)
    )
    fluid.angular_momentum_fct_factors = as_named_array(factors)
    if hasattr(fluid, 'rotational_energy_flux'):
        radius = np.abs(np.asarray(mesh.boundary[:-1], dtype=float))
        new_rotational = np.zeros_like(mass_flux)
        valid = (radius > 0.0) & np.isfinite(radius)
        new_rotational[valid] = (
            0.5 * fluid.angular_momentum_face[valid]**2
            / radius[valid]**2 * mass_flux[valid]
        )
        fluid.Energy.flux += new_rotational - np.asarray(
            fluid.rotational_energy_flux, dtype=float
        )
        fluid.rotational_energy_flux = as_named_array(new_rotational)

def _set_rotational_energy_flux(solver, mesh, fluid, par, j_face=None):
    """Add the advected rotational-energy flux to the total-energy flux."""
    if not solver._rotational_energy_enabled(par):
        return
    mass_flux = np.asarray(fluid.Mass.flux, dtype=float)
    if j_face is None:
        j_left = np.asarray(fluid.specific_angular_momentum.L, dtype=float)
        j_right = np.asarray(fluid.specific_angular_momentum.R, dtype=float)
        j_face = np.where(mass_flux >= 0.0, j_left, j_right)
    radius = np.abs(np.asarray(mesh.boundary[:-1], dtype=float))
    rotational_specific = np.zeros_like(radius)
    valid = np.isfinite(radius) & (radius > 0.0) & np.isfinite(j_face)
    rotational_specific[valid] = 0.5 * j_face[valid]**2 / radius[valid]**2
    rotational_flux = mass_flux * rotational_specific
    rotational_flux[~valid] = 0.0
    fluid.rotational_energy_flux = as_named_array(rotational_flux)
    fluid.Energy.flux += fluid.rotational_energy_flux

def _apply_local_angular_energy_fallback(solver, mesh, fluid, par):
    """Use first-order hydro fluxes only near a cold rotating cell."""
    if not (
        solver._rotational_energy_enabled(par)
        and hasattr(fluid, 'angular_momentum_mass_flux_low')
    ):
        return
    threshold = max(0.0, float(getattr(
        par, 'angular_momentum_energy_margin_fraction', 1.0e-4
    )))
    mass = np.asarray(fluid.Mass, dtype=float)
    momentum = np.asarray(fluid.Mom, dtype=float)
    energy = np.asarray(fluid.Energy, dtype=float)
    angular = np.asarray(fluid.AngularMomentum, dtype=float)
    radius = np.abs(np.asarray(mesh.coordinate, dtype=float))
    kinetic = np.zeros_like(mass)
    np.divide(0.5 * momentum**2, mass, out=kinetic, where=mass > 0.0)
    rotational = np.zeros_like(mass)
    valid_radius = (mass > 0.0) & (radius > 0.0)
    rotational[valid_radius] = (
        0.5 * angular[valid_radius]**2
        / (mass[valid_radius] * radius[valid_radius]**2)
    )
    thermal = energy - kinetic - rotational
    fraction = np.divide(
        thermal, np.maximum(np.abs(energy), 1.0e-300),
        out=np.full_like(thermal, -np.inf), where=np.isfinite(energy)
    )
    first = int(getattr(par, 'noghost', 0))
    last = min(first + int(getattr(par, 'nogrid', len(mass) - first)), len(mass))
    problematic = np.zeros(len(mass), dtype=bool)
    problematic[first:last] = fraction[first:last] <= threshold
    # Face i bounds cells i-1 and i.
    face_mask = problematic | np.roll(problematic, -1)
    if not np.any(face_mask):
        return
    fluid.Mass.flux[face_mask] = np.asarray(
        fluid.angular_momentum_mass_flux_low, dtype=float
    )[face_mask]
    fluid.Mom.flux[face_mask] = np.asarray(
        fluid.angular_momentum_mom_flux_low, dtype=float
    )[face_mask]
    fluid.Energy.flux[face_mask] = np.asarray(
        fluid.angular_momentum_energy_flux_low, dtype=float
    )[face_mask]
    fluid.angular_momentum_local_fallback = as_named_array(face_mask)

