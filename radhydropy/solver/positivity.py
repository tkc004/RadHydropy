"""Invariant-domain and dual-energy positivity helpers."""

import numpy as np


def limit_internal_flux(old_internal, flux, area, dt,
                        physical):
    """Limit dual internal-energy face fluxes to preserve positivity.

    The total-energy flux has its own paired-face limiter.  This second
    limiter acts on the independently evolved dual field, scaling only a
    face flux when either cell sharing that face would become negative.
    InternalEnergy is not the conservative authority, so this local
    limiter may be more restrictive than the total-energy limiter without
    changing conserved mass, momentum, or total energy.
    """
    result = np.ones(len(flux), dtype=float)
    state = np.asarray(old_internal, dtype=float).copy()
    area = np.asarray(area, dtype=float)
    dt_value = float(np.asarray(dt, dtype=float))
    for face in range(len(flux)):
        left = (face - 1) % len(state)
        right = face
        if not (physical[left] or physical[right]):
            continue
        increment = dt_value * float(flux[face]) * float(area[face])
        alpha = 1.0
        if increment > 0.0 and physical[left] and state[left] < increment:
            alpha = max(0.0, state[left] / increment)
        elif increment < 0.0 and physical[right] and state[right] < -increment:
            alpha = max(0.0, state[right] / -increment)
        result[face] = alpha
        applied = alpha * increment
        if physical[left]:
            state[left] -= applied
        if physical[right]:
            state[right] += applied
        # Roundoff at the positivity boundary must not be turned into a
        # negative dual state by the next vectorized update.
        if physical[left]:
            state[left] = max(0.0, state[left])
        if physical[right]:
            state[right] = max(0.0, state[right])
    return result

def positive_conserved_state(mass, momentum, energy, mass_floor=0.0,
                              energy_floor=0.0, relative_tolerance=1.0e-12,
                              angular_momentum=None, radius=None):
    """Return the invariant-domain admissibility mask for Euler states."""
    mass = np.asarray(mass, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    energy = np.asarray(energy, dtype=float)
    finite = np.isfinite(mass) & np.isfinite(momentum) & np.isfinite(energy)
    mass_ok = mass >= mass_floor
    internal = np.zeros_like(energy)
    positive_mass = mass > np.maximum(mass_floor, 0.0)
    internal[positive_mass] = (
        energy[positive_mass]
        - 0.5 * momentum[positive_mass]**2 / mass[positive_mass]
    )
    if angular_momentum is not None and radius is not None:
        angular_momentum = np.asarray(angular_momentum, dtype=float)
        radius = np.asarray(radius, dtype=float)
        valid_radius = positive_mass & np.isfinite(radius) & (radius > 0.0)
        internal[valid_radius] -= (
            0.5 * angular_momentum[valid_radius]**2
            / (mass[valid_radius] * radius[valid_radius]**2)
        )
    vacuum = ~positive_mass
    internal[vacuum] = energy[vacuum]
    kinetic = np.zeros_like(energy)
    kinetic[positive_mass] = (
        0.5 * momentum[positive_mass]**2 / mass[positive_mass]
    )
    # Cold pressureless states lie on the invariant-domain boundary.  A
    # relative tolerance prevents harmless cancellation in E-K from
    # turning that boundary state into a negative internal energy.
    tolerance = relative_tolerance * np.maximum(
        np.maximum(np.abs(energy), kinetic),
        np.maximum(np.abs(energy_floor), np.finfo(float).tiny),
    )
    return finite & mass_ok & (internal >= energy_floor - tolerance)
