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


def _cfl_density_floor(par):
    return max(0.0, float(np.asarray(
        getattr(par, 'cfl_density_floor', 0.0), dtype=float
    )))

def _dual_energy_enabled(par):
    return bool(getattr(par, 'dual_energy', False))

def _rotational_energy_enabled(par):
    return bool(getattr(par, 'gas_rotational_energy', False))

def _gravity_potential_energy_enabled(par):
    return bool(getattr(par, 'gravity_potential_energy', False))

def _gravity_potential(solver, mesh, par):
    if not solver._gravity_potential_energy_enabled(par):
        return None
    gravity = solver._gravity_model(par)
    if gravity is None or not hasattr(gravity, 'potential_on'):
        raise ValueError(
            'gravity_potential_energy requires a gravity model with potential_on'
        )
    return np.asarray(gravity.potential_on(mesh.coordinate), dtype=float)

def _gravity_potential_faces(solver, mesh, par):
    if not solver._gravity_potential_energy_enabled(par):
        return None
    gravity = solver._gravity_model(par)
    if gravity is None or not hasattr(gravity, 'potential_on'):
        raise ValueError(
            'gravity_potential_energy requires a gravity model with potential_on'
        )
    return np.asarray(gravity.potential_on(mesh.boundary[:-1]), dtype=float)

def _rotational_energy_density(solver, mesh, fluid, par):
    """Return opt-in rotational kinetic-energy density."""
    rho = np.asarray(fluid.rho_code, dtype=float)
    result = np.zeros_like(rho)
    if not solver._rotational_energy_enabled(par):
        return result
    if not getattr(par, 'gas_angular_momentum', False):
        raise ValueError(
            'gas_rotational_energy requires gas_angular_momentum: true'
        )
    if getattr(mesh, 'coordsys', None) != 'spherical':
        raise ValueError('gas_rotational_energy requires a spherical mesh')
    radius = np.asarray(mesh.coordinate, dtype=float)
    specific = np.asarray(fluid.specific_angular_momentum_code, dtype=float)
    valid = (
        np.isfinite(rho) & (rho > 0.0)
        & np.isfinite(specific) & np.isfinite(radius) & (radius > 0.0)
    )
    result[valid] = 0.5 * rho[valid] * specific[valid]**2 / radius[valid]**2
    return result

def _rotational_energy_from_conserved(solver, mesh, fluid, par):
    """Return opt-in rotational kinetic energy from conserved J and M."""
    result = np.zeros_like(np.asarray(fluid.Mass_code, dtype=float))
    if not solver._rotational_energy_enabled(par):
        return result
    if not hasattr(fluid, 'AngularMomentum_code'):
        return result
    mass = np.asarray(fluid.Mass_code, dtype=float)
    angular_momentum = np.asarray(fluid.AngularMomentum_code, dtype=float)
    radius = np.abs(np.asarray(mesh.coordinate, dtype=float))
    valid = (
        np.isfinite(mass) & (mass > 0.0)
        & np.isfinite(angular_momentum)
        & np.isfinite(radius) & (radius > 0.0)
    )
    result[valid] = (
        0.5 * angular_momentum[valid]**2
        / (mass[valid] * radius[valid]**2)
    )
    return result

def _dual_energy_eta(par, name, legacy):
    value = getattr(par, name, None)
    if value is None:
        value = getattr(par, 'dual_energy_switch', legacy)
    return max(0.0, float(value))
