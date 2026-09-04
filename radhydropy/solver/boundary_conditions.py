"""Numerical solver subsystem helpers."""

import numpy as np
from types import SimpleNamespace
import unyt

import radhydropy.utils as ru
import radhydropy.chemistry_species.hydrogen as rh
import radhydropy.radiative_transfer as rrt
import radhydropy.thermo_chemistry as rtc
import radhydropy.gravity as rg
from radhydropy.constants import DEFAULT_SIGMA_GAMMA_CGS_CM2, SPEED_OF_LIGHT_CGS
from radhydropy.units import (
    CGS_AREA_UNIT, CGS_MASS_DENSITY_UNIT, CGS_NUMBER_DENSITY_UNIT,
    CGS_PHOTON_FLUX_UNIT, CGS_RATE_UNIT, CGS_VOLUME_UNIT,
    code_unit_scales, _as_cgs_float, _code_units, code_quantity_to_cgs,
    photon_number_density,
)
from radhydropy.arrays import as_named_array


def _boundary_field_names(solver, fluid):
    fields = ['rho_code', 'vel_code', 'pre_code']
    if hasattr(fluid, 'specific_angular_momentum_code'):
        fields.append('specific_angular_momentum_code')
    if hasattr(fluid, 'xHI'):
        fields.append('xHI')
    if hasattr(fluid, 'ngamma_code'):
        fields.append('ngamma_code')
    return fields

def _copy_boundary_state(solver, fluid, target_slice, values):
    for attr, value in values.items():
        target = getattr(fluid, attr)
        if attr == 'ngamma_code' and np.ndim(target) == 2:
            value_array = np.asarray(value)
            if value_array.ndim == 1:
                value_array = value_array[:, None]
            target[:, target_slice] = value_array
        else:
            target[target_slice] = value

def _boundary_state(
    solver,
    fluid,
    source,
    include_velocity=True,
    negate_velocity=False,
    reverse=False,
):
    state = {
        'rho_code': fluid.rho_code[source],
        'pre_code': fluid.pre_code[source],
    }
    if include_velocity:
        velocity = fluid.vel_code[source]
        state['vel_code'] = -velocity if negate_velocity else velocity
    if hasattr(fluid, 'specific_angular_momentum_code'):
        state['specific_angular_momentum_code'] = fluid.specific_angular_momentum_code[source]
    if hasattr(fluid, 'xHI'):
        state['xHI'] = fluid.xHI[source]
    if hasattr(fluid, 'ngamma_code'):
        if np.ndim(fluid.ngamma_code) == 2:
            state['ngamma_code'] = fluid.ngamma_code[:, source]
        else:
            state['ngamma_code'] = fluid.ngamma_code[source]
    if reverse:
        for key, value in list(state.items()):
            state[key] = value[::-1]
    return state

def _to_code_number_density(solver, value, scales):
    density = np.asarray(photon_number_density(value).to_value(unyt.cm**-3), dtype=float)
    if scales is None:
        return density
    return density / scales['number_density_cgs_cm3']

def _apply_periodic_boundary(solver, fluid, interior, left_ghost, right_ghost, noghost):
    fields = solver._boundary_field_names(fluid)
    for attr in fields:
        quan = getattr(fluid, attr)
        if attr == 'ngamma_code' and np.ndim(quan) == 2:
            quan[:, left_ghost] = quan[:, interior][:, -noghost:]
            quan[:, right_ghost] = quan[:, interior][:, :noghost]
        else:
            quan[left_ghost] = quan[interior][-noghost:]
            quan[right_ghost] = quan[interior][:noghost]

def _apply_open_boundary(solver, fluid, first, nolast, left_ghost, right_ghost):
    fields = solver._boundary_field_names(fluid)
    for attr in fields:
        quan = getattr(fluid, attr)
        if attr == 'ngamma_code' and np.ndim(quan) == 2:
            quan[:, left_ghost] = quan[:, first]
            quan[:, right_ghost] = quan[:, nolast]
        else:
            quan[left_ghost] = quan[first]
            quan[right_ghost] = quan[nolast]

def _apply_reflecting_boundary(solver, fluid, interior, left_ghost, right_ghost, noghost):
    for attr in ('rho_code', 'pre_code', 'specific_angular_momentum_code'):
        if not hasattr(fluid, attr):
            continue
        quan = getattr(fluid, attr)
        quan[left_ghost] = quan[interior][:noghost][::-1]
        quan[right_ghost] = quan[interior][-noghost:][::-1]
    fluid.vel_code[left_ghost] = -fluid.vel_code[interior][:noghost][::-1]
    fluid.vel_code[right_ghost] = -fluid.vel_code[interior][-noghost:][::-1]

def _apply_spherical_inner_boundary(solver, mesh, fluid, first, noghost):
    mirror_start = first
    if mesh is not None and hasattr(mesh, 'boundary'):
        boundary_units = getattr(mesh.boundary, 'units', None)
        origin = 0.0 * boundary_units if boundary_units is not None else 0.0
        if mesh.boundary[first] < origin and mesh.boundary[first+1] > origin:
            mirror_start = first + 1
    left_state = solver._boundary_state(
        fluid,
        slice(mirror_start, mirror_start + noghost),
        negate_velocity=True,
        reverse=True,
    )
    solver._copy_boundary_state(fluid, slice(0, noghost), left_state)

def _apply_open_spherical_boundary(
    solver,
    mesh,
    fluid,
    par,
    scales,
    first,
    nolast,
    left_ghost,
    right_ghost,
    noghost,
):
    solver._apply_spherical_inner_boundary(mesh, fluid, first, noghost)
    right_state = solver._boundary_state(fluid, nolast)
    solver._copy_boundary_state(fluid, right_ghost, right_state)

def _apply_inflow_spherical_boundary(
    solver,
    mesh,
    fluid,
    par,
    scales,
    first,
    nolast,
    left_ghost,
    right_ghost,
    noghost,
):
    solver._apply_spherical_inner_boundary(mesh, fluid, first, noghost)
    right_state = {
        'rho_code': par.boundary.inflow_density,
        'vel_code': par.boundary.inflow_velocity,
        'pre_code': fluid.eos.pressure(
            par.boundary.inflow_density,
            par.boundary.inflow_temperature,
            par.boundary.inflow_mu,
        ),
    }
    if hasattr(fluid, 'specific_angular_momentum_code'):
        right_state['specific_angular_momentum_code'] = getattr(
            par, 'specific_angular_momentum_inflow', 0.0
        )
    if hasattr(fluid, 'xHI'):
        right_state['xHI'] = getattr(par, 'hydrogen_xHI_inflow', 1.0)
    if hasattr(fluid, 'ngamma_code'):
        right_state['ngamma_code'] = solver._to_code_number_density(
            getattr(par, 'hydrogen_ngamma_inflow', 0.0),
            scales,
        )
    solver._copy_boundary_state(fluid, right_ghost, right_state)

def _apply_outflow_spherical_boundary(
    solver,
    mesh,
    fluid,
    par,
    scales,
    first,
    nolast,
    left_ghost,
    right_ghost,
    noghost,
):
    left_state = {
        'rho_code': par.boundary.outflow_density,
        'vel_code': par.boundary.outflow_velocity,
        'pre_code': fluid.eos.pressure(
            par.boundary.outflow_density,
            par.boundary.outflow_temperature,
            par.boundary.outflow_mu,
        ),
    }
    if hasattr(fluid, 'specific_angular_momentum_code'):
        left_state['specific_angular_momentum_code'] = getattr(
            par, 'specific_angular_momentum_outflow', 0.0
        )
    if hasattr(fluid, 'xHI'):
        left_state['xHI'] = getattr(par, 'hydrogen_xHI_outflow', 1.0)
    if hasattr(fluid, 'ngamma_code'):
        left_state['ngamma_code'] = solver._to_code_number_density(
            getattr(par, 'hydrogen_ngamma_outflow', 0.0),
            scales,
        )
    solver._copy_boundary_state(fluid, left_ghost, left_state)
    right_state = solver._boundary_state(fluid, nolast)
    solver._copy_boundary_state(fluid, right_ghost, right_state)


def _apply_wind_spherical_boundary(
    solver,
    mesh,
    fluid,
    par,
    scales,
    first,
    nolast,
    left_ghost,
    right_ghost,
    noghost,
):
    """Inject a resolved, steady spherical wind at the inner boundary.

    The density in the ghost cells follows ``rho r**2 = constant``.  This
    supplies the finite-volume Riemann problem with the same radial profile
    as a freely expanding wind instead of presenting a constant, dense
    reservoir to the first active cell.  The active launch cells are
    initialized by the example/IC builder with the matching profile.
    """
    boundary_position = np.asarray(mesh.boundary, dtype=float)
    radius = np.abs(
        0.5 * (boundary_position[:noghost] + boundary_position[1:noghost + 1])
    )
    boundary_radius = np.abs(boundary_position)
    reference_radius = boundary_radius[first]
    if reference_radius <= 0.0 or np.any(radius <= 0.0):
        raise ValueError('WindSph requires a positive inner radius')

    density = par.boundary.outflow_density * (
        reference_radius / radius
    ) ** 2
    velocity = par.boundary.outflow_velocity * np.ones(noghost)
    mu = float(par.boundary.outflow_mu)
    pressure = fluid.eos.pressure(
        density,
        par.boundary.outflow_temperature * np.ones(noghost),
        mu,
    )
    left_state = {
        'rho_code': density,
        'vel_code': velocity,
        'pre_code': pressure,
    }
    if hasattr(fluid, 'specific_angular_momentum_code'):
        left_state['specific_angular_momentum_code'] = np.full(
            noghost, getattr(par, 'specific_angular_momentum_outflow', 0.0)
        )
    if hasattr(fluid, 'xHI'):
        left_state['xHI'] = np.full(
            noghost, getattr(par, 'hydrogen_xHI_outflow', 1.0)
        )
    if hasattr(fluid, 'ngamma_code'):
        left_state['ngamma_code'] = solver._to_code_number_density(
            getattr(par, 'hydrogen_ngamma_outflow', 0.0), scales
        )
    solver._copy_boundary_state(fluid, left_ghost, left_state)
    right_state = solver._boundary_state(fluid, nolast)
    solver._copy_boundary_state(fluid, right_ghost, right_state)
