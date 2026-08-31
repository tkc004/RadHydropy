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


def _boundary_field_names(solver, fluid):
    fields = ['rho', 'vel', 'pre']
    if hasattr(fluid, 'specific_angular_momentum'):
        fields.append('specific_angular_momentum')
    if hasattr(fluid, 'xHI'):
        fields.append('xHI')
    if hasattr(fluid, 'ngamma'):
        fields.append('ngamma')
    return fields

def _copy_boundary_state(solver, fluid, target_slice, values):
    for attr, value in values.items():
        target = getattr(fluid, attr)
        if attr == 'ngamma' and np.ndim(target) == 2:
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
        'rho': fluid.rho[source],
        'pre': fluid.pre[source],
    }
    if include_velocity:
        velocity = fluid.vel[source]
        state['vel'] = -velocity if negate_velocity else velocity
    if hasattr(fluid, 'specific_angular_momentum'):
        state['specific_angular_momentum'] = fluid.specific_angular_momentum[source]
    if hasattr(fluid, 'xHI'):
        state['xHI'] = fluid.xHI[source]
    if hasattr(fluid, 'ngamma'):
        if np.ndim(fluid.ngamma) == 2:
            state['ngamma'] = fluid.ngamma[:, source]
        else:
            state['ngamma'] = fluid.ngamma[source]
    if reverse:
        for key, value in list(state.items()):
            state[key] = value[::-1]
    return state

def _to_code_number_density(solver, value, scales):
    density = np.asarray(photon_number_density(value).to_value(unyt.cm**-3), dtype=float)
    if scales is None:
        return density
    return density / scales['number_density_cm3']

def _apply_periodic_boundary(solver, fluid, interior, left_ghost, right_ghost, noghost):
    fields = solver._boundary_field_names(fluid)
    for attr in fields:
        quan = getattr(fluid, attr)
        if attr == 'ngamma' and np.ndim(quan) == 2:
            quan[:, left_ghost] = quan[:, interior][:, -noghost:]
            quan[:, right_ghost] = quan[:, interior][:, :noghost]
        else:
            quan[left_ghost] = quan[interior][-noghost:]
            quan[right_ghost] = quan[interior][:noghost]

def _apply_open_boundary(solver, fluid, first, nolast, left_ghost, right_ghost):
    fields = solver._boundary_field_names(fluid)
    for attr in fields:
        quan = getattr(fluid, attr)
        if attr == 'ngamma' and np.ndim(quan) == 2:
            quan[:, left_ghost] = quan[:, first]
            quan[:, right_ghost] = quan[:, nolast]
        else:
            quan[left_ghost] = quan[first]
            quan[right_ghost] = quan[nolast]

def _apply_reflecting_boundary(solver, fluid, interior, left_ghost, right_ghost, noghost):
    for attr in ('rho', 'pre', 'specific_angular_momentum'):
        if not hasattr(fluid, attr):
            continue
        quan = getattr(fluid, attr)
        quan[left_ghost] = quan[interior][:noghost][::-1]
        quan[right_ghost] = quan[interior][-noghost:][::-1]
    fluid.vel[left_ghost] = -fluid.vel[interior][:noghost][::-1]
    fluid.vel[right_ghost] = -fluid.vel[interior][-noghost:][::-1]

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
        'rho': par.rho_inflow,
        'vel': par.vel_inflow,
        'pre': fluid.eos.pressure(par.rho_inflow, par.temp_inflow, par.mu_inflow),
    }
    if hasattr(fluid, 'specific_angular_momentum'):
        right_state['specific_angular_momentum'] = getattr(
            par, 'specific_angular_momentum_inflow', 0.0
        )
    if hasattr(fluid, 'xHI'):
        right_state['xHI'] = getattr(par, 'hydrogen_xHI_inflow', 1.0)
    if hasattr(fluid, 'ngamma'):
        right_state['ngamma'] = solver._to_code_number_density(
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
        'rho': par.rho_outflow,
        'vel': par.vel_outflow,
        'pre': fluid.eos.pressure(par.rho_outflow, par.temp_outflow, par.mu_outflow),
    }
    if hasattr(fluid, 'specific_angular_momentum'):
        left_state['specific_angular_momentum'] = getattr(
            par, 'specific_angular_momentum_outflow', 0.0
        )
    if hasattr(fluid, 'xHI'):
        left_state['xHI'] = getattr(par, 'hydrogen_xHI_outflow', 1.0)
    if hasattr(fluid, 'ngamma'):
        left_state['ngamma'] = solver._to_code_number_density(
            getattr(par, 'hydrogen_ngamma_outflow', 0.0),
            scales,
        )
    solver._copy_boundary_state(fluid, left_ghost, left_state)
    right_state = solver._boundary_state(fluid, nolast)
    solver._copy_boundary_state(fluid, right_ghost, right_state)
