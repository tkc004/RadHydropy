"""Helpers for the Einstein--de Sitter dark-matter shell growth test."""

import numpy as np

from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.units import CodeUnits, _gravitational_constant_code


def load_units(runparams):
    return CodeUnits.from_mapping(runparams['units']['CodeUnits'])


def volume_midpoint_boundaries(rmin, rmax, number):
    boundaries = np.linspace(rmin**3, rmax**3, number + 1)
    return boundaries**(1.0 / 3.0)


def make_shells(icparams, code_units, cosmology, overdensity=None):
    default_number = int(icparams.get('number_of_shells', 2))
    number_inner = int(icparams.get('number_of_inner_shells', default_number // 2))
    number_outer = int(icparams.get('number_of_outer_shells', number_inner))
    rmin = float(icparams['inner_radius'])
    rmax = float(icparams['outer_radius'])
    top_hat_radius = float(icparams['top_hat_radius'])
    inner_boundaries = volume_midpoint_boundaries(rmin, top_hat_radius, number_inner)
    outer_boundaries = volume_midpoint_boundaries(top_hat_radius, rmax, number_outer)
    boundaries = np.concatenate((inner_boundaries, outer_boundaries[1:]))
    radius = ((boundaries[:-1]**3 + boundaries[1:]**3) / 2.0)**(1.0 / 3.0)
    volume = 4.0 * np.pi / 3.0 * np.diff(boundaries**3)
    cosmic_time = float(icparams['cosmic_time'])
    scale_factor = float(cosmology.scale_factor(cosmic_time))
    rho_comoving = float(cosmology.background_density(cosmic_time)) * scale_factor**3
    delta = float(icparams['overdensity'] if overdensity is None else overdensity)
    inside = radius < float(icparams['top_hat_radius'])
    mass = rho_comoving * volume * (1.0 + delta * inside)
    hubble = float(cosmology.hubble(cosmic_time))
    velocity = np.zeros_like(radius)
    velocity[inside] = -scale_factor**2 * hubble * delta * radius[inside] / 3.0
    velocity[~inside] = (
        -scale_factor**2 * hubble * delta * top_hat_radius**3
        / (3.0 * radius[~inside]**2)
    )
    return DarkMatterShells(
        radius, velocity, mass,
        softening=float(icparams['softening']),
        code_units=code_units,
    ), boundaries


def lagrangian_boundary_acceleration(radius, enclosed_mass, background_density,
                                     scale_factor, code_units):
    g_code = _gravitational_constant_code(code_units)
    background_mass = 4.0 * np.pi / 3.0 * background_density * radius**3
    return -g_code * scale_factor * (enclosed_mass - background_mass) / radius**2


def overdensity_inside(radius, target_mass, background_density):
    background_mass = 4.0 * np.pi / 3.0 * background_density * radius**3
    return float(target_mass / background_mass - 1.0)


def step_lagrangian_boundary(radius, velocity, dt, enclosed_mass,
                             background_density_start, background_density_end,
                             scale_factor_start, scale_factor_end, code_units):
    acceleration = lagrangian_boundary_acceleration(
        radius, enclosed_mass, background_density_start, scale_factor_start,
        code_units,
    )
    velocity_half = velocity + 0.5 * dt * acceleration
    radius_new = radius + dt * velocity_half
    acceleration_new = lagrangian_boundary_acceleration(
        radius_new, enclosed_mass, background_density_end, scale_factor_end,
        code_units,
    )
    return radius_new, velocity_half + 0.5 * dt * acceleration_new
