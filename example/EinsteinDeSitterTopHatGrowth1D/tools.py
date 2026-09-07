"""Helpers for the Einstein--de Sitter linear-growth benchmark."""

import numpy as np
import unyt

from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits, quantity_to_value
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    SUPERCOMOVING_RUNTIME_FIELDS,
)
import radhydropy.io as rio


def spherical_cell_centers(boundary):
    inner, outer = boundary[:-1], boundary[1:]
    return 0.75 * (outer**4 - inner**4) / (outer**3 - inner**3)


def growing_mode_velocity(radius, overdensity, scale_factor, hubble):
    """Supercomoving peculiar velocity for the EdS growing mode."""
    return -(scale_factor**2 * hubble * overdensity / 3.0) * np.asarray(radius)


def enclosed_mass_radius(boundary, density, cell_volume, target_mass):
    """Interpolate the radius enclosing ``target_mass`` from cell masses."""
    cumulative = np.concatenate(([0.0], np.cumsum(np.asarray(density) * cell_volume)))
    target_mass = float(np.clip(target_mass, cumulative[0], cumulative[-1]))
    return float(np.interp(target_mass, cumulative, np.asarray(boundary)))


def linear_overdensity(delta_initial, scale_factor, initial_scale_factor):
    return float(delta_initial) * float(scale_factor) / float(initial_scale_factor)


def build_initial_condition(config):
    code_units = config['_code_units']
    cosmology = config['_cosmology']
    initial_condition = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    sim = Rsim(config['par'])
    sim.par.mesh.grid_cells = grid_cells
    sim.par.mesh.ghost_cells = 0
    boxsize_code = quantity_to_value(initial_condition['boxsize'], code_units.length_unit)
    cosmic_time = float(initial_condition['cosmic_time'])
    sim.par.simulation.box_size = np.ones(1) * boxsize_code
    sim.par.simulation.coordinate_system = 'spherical'
    scale_factor = cosmology.scale_factor(cosmic_time)
    hubble = cosmology.hubble(cosmic_time)
    sim.par.tau_supercomoving_code = np.ones(1) * cosmology.supercomoving_time(cosmic_time)
    sim.par.simulation.tau_supercomoving_code = sim.par.tau_supercomoving_code
    sim.par.cosmological_expansion = True
    sim.par.supercomoving_coordinates = True
    sim.par.cosmological_gravity = True
    sim.par.selfgravity = True
    sim.par.externalgravity = False
    sim.par.cosmology = cosmology
    sim.par.cosmology_type = cosmology.type_name
    sim.par.cosmology_t_ref = cosmology.t_ref
    sim.par.cosmology_a_ref = cosmology.a_ref
    sim.par.coordinate_frame = 'comoving'
    sim.par.time_coordinate = 'supercomoving'
    sim.par.velocity_representation = 'supercomoving_peculiar'
    sim.par.density_representation = 'comoving'
    sim.par.pressure_representation = 'supercomoving'
    sim.par.temperature_representation = 'supercomoving'

    rmin_code = quantity_to_value(initial_condition['rmin'], code_units.length_unit)
    rmax_code = quantity_to_value(initial_condition['rmax'], code_units.length_unit)
    sim.mesh.boundary_comoving_code = np.linspace(
        rmin_code, rmax_code, grid_cells + 1,
    )
    sim.mesh.x_comoving_code = spherical_cell_centers(sim.mesh.boundary_comoving_code)
    sim.mesh.area_comoving_code = 4.0 * np.pi * sim.mesh.boundary_comoving_code[:-1]**2
    sim.mesh.volume_comoving_code = 4.0 * np.pi / 3.0 * (
        sim.mesh.boundary_comoving_code[1:]**3 - sim.mesh.boundary_comoving_code[:-1]**3
    )

    rho_background = cosmology.background_density(cosmic_time)
    rho_comoving = rho_background * scale_factor**3
    delta = float(initial_condition['overdensity'])
    inside = sim.mesh.x_comoving_code < float(initial_condition['top_hat_radius'])
    sim.fluid.rho_comoving_code = rho_comoving * (1.0 + delta * inside) * np.ones(grid_cells)
    sim.fluid.vel_supercomoving_code = growing_mode_velocity(
        sim.mesh.x_comoving_code, delta, scale_factor, hubble,
    )
    sim.fluid.temp_supercomoving_code = np.ones(grid_cells) * quantity_to_value(
        initial_condition['tempini'], code_units.temperature_unit,
    ) * scale_factor**2
    sim.fluid.mu = np.ones(grid_cells) * float(initial_condition['muini'])
    sim.mesh.width_comoving_code = np.diff(sim.mesh.boundary_comoving_code)
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS,
        coordinate=sim.mesh.x_comoving_code,
        boundary=sim.mesh.boundary_comoving_code,
        width=sim.mesh.width_comoving_code,
        area=sim.mesh.area_comoving_code,
        volume=sim.mesh.volume_comoving_code,
    )
    sim.fluid.tau_supercomoving_code = float(
        np.asarray(sim.par.tau_supercomoving_code, dtype=float).reshape(-1)[0]
    )
    sim.fluid.runtime_state = FluidRuntimeState.from_arrays(
        SUPERCOMOVING_RUNTIME_FIELDS,
        density=sim.fluid.rho_comoving_code,
        velocity=sim.fluid.vel_supercomoving_code,
        pressure=np.zeros_like(sim.fluid.rho_comoving_code),
        temperature=sim.fluid.temp_supercomoving_code,
        time=sim.fluid.tau_supercomoving_code,
        mu=sim.fluid.mu,
    )


    return Rsim.FromComponents(sim.par, sim.mesh, sim.fluid, sim.solver)

def read_snapshot(filename, par_config):
    units = CodeUnits.from_mapping(par_config['CodeUnits'])
    cosmology = EinsteinDeSitter.from_code_units(units)
    result = build_initial_condition({
        'par': {'mesh': {'grid_cells': 1}},
        'initial_condition': {
            'boxsize': units.length_unit, 'rmin': 0.0 * units.length_unit,
            'rmax': units.length_unit, 'cosmic_time': 1.0,
            'top_hat_radius': 0.5, 'overdensity': 0.0,
            'tempini': units.temperature_unit, 'muini': 1.0,
        },
        '_code_units': units, '_cosmology': cosmology,
    })
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    return result
