"""Helpers for the uniform-density spherical self-gravity diagnostic."""

import numpy as np
import unyt
from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS
import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import CodeUnits, quantity_to_value


def spherical_cell_centers(boundary_proper_code):
    """Return volume-weighted centers for spherical cells."""
    inner = boundary_proper_code[:-1]
    outer = boundary_proper_code[1:]
    denominator = outer**3 - inner**3
    return 0.75 * (outer**4 - inner**4) / denominator


def uniform_sphere_acceleration(radius, rho0):
    """Return the analytic interior field of a uniform-density sphere."""
    radius = radius.to(unyt.cm)
    rho0 = rho0.to(unyt.g / unyt.cm**3)
    return (
        -4.0 * np.pi / 3.0
        * (GRAVITATIONAL_CONSTANT_CGS * unyt.cm**3 / (unyt.g * unyt.s**2))
        * rho0
        * radius
    ).to(unyt.cm / unyt.s**2)


def build_initial_condition(config):
    code_unit_system = CodeUnits.from_mapping(
        config['par']['units']['CodeUnits']
    )
    initial_condition = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    sim = Rsim(config['par'])
    sim.par.mesh.grid_cells = grid_cells
    sim.par.mesh.ghost_cells = 0
    sim.par.simulation.coordinate_system = initial_condition['coordsys']
    sim.par.simulation.box_size_proper_code = np.ones(1) * quantity_to_value(initial_condition['box_size_proper'], code_unit_system.length_unit)
    sim.par.simulation.time_proper_code = quantity_to_value(initial_condition['time_proper'], code_unit_system.time_unit)

    boundary_proper_code = np.linspace(
        quantity_to_value(initial_condition['rmin'], code_unit_system.length_unit),
        quantity_to_value(initial_condition['rmax'], code_unit_system.length_unit),
        grid_cells + 1,
    )
    sim.mesh.boundary_proper_code = boundary_proper_code
    sim.mesh.x_proper_code = spherical_cell_centers(boundary_proper_code)
    sim.mesh.width_proper_code = np.diff(boundary_proper_code)
    sim.mesh.area_proper_code = 4.0 * np.pi * boundary_proper_code[:-1]**2
    sim.mesh.volume_proper_code = 4.0 * np.pi / 3.0 * (
        boundary_proper_code[1:]**3 - boundary_proper_code[:-1]**3
    )
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=sim.mesh.x_proper_code,
        boundary_proper_code=sim.mesh.boundary_proper_code,
        width_proper_code=sim.mesh.width_proper_code,
        area_proper_code=sim.mesh.area_proper_code,
        volume_proper_code=sim.mesh.volume_proper_code,
    )

    sim.fluid.rho_proper_code = np.ones(grid_cells) * quantity_to_value(initial_condition['rho0'], code_unit_system.density_unit)
    sim.fluid.temp_proper_code = np.ones(grid_cells) * quantity_to_value(initial_condition['tempini'], code_unit_system.temperature_unit)
    sim.fluid.mu = np.ones(grid_cells) * float(initial_condition['muini'])
    sim.fluid.vel_proper_code = np.zeros(grid_cells, dtype=float)
    sim.fluid.SetUpFluid(sim.par, sim.mesh)
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)

    return Rsim.FromComponents(sim.par, sim.mesh, sim.fluid, sim.solver)

def load_output_state(filename, config):
    result = build_initial_condition(
        config,
    )
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    return result
