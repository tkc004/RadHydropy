"""Analytic and I/O helpers for the self-gravitating n=1 polytrope."""

import numpy as np
import unyt

from radhydropy.constants import (
    BOLTZMANN_CONSTANT_CGS,
    GRAVITATIONAL_CONSTANT_CGS,
    PROTON_MASS_CGS,
)
import radhydropy.io as rio
from radhydropy.eos import EOS
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import CodeUnits, quantity_to_value


def spherical_cell_centers(boundary_proper_code):
    inner = boundary_proper_code[:-1]
    outer = boundary_proper_code[1:]
    return 0.75 * (outer**4 - inner**4) / (outer**3 - inner**3)


def polytropic_constant(radius):
    """Return K for an n=1 sphere with radius ``pi * radius``."""
    radius = radius.to(unyt.cm)
    return 2.0 * np.pi * (
        GRAVITATIONAL_CONSTANT_CGS * unyt.cm**3 / (unyt.g * unyt.s**2)
    ) * radius**2


def equilibrium_density(radius, central_density, polytropic_radius):
    """Return rho_c sin(x)/x, with x=r/a and R=pi*a."""
    radius = radius.to(unyt.cm)
    scale = polytropic_radius.to(unyt.cm)
    x = np.asarray(radius / scale, dtype=float)
    profile = np.ones_like(x)
    nonzero = x != 0.0
    profile[nonzero] = np.sin(x[nonzero]) / x[nonzero]
    return central_density.to(unyt.g / unyt.cm**3) * profile


def equilibrium_pressure(density, polytropic_k):
    return polytropic_k * density**2


def equilibrium_temperature(density, polytropic_k, mu):
    pressure = equilibrium_pressure(density, polytropic_k)
    return (
        pressure * mu * PROTON_MASS_CGS
        * (unyt.g)
        / (
            density
            * BOLTZMANN_CONSTANT_CGS
            * (unyt.erg / unyt.K)
        )
    ).to(unyt.K)


def hydrostatic_residual(radius, density, pressure, acceleration):
    """Return dP/dr + rho*g on cell centers using centered differences."""
    radius = np.asarray(radius, dtype=float)
    density = np.asarray(density, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    gradient = np.gradient(pressure, radius)
    return gradient + density * acceleration


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    result = Rsim(config['par'])
    result.par.mesh.ghost_cells = 0
    box_size = np.ones(1) * initial['boxsize']
    result.par.time_proper_code = np.ones(1) * initial['time']
    result.par.simulation.box_size = box_size
    result.par.simulation.coordinate_system = 'spherical'
    result.mesh.boundary_proper_code = np.linspace(
        quantity_to_value(initial['rmin'], code_units.length_unit),
        quantity_to_value(initial['rmax'], code_units.length_unit), grid_cells + 1)
    result.mesh.x_proper_code = spherical_cell_centers(result.mesh.boundary_proper_code)
    result.mesh.width_proper_code = np.diff(result.mesh.boundary_proper_code)
    result.mesh.area_proper_code = 4.0 * np.pi * result.mesh.boundary_proper_code[:-1]**2
    result.mesh.volume_proper_code = 4.0 * np.pi / 3.0 * np.diff(result.mesh.boundary_proper_code**3)
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, coordinate=result.mesh.x_proper_code,
        boundary=result.mesh.boundary_proper_code,
        width=result.mesh.width_proper_code, area=result.mesh.area_proper_code,
        volume=result.mesh.volume_proper_code)
    radius = np.asarray(result.mesh.x_proper_code, dtype=float) * code_units.length_unit
    density = equilibrium_density(radius, initial['central_density'], initial['polytropic_radius'])
    k_poly = polytropic_constant(initial['polytropic_radius'])
    result.fluid.rho_proper_code = quantity_to_value(density, code_units.density_unit)
    result.fluid.temp_proper_code = quantity_to_value(
        equilibrium_temperature(density, k_poly, initial['mu']), code_units.temperature_unit
    )
    result.fluid.mu = np.ones(grid_cells) * initial['mu']
    radius_fraction = radius / initial['polytropic_radius']
    result.fluid.vel_proper_code = quantity_to_value(initial['velocity_perturbation'], code_units.velocity_unit) * np.asarray(radius_fraction, dtype=float)
    result.fluid.SetUpFluid(result.par, result.mesh)
    result.solver.SetConserved(result.mesh, result.fluid, verbose=0)
    return Rsim.FromComponents(result.par, result.mesh, result.fluid, result.solver)


def read_output(filename, config):
    par = config['par']
    code_units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    result = Rsim(config["par"])
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    result.mesh.x_proper_code = spherical_cell_centers(
        result.mesh.boundary_proper_code
    )
    result.fluid.eos = EOS(
        result.par.hydrodynamics.eos_type,
        result.par.hydrodynamics.gamma,
        code_units,
    )
    result.fluid.pre_proper_code = result.fluid.eos.pressure(
        result.fluid.rho_proper_code,
        result.fluid.temp_proper_code,
        result.fluid.mu,
    )
    return result
