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


def polytropic_constant(radius_polytropic_proper_unyt):
    """Return K for an n=1 sphere with radius ``pi * a``."""
    radius_polytropic_cgs_cm_unyt = radius_polytropic_proper_unyt.to(unyt.cm)
    return 2.0 * np.pi * (
        GRAVITATIONAL_CONSTANT_CGS * unyt.cm**3 / (unyt.g * unyt.s**2)
    ) * radius_polytropic_cgs_cm_unyt**2


def equilibrium_density(
    radius_proper_cgs_cm_unyt,
    rho_central_proper_unyt,
    radius_polytropic_proper_unyt,
):
    """Return rho_c sin(x)/x, with x=r/a and R=pi*a."""
    radius_proper_cgs_cm_unyt = radius_proper_cgs_cm_unyt.to(unyt.cm)
    radius_polytropic_cgs_cm_unyt = radius_polytropic_proper_unyt.to(unyt.cm)
    radius_dimensionless = np.asarray(
        radius_proper_cgs_cm_unyt / radius_polytropic_cgs_cm_unyt, dtype=float
    )
    profile_dimensionless = np.ones_like(radius_dimensionless)
    nonzero = radius_dimensionless != 0.0
    profile_dimensionless[nonzero] = (
        np.sin(radius_dimensionless[nonzero]) / radius_dimensionless[nonzero]
    )
    return rho_central_proper_unyt.to(unyt.g / unyt.cm**3) * profile_dimensionless


def equilibrium_pressure(rho_proper_cgs_g_cm3_unyt, polytropic_k_cgs):
    return polytropic_k_cgs * rho_proper_cgs_g_cm3_unyt**2


def equilibrium_temperature(
    rho_proper_cgs_g_cm3_unyt, polytropic_k_cgs, mu_dimensionless
):
    pre_proper_cgs_erg_cm3_unyt = equilibrium_pressure(
        rho_proper_cgs_g_cm3_unyt, polytropic_k_cgs
    )
    return (
        pre_proper_cgs_erg_cm3_unyt * mu_dimensionless * PROTON_MASS_CGS
        * (unyt.g)
        / (
            rho_proper_cgs_g_cm3_unyt
            * BOLTZMANN_CONSTANT_CGS
            * (unyt.erg / unyt.K)
        )
    ).to(unyt.K)


def hydrostatic_residual(
    radius_proper_cgs_cm,
    rho_proper_cgs_g_cm3,
    pre_proper_cgs_erg_cm3,
    acceleration_proper_cgs_cm_s2,
):
    """Return dP/dr + rho*g on cell centers using centered differences."""
    radius_proper_cgs_cm = np.asarray(radius_proper_cgs_cm, dtype=float)
    rho_proper_cgs_g_cm3 = np.asarray(rho_proper_cgs_g_cm3, dtype=float)
    pre_proper_cgs_erg_cm3 = np.asarray(pre_proper_cgs_erg_cm3, dtype=float)
    acceleration_proper_cgs_cm_s2 = np.asarray(acceleration_proper_cgs_cm_s2, dtype=float)
    gradient_proper_cgs_erg_cm4 = np.gradient(
        pre_proper_cgs_erg_cm3, radius_proper_cgs_cm
    )
    return gradient_proper_cgs_erg_cm4 + rho_proper_cgs_g_cm3 * acceleration_proper_cgs_cm_s2


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    result = Rsim(config['par'])
    result.par.mesh.ghost_cells = 0
    box_size_proper_code = np.ones(1) * quantity_to_value(
        initial['box_size_proper'], code_units.length_unit
    )
    result.par.time_proper_code = np.ones(1) * quantity_to_value(
        initial['time_proper'], code_units.time_unit
    )
    result.par.simulation.box_size_proper_code = box_size_proper_code
    result.par.simulation.coordinate_system = 'spherical'
    result.mesh.boundary_proper_code = np.linspace(
        quantity_to_value(initial['radius_inner_proper'], code_units.length_unit),
        quantity_to_value(initial['radius_outer_proper'], code_units.length_unit),
        grid_cells + 1,
    )
    result.mesh.x_proper_code = spherical_cell_centers(result.mesh.boundary_proper_code)
    result.mesh.width_proper_code = np.diff(result.mesh.boundary_proper_code)
    result.mesh.area_proper_code = 4.0 * np.pi * result.mesh.boundary_proper_code[:-1]**2
    result.mesh.volume_proper_code = 4.0 * np.pi / 3.0 * np.diff(result.mesh.boundary_proper_code**3)
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS, x_proper_code=result.mesh.x_proper_code,
        boundary_proper_code=result.mesh.boundary_proper_code,
        width_proper_code=result.mesh.width_proper_code, area_proper_code=result.mesh.area_proper_code,
        volume_proper_code=result.mesh.volume_proper_code)
    radius_proper_cgs_cm_unyt = np.asarray(result.mesh.x_proper_code, dtype=float) * code_units.length_unit
    rho_proper_cgs_g_cm3_unyt = equilibrium_density(
        radius_proper_cgs_cm_unyt,
        initial['rho_central_proper'],
        initial['radius_polytropic_proper'],
    )
    k_poly_cgs = polytropic_constant(initial['radius_polytropic_proper'])
    result.fluid.rho_proper_code = quantity_to_value(rho_proper_cgs_g_cm3_unyt, code_units.density_unit)
    result.fluid.temp_proper_code = quantity_to_value(
        equilibrium_temperature(rho_proper_cgs_g_cm3_unyt, k_poly_cgs, initial['mu_dimensionless']),
        code_units.temperature_unit,
    )
    result.fluid.mu = np.ones(grid_cells) * initial['mu_dimensionless']
    radius_fraction_dimensionless = radius_proper_cgs_cm_unyt / initial['radius_polytropic_proper']
    result.fluid.vel_proper_code = float(
        initial['vel_perturbation_dimensionless']
    ) * np.asarray(radius_fraction_dimensionless, dtype=float)
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
