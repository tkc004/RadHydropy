"""Analytic and I/O helpers for the self-gravitating n=1 polytrope."""

import numpy as np
import unyt

from radhydropy.constants import (
    BOLTZMANN_CONSTANT_CGS,
    GRAVITATIONAL_CONSTANT_CGS,
    PROTON_MASS_CGS,
)
import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
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
    writer = InitialConditionWriter(
        par_config=config['par'], code_units=code_units,
    )
    writer.simulation.par.simulation.coordinate_system = 'spherical'
    boundary_proper_unyt = np.linspace(
        initial['radius_inner_proper'], initial['radius_outer_proper'], grid_cells + 1,
    )
    radius_proper_code = spherical_cell_centers(
        quantity_to_value(boundary_proper_unyt, code_units.length_unit)
    )
    radius_proper_cgs_cm_unyt = radius_proper_code * code_units.length_unit
    rho_proper_cgs_g_cm3_unyt = equilibrium_density(
        radius_proper_cgs_cm_unyt,
        initial['rho_central_proper'],
        initial['radius_polytropic_proper'],
    )
    k_poly_cgs = polytropic_constant(initial['radius_polytropic_proper'])
    temperature_proper_unyt = equilibrium_temperature(
        rho_proper_cgs_g_cm3_unyt, k_poly_cgs, initial['mu_dimensionless']
    )
    radius_fraction_dimensionless = radius_proper_cgs_cm_unyt / initial['radius_polytropic_proper']
    velocity_proper_code_unyt = float(initial['vel_perturbation_dimensionless']) * np.asarray(
        radius_fraction_dimensionless, dtype=float
    ) * code_units.velocity_unit
    writer.box_size = writer.radquantity(initial['box_size_proper'])
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.mesh.x_radarray = writer.radarray(radius_proper_code * code_units.length_unit)
    writer.fluid.rho_radarray = writer.radarray(rho_proper_cgs_g_cm3_unyt)
    writer.fluid.vel_radarray = writer.radarray(velocity_proper_code_unyt)
    writer.fluid.temp_radarray = writer.radarray(temperature_proper_unyt)
    writer.simulation.fluid.mu = np.full(grid_cells, float(initial['mu_dimensionless']))
    writer.simulation.par.simulation.time_proper_code = quantity_to_value(
        initial['time_proper'], code_units.time_unit
    )
    return writer


def read_output(filename, config):
    return rio.loadhdf5(config, filename)
