"""NFW halo and isothermal hydrostatic-gas helpers."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS
from radhydropy.gravity import nfw_potential
import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, code_quantity_to_cgs, quantity_to_value


GRAVITATIONAL_CONSTANT = unyt.physical_constants.gravitational_constant
DEFAULT_H0 = 70.0 * unyt.km / unyt.s / unyt.Mpc
DEFAULT_BARYON_FRACTION = 0.157


def nfw_halo_parameters(
    halo_mass,
    concentration=10.0,
    redshift=0.0,
    overdensity=200.0,
    h0=DEFAULT_H0,
):
    """Return ``R_delta``, ``r_s``, ``rho_s``, and ``T_vir`` for an NFW halo."""
    halo_mass_proper_g = halo_mass.to(unyt.g)
    h0_cgs = h0.to(1.0 / unyt.s)
    rho_critical = 3.0 * h0_cgs**2 / (8.0 * np.pi * GRAVITATIONAL_CONSTANT)
    virial_radius = (
        3.0 * halo_mass_proper_g / (4.0 * np.pi * overdensity * rho_critical)
    ) ** (1.0 / 3.0)
    scale_radius = virial_radius / float(concentration)
    shape = np.log1p(concentration) - concentration / (1.0 + concentration)
    scale_density = halo_mass_proper_g / (4.0 * np.pi * scale_radius**3 * shape)
    circular_velocity_squared = GRAVITATIONAL_CONSTANT * halo_mass_proper_g / virial_radius
    return {
        'mass_halo_proper_g_unyt': halo_mass_proper_g,
        'redshift': float(redshift),
        'overdensity': float(overdensity),
        'concentration': float(concentration),
        'rho_critical_cgs_g_cm3_unyt': rho_critical.to(unyt.g / unyt.cm**3),
        'radius_virial_proper_kpc_unyt': virial_radius.to(unyt.kpc),
        'radius_scale_proper_kpc_unyt': scale_radius.to(unyt.kpc),
        'rho_scale_cgs_g_cm3_unyt': scale_density.to(unyt.g / unyt.cm**3),
        'vel_virial_proper_km_s_unyt': np.sqrt(circular_velocity_squared).to(unyt.km / unyt.s),
    }


def virial_temperature(halo, mu=0.59):
    """Return the gas virial temperature using ``kT=mu mp V_vir^2/2``."""
    virial_velocity_cgs_cm_s = halo['vel_virial_proper_km_s_unyt'].to(unyt.cm / unyt.s)
    virial_temperature_proper_K = (
        float(mu) * PROTON_MASS_CGS * virial_velocity_cgs_cm_s.value**2
        / (2.0 * BOLTZMANN_CONSTANT_CGS)
    ) * unyt.K
    return virial_temperature_proper_K


def spherical_cell_centers(boundary_proper_code):
    """Return volume-weighted radial centers for spherical cells."""
    inner = boundary_proper_code[:-1]
    outer = boundary_proper_code[1:]
    denominator = outer**3 - inner**3
    return 0.75 * (outer**4 - inner**4) / denominator


def nfw_enclosed_mass(radius_proper_unyt, halo):
    """Return the NFW dark-matter mass enclosed by ``radius``."""
    radius_proper_cgs_cm = radius_proper_unyt.to(unyt.cm)
    radius_scale_proper_cgs_cm_unyt = halo['radius_scale_proper_kpc_unyt'].to(unyt.cm)
    x = radius_proper_cgs_cm / radius_scale_proper_cgs_cm_unyt
    c = halo['concentration']
    shape = np.log1p(c) - c / (1.0 + c)
    return halo['mass_halo_proper_g_unyt'] * (
        (np.log1p(x) - x / (1.0 + x)) / shape
    )


def hydrostatic_density_profile(
    radius_proper_unyt,
    boundaries_proper_unyt,
    halo,
    temperature_proper_unyt,
    mu,
    gas_fraction=DEFAULT_BARYON_FRACTION,
):
    """Return an isothermal gas profile normalized to a chosen gas mass.

    The profile solves ``dP/dr = -rho G M(<r)/r^2`` in the NFW potential,
    with ``P=rho*k*T/(mu*m_p)``. Its normalization is selected so the gas mass
    over the supplied spherical mesh equals ``gas_fraction * halo['mass_halo_proper_g_unyt']``.
    """
    radius_proper_cgs_cm_unyt = radius_proper_unyt.to(unyt.cm)
    boundaries_proper_cgs_cm_unyt = boundaries_proper_unyt.to(unyt.cm)
    temperature_proper_cgs_K = temperature_proper_unyt.to_value(unyt.K)
    potential = nfw_potential(
        radius_proper_cgs_cm_unyt,
        halo['rho_scale_cgs_g_cm3_unyt'],
        halo['radius_scale_proper_kpc_unyt'],
    ).to_value(unyt.cm**2 / unyt.s**2)
    beta = float(mu) * PROTON_MASS_CGS / (BOLTZMANN_CONSTANT_CGS * temperature_proper_cgs_K)
    shape = np.exp(-beta * (potential - potential[0]))
    shell_volume = 4.0 * np.pi / 3.0 * (
        boundaries_proper_cgs_cm_unyt[1:].value**3 - boundaries_proper_cgs_cm_unyt[:-1].value**3
    )
    gas_mass = float(gas_fraction) * halo['mass_halo_proper_g_unyt'].to_value(unyt.g)
    normalization = gas_mass / np.sum(shape * shell_volume)
    return normalization * shape * (unyt.g / unyt.cm**3)


def build_initial_condition(config):
    initial_condition = config['initial_condition']
    code_units = config['_code_units']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    box_size_proper_unyt = initial_condition['box_size_proper']
    time_proper_unyt = initial_condition['time_proper']
    radius_inner_proper_unyt = initial_condition['radius_inner_proper']
    radius_outer_proper_unyt = initial_condition['radius_outer_proper']
    boundary_proper_unyt = (
        np.linspace(0.0, 1.0, grid_cells + 1)
        * (radius_outer_proper_unyt - radius_inner_proper_unyt)
        + radius_inner_proper_unyt
    )
    coordinate_proper_unyt = spherical_cell_centers(boundary_proper_unyt)
    halo = nfw_halo_parameters(
        initial_condition['halo_mass'],
        initial_condition['concentration'],
        initial_condition['redshift'],
        initial_condition['overdensity'],
        initial_condition['h0'],
    )
    temperature_proper_unyt = virial_temperature(halo, initial_condition['mu'])
    density_proper_cgs_g_cm3_unyt = hydrostatic_density_profile(
        coordinate_proper_unyt,
        boundary_proper_unyt,
        halo,
        temperature_proper_unyt,
        initial_condition['mu'],
        initial_condition['gas_fraction'],
    )
    writer = InitialConditionWriter(par_config=config['par'], code_units=code_units)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.mesh.x_radarray = writer.radarray(coordinate_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(density_proper_cgs_g_cm3_unyt)
    writer.fluid.vel_radarray = writer.radarray(np.zeros(grid_cells) * code_units.velocity_unit)
    writer.fluid.temp_radarray = writer.radarray(np.ones(grid_cells) * temperature_proper_unyt)
    writer.simulation.fluid.mu = np.full(grid_cells, initial_condition['mu'])
    return writer

def read_and_plot(outfilename, config, halo, temperature_proper_unyt, figure_filename):
    """Read the evolved snapshot and plot its NFW hydrostatic residuals."""
    initial_condition = config['initial_condition']
    par = config['par']
    code_units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    config['_code_units'] = code_units
    rout = rio.loadhdf5(config, outfilename)
    nghost = int(par['mesh']['ghost_cells'])
    boundary_proper_cgs_cm_unyt = code_quantity_to_cgs(
        np.asarray(rout.mesh.boundary_radarray.value),
        code_units,
        'length_cgs_cm',
    ) * unyt.cm
    radius_proper_cgs_cm_all_unyt = spherical_cell_centers(boundary_proper_cgs_cm_unyt)
    first = nghost
    last = first + int(par['mesh']['grid_cells'])
    radius_proper_cgs_cm_unyt = radius_proper_cgs_cm_all_unyt[first:last]
    rho_values = np.asarray(rout.fluid.rho_radarray.value)
    vel_values = np.asarray(rout.fluid.vel_radarray.value)
    if rho_values.size == int(par['mesh']['grid_cells']):
        first, last = 0, rho_values.size
    rho_proper_code = rho_values[first:last]
    vel_proper_code = vel_values[first:last]
    rho_expected_proper_cgs_g_cm3_unyt = hydrostatic_density_profile(
        radius_proper_cgs_cm_all_unyt,
        boundary_proper_cgs_cm_unyt,
        halo,
        temperature_proper_unyt,
        initial_condition['mu'],
        initial_condition['gas_fraction'],
    )[first:last]
    radius_proper_kpc = quantity_to_value(radius_proper_cgs_cm_unyt, unyt.cm) / float((1.0 * unyt.kpc).to_value(unyt.cm))
    rho_proper_cgs_g_cm3 = code_quantity_to_cgs(rho_proper_code, code_units, 'density_cgs_g_cm3')
    rho_expected_proper_cgs_g_cm3 = quantity_to_value(
        rho_expected_proper_cgs_g_cm3_unyt, unyt.g / unyt.cm**3
    )
    vel_proper_km_s = code_quantity_to_cgs(vel_proper_code, code_units, 'velocity_cgs_cm_s') / 1.0e5

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].plot(radius_proper_kpc, rho_expected_proper_cgs_g_cm3, color='black', lw=2.0, label='analytic HSE')
    axes[0].plot(radius_proper_kpc, rho_proper_cgs_g_cm3, 'o', ms=3.0, mfc='none', label='RHD snapshot')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('r [kpc]')
    axes[0].set_ylabel(r'$\rho_{\rm gas}$ [g cm$^{-3}$]')
    axes[0].grid(True, which='both', alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(radius_proper_kpc, vel_proper_km_s, color='tab:blue')
    axes[1].axhline(0.0, color='black', ls='--')
    axes[1].set_xlabel('r [kpc]')
    axes[1].set_ylabel(r'$v_r$ [km s$^{-1}$]')
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)
    return np.max(np.abs((rho_proper_cgs_g_cm3 - rho_expected_proper_cgs_g_cm3) / rho_expected_proper_cgs_g_cm3))
