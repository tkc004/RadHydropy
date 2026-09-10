"""NFW halo and isothermal hydrostatic-gas helpers."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS
from radhydropy.gravity import nfw_potential
import radhydropy.io as rio
from radhydropy.units import CodeUnits, code_quantity_to_cgs, quantity_to_value
from radhydropy.rsim import Rsim
from basic_hydro_utils import make_initial_condition


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
        'mass': halo_mass_proper_g,
        'redshift': float(redshift),
        'overdensity': float(overdensity),
        'concentration': float(concentration),
        'critical_density': rho_critical.to(unyt.g / unyt.cm**3),
        'virial_radius': virial_radius.to(unyt.kpc),
        'scale_radius': scale_radius.to(unyt.kpc),
        'scale_density': scale_density.to(unyt.g / unyt.cm**3),
        'virial_velocity': np.sqrt(circular_velocity_squared).to(unyt.km / unyt.s),
    }


def virial_temperature(halo, mu=0.59):
    """Return the gas virial temperature using ``kT=mu mp V_vir^2/2``."""
    virial_velocity_cgs_cm_s = halo['virial_velocity'].to(unyt.cm / unyt.s)
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
    scale_radius = halo['scale_radius'].to(unyt.cm)
    x = radius_proper_cgs_cm / scale_radius
    c = halo['concentration']
    shape = np.log1p(c) - c / (1.0 + c)
    return halo['mass'] * (
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
    over the supplied spherical mesh equals ``gas_fraction * halo['mass']``.
    """
    radius_proper_cgs_cm_unyt = radius_proper_unyt.to(unyt.cm)
    boundaries_proper_cgs_cm_unyt = boundaries_proper_unyt.to(unyt.cm)
    temperature_proper_cgs_K = temperature_proper_unyt.to_value(unyt.K)
    potential = nfw_potential(
        radius_proper_cgs_cm_unyt,
        halo['scale_density'],
        halo['scale_radius'],
    ).to_value(unyt.cm**2 / unyt.s**2)
    beta = float(mu) * PROTON_MASS_CGS / (BOLTZMANN_CONSTANT_CGS * temperature_proper_cgs_K)
    shape = np.exp(-beta * (potential - potential[0]))
    shell_volume = 4.0 * np.pi / 3.0 * (
        boundaries_proper_cgs_cm_unyt[1:].value**3 - boundaries_proper_cgs_cm_unyt[:-1].value**3
    )
    gas_mass = float(gas_fraction) * halo['mass'].to_value(unyt.g)
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
    return make_initial_condition(
        config,
        boundary_proper_code=quantity_to_value(boundary_proper_unyt, code_units.length_unit),
        rho_proper_code=quantity_to_value(density_proper_cgs_g_cm3_unyt, code_units.density_unit),
        vel_proper_code=np.zeros(grid_cells),
        temp_proper_code=np.full(grid_cells, quantity_to_value(temperature_proper_unyt, code_units.temperature_unit)),
        mu_dimensionless=np.full(grid_cells, initial_condition['mu']),
    )

def read_and_plot(outfilename, config, halo, temperature_proper_unyt, figure_filename):
    """Read the evolved snapshot and plot its NFW hydrostatic residuals."""
    initial_condition = config['initial_condition']
    par = config['par']
    code_units = CodeUnits.from_mapping(par['units']['CodeUnits'])
    config['_code_units'] = code_units
    rout = Rsim(config["par"])
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    nghost = int(par['mesh']['ghost_cells'])
    boundary_cgs = code_quantity_to_cgs(
        rout.mesh.boundary_proper_code,
        code_units,
        'length_cgs_cm',
    ) * unyt.cm
    radius_proper_cgs_cm_all_unyt = spherical_cell_centers(boundary_cgs)
    first = nghost
    last = first + int(par['mesh']['grid_cells'])
    radius_proper_cgs_cm_unyt = radius_proper_cgs_cm_all_unyt[first:last]
    rho_proper_code = rout.fluid.rho_proper_code[first:last]
    vel_proper_code = rout.fluid.vel_proper_code[first:last]
    rho_expected = hydrostatic_density_profile(
        radius_proper_cgs_cm_all_unyt,
        boundary_cgs,
        halo,
        temperature_proper_unyt,
        initial_condition['mu'],
        initial_condition['gas_fraction'],
    )[first:last]
    radius_kpc = quantity_to_value(radius_proper_cgs_cm_unyt, unyt.cm) / float((1.0 * unyt.kpc).to_value(unyt.cm))
    rho_cgs = code_quantity_to_cgs(rho_proper_code, code_units, 'density_cgs_g_cm3')
    rho_expected_cgs = quantity_to_value(rho_expected, unyt.g / unyt.cm**3)
    velocity_km_s = code_quantity_to_cgs(vel_proper_code, code_units, 'velocity_cgs_cm_s') / 1.0e5

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].plot(radius_kpc, rho_expected_cgs, color='black', lw=2.0, label='analytic HSE')
    axes[0].plot(radius_kpc, rho_cgs, 'o', ms=3.0, mfc='none', label='RHD snapshot')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('r [kpc]')
    axes[0].set_ylabel(r'$\rho_{\rm gas}$ [g cm$^{-3}$]')
    axes[0].grid(True, which='both', alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(radius_kpc, velocity_km_s, color='tab:blue')
    axes[1].axhline(0.0, color='black', ls='--')
    axes[1].set_xlabel('r [kpc]')
    axes[1].set_ylabel(r'$v_r$ [km s$^{-1}$]')
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)
    return np.max(np.abs((rho_cgs - rho_expected_cgs) / rho_expected_cgs))
