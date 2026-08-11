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
    mass = halo_mass.to(unyt.g)
    h0_cgs = h0.to(1.0 / unyt.s)
    rho_critical = 3.0 * h0_cgs**2 / (8.0 * np.pi * GRAVITATIONAL_CONSTANT)
    virial_radius = (
        3.0 * mass / (4.0 * np.pi * overdensity * rho_critical)
    ) ** (1.0 / 3.0)
    scale_radius = virial_radius / float(concentration)
    shape = np.log1p(concentration) - concentration / (1.0 + concentration)
    scale_density = mass / (4.0 * np.pi * scale_radius**3 * shape)
    circular_velocity_squared = GRAVITATIONAL_CONSTANT * mass / virial_radius
    return {
        'mass': mass,
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
    velocity = halo['virial_velocity'].to(unyt.cm / unyt.s)
    temperature = (
        float(mu) * PROTON_MASS_CGS * velocity.value**2
        / (2.0 * BOLTZMANN_CONSTANT_CGS)
    ) * unyt.K
    return temperature


def spherical_cell_centers(boundary):
    """Return volume-weighted radial centers for spherical cells."""
    inner = boundary[:-1]
    outer = boundary[1:]
    denominator = outer**3 - inner**3
    return 0.75 * (outer**4 - inner**4) / denominator


def nfw_enclosed_mass(radius, halo):
    """Return the NFW dark-matter mass enclosed by ``radius``."""
    radius = radius.to(unyt.cm)
    scale_radius = halo['scale_radius'].to(unyt.cm)
    x = radius / scale_radius
    c = halo['concentration']
    shape = np.log1p(c) - c / (1.0 + c)
    return halo['mass'] * (
        (np.log1p(x) - x / (1.0 + x)) / shape
    )


def hydrostatic_density_profile(
    radius,
    boundaries,
    halo,
    temperature,
    mu,
    gas_fraction=DEFAULT_BARYON_FRACTION,
):
    """Return an isothermal gas profile normalized to a chosen gas mass.

    The profile solves ``dP/dr = -rho G M(<r)/r^2`` in the NFW potential,
    with ``P=rho*k*T/(mu*m_p)``. Its normalization is selected so the gas mass
    over the supplied spherical mesh equals ``gas_fraction * halo['mass']``.
    """
    radius = radius.to(unyt.cm)
    boundaries = boundaries.to(unyt.cm)
    temperature_K = temperature.to_value(unyt.K)
    potential = nfw_potential(
        radius,
        halo['scale_density'],
        halo['scale_radius'],
    ).to_value(unyt.cm**2 / unyt.s**2)
    beta = float(mu) * PROTON_MASS_CGS / (BOLTZMANN_CONSTANT_CGS * temperature_K)
    shape = np.exp(-beta * (potential - potential[0]))
    shell_volume = 4.0 * np.pi / 3.0 * (
        boundaries[1:].value**3 - boundaries[:-1].value**3
    )
    gas_mass = float(gas_fraction) * halo['mass'].to_value(unyt.g)
    normalization = gas_mass / np.sum(shape * shell_volume)
    return normalization * shape * (unyt.g / unyt.cm**3)


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class Simwrap:
    def __init__(self, icparams, code_units=None):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.CodeUnits = code_units
        self.par.unit_system = code_units.unit_system
        self.par.nogrid = int(icparams['nogrid'])
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = np.ones(1) * icparams['boxsize']
        self.par.time = np.ones(1) * icparams['time']
        self.mesh.boundary = np.linspace(
            icparams['rmin'],
            icparams['rmax'],
            self.par.nogrid + 1,
        )
        self.mesh.coordinate = spherical_cell_centers(self.mesh.boundary)
        self.mesh.area = 4.0 * np.pi * self.mesh.boundary[:-1]**2
        self.mesh.vol = 4.0 * np.pi / 3.0 * (
            self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3
        )
        halo = nfw_halo_parameters(
            icparams['halo_mass'],
            icparams['concentration'],
            icparams['redshift'],
            icparams['overdensity'],
            icparams['h0'],
        )
        temperature = virial_temperature(halo, icparams['mu'])
        self.fluid.temp = np.ones(self.par.nogrid) * temperature
        self.fluid.mu = np.ones(self.par.nogrid) * icparams['mu']
        self.fluid.vel = np.zeros(self.par.nogrid) * unyt.cm / unyt.s
        self.fluid.rho = hydrostatic_density_profile(
            self.mesh.coordinate,
            self.mesh.boundary,
            halo,
            temperature,
            icparams['mu'],
            icparams['gas_fraction'],
        )


def read_and_plot(outfilename, icparams, runparams, halo, temperature, figure_filename):
    """Read the evolved snapshot and plot its NFW hydrostatic residuals."""
    code_units = CodeUnits.from_mapping(runparams['CodeUnits'])
    rout = Simwrap(icparams, code_units=code_units)
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    nghost = int(runparams.get('noghost', 0))
    boundary_cgs = code_quantity_to_cgs(
        rout.mesh.boundary,
        code_units,
        'length_cm',
    ) * unyt.cm
    radius_all = spherical_cell_centers(boundary_cgs)
    radius = radius_all[nghost:-nghost]
    rho = rout.fluid.rho[nghost:-nghost]
    velocity = rout.fluid.vel[nghost:-nghost]
    rho_expected = hydrostatic_density_profile(
        radius_all,
        boundary_cgs,
        halo,
        temperature,
        icparams['mu'],
        icparams['gas_fraction'],
    )[nghost:-nghost]
    radius_kpc = quantity_to_value(radius, unyt.cm) / float((1.0 * unyt.kpc).to_value(unyt.cm))
    rho_cgs = code_quantity_to_cgs(rho, code_units, 'density_g_cm3')
    rho_expected_cgs = quantity_to_value(rho_expected, unyt.g / unyt.cm**3)
    velocity_km_s = code_quantity_to_cgs(velocity, code_units, 'velocity_cm_s') / 1.0e5

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
