"""Static Stromgren sphere at constant temperature.

This benchmark keeps the gas density and temperature fixed. A central source
emits ionizing photons at a constant rate, the long-characteristic
radiative-transfer update supplies ``n_gamma``, and the hydrogen neutral
fraction is advanced with the implicit chemistry solver. Hydrodynamics,
heating, and cooling are disabled.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.fluid import Fluid
import radhydropy.hydrogen as rh
from radhydropy.mesh import Mesh
from radhydropy.solver import Solver


rundir = os.path.dirname(os.path.abspath(__file__))
figure_filename = os.path.join(rundir, 'StaticStromgrenSphere1D.jpg')

hydrogen_number_density = 1.0e-3 / unyt.cm**3
alpha_B_coefficient = 2.59e-13 * unyt.cm**3 / unyt.s
beta_coefficient = 3.1e-16 * unyt.cm**3 / unyt.s
sigma_gamma = 8.13e-18 * unyt.cm**2
source_photon_rate = 5.0e48 / unyt.s
boxsize = 20.0 * unyt.kpc
final_time = 500.0 * unyt.Myr
chemistry_timestep = 1.0 * unyt.Myr
number_of_cells = 256


def stromgren_radius():
    radius_cubed = (
        3.0
        * source_photon_rate
        / (
            4.0
            * np.pi
            * alpha_B_coefficient
            * hydrogen_number_density**2
        )
    )
    return radius_cubed**(1.0 / 3.0)


def stromgren_optical_depth():
    return (hydrogen_number_density * sigma_gamma * stromgren_radius()).to_value('')


def recombination_time():
    return (1.0 / (alpha_B_coefficient * hydrogen_number_density)).to(unyt.Myr)


def analytic_front_radius(time):
    value = 1.0 - np.exp(-(time / recombination_time()).to_value(''))
    return stromgren_radius() * value**(1.0 / 3.0)


def analytic_stromgren_neutral_fraction(radius):
    q_target = np.asarray((radius / stromgren_radius()).to_value(''), dtype=float)
    q_start = min(np.amin(q_target[q_target > 0.0]), 1.0e-5)
    q_end = np.amax(q_target)
    tau_s = stromgren_optical_depth()
    q_grid = np.linspace(q_start, q_end, 60000)
    x_grid = np.empty_like(q_grid)
    x_grid[0] = np.clip(3.0 * q_start**2 / tau_s, 1.0e-300, 1.0 - 1.0e-12)

    def derivative(q, x):
        x = np.clip(x, 1.0e-300, 1.0 - 1.0e-12)
        return x * (1.0 - x) * (tau_s * x + 2.0 / q) / (1.0 + x)

    for i in range(len(q_grid) - 1):
        q = q_grid[i]
        h = q_grid[i + 1] - q
        x = x_grid[i]
        k1 = derivative(q, x)
        k2 = derivative(q + 0.5 * h, x + 0.5 * h * k1)
        k3 = derivative(q + 0.5 * h, x + 0.5 * h * k2)
        k4 = derivative(q + h, x + h * k3)
        x_grid[i + 1] = np.clip(
            x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,
            1.0e-300,
            1.0 - 1.0e-12,
        )

    return np.interp(q_target, q_grid, x_grid)


def build_static_problem():
    par = SimpleNamespace(
        coordsys='spherical',
        boundcond='OpenSph',
        nogrid=number_of_cells,
        noghost=2,
        area=1.0 * unyt.cm**2,
        hydrogen_chemistry=True,
        hydrogen_mass_fraction=1.0,
        hydrogen_xHI_initial=1.0,
        hydrogen_xHI_inflow=1.0,
        hydrogen_xHI_outflow=1.0,
        hydrogen_source_CFL=1.0,
        hydrogen_update_mu=False,
        hydrogen_thermal_coupling=False,
        hydrogen_recombination=True,
        hydrogen_collisional_ionization=True,
        hydrogen_alpha_B=alpha_B_coefficient,
        hydrogen_beta=beta_coefficient,
        hydrogen_radiation_field=False,
        hydrogen_radiation_evolution=False,
        hydrogen_ngamma_initial=0.0 / unyt.cm**3,
        hydrogen_sigma_gamma=sigma_gamma,
        hydrogen_epsilon_gamma=0.0 * unyt.erg,
        radiative_transfer=True,
        radiative_transfer_method='long_characteristics',
        radiative_transfer_boundary_flux=0.0 / (unyt.cm**2 * unyt.s),
        radiative_transfer_source_photon_rate=source_photon_rate,
        radiative_transfer_direction=1,
    )

    mesh = Mesh()
    mesh.boundary = np.linspace(
        0.0,
        boxsize.to_value(unyt.cm),
        par.nogrid + 1,
    ) * unyt.cm
    mesh.SetUpMesh(par)

    fluid = Fluid()
    fluid.rho = (
        np.ones(par.nogrid)
        * hydrogen_number_density
        * unyt.mp
    ).to(unyt.g / unyt.cm**3)
    fluid.vel = np.zeros(par.nogrid) * unyt.cm / unyt.s
    fluid.temp = np.ones(par.nogrid) * 1.0e4 * unyt.K
    fluid.mu = np.ones(par.nogrid)
    fluid.xHI = np.ones(par.nogrid)
    fluid.SetUpFluid(par)
    fluid.SetFluidTime(0.0 * unyt.Myr)

    solver = Solver()
    solver.SetBoundary(mesh, fluid, par)
    solver.ApplyRadiativeTransfer(mesh, fluid, par)
    return par, mesh, fluid, solver


def interior_slice(par):
    first = par.noghost
    return slice(first, first + par.nogrid)


def evolve_static_chemistry(mesh, fluid, par, solver):
    interior = interior_slice(par)
    elapsed = 0.0 * unyt.Myr
    while elapsed < final_time:
        dt = min(chemistry_timestep, final_time - elapsed)
        solver.SetBoundary(mesh, fluid, par)
        solver.ApplyRadiativeTransfer(mesh, fluid, par)
        fluid.xHI[interior] = rh.hydrogen_neutral_fraction_implicit_update(
            fluid.rho[interior],
            fluid.temp[interior],
            fluid.xHI[interior],
            dt,
            hydrogen_mass_fraction=par.hydrogen_mass_fraction,
            recombination=par.hydrogen_recombination,
            collisional_ionization=par.hydrogen_collisional_ionization,
            ngamma=fluid.ngamma[interior],
            sigma_gamma=par.hydrogen_sigma_gamma,
            recombination_coefficient=par.hydrogen_alpha_B,
            ionization_coefficient=par.hydrogen_beta,
        )
        elapsed += dt
        fluid.time = elapsed
    solver.SetBoundary(mesh, fluid, par)
    solver.ApplyRadiativeTransfer(mesh, fluid, par)


def save_plot(mesh, fluid, par):
    interior = interior_slice(par)
    radius = mesh.coordinate[interior].to(unyt.kpc)
    xHI = fluid.xHI[interior]
    xHII = 1.0 - xHI
    xHI_analytic = analytic_stromgren_neutral_fraction(radius)
    xHII_analytic = 1.0 - xHI_analytic
    radius_stromgren = stromgren_radius().to(unyt.kpc)
    plot_floor = 1.0e-6

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        radius.to_value(unyt.kpc),
        np.clip(xHI, plot_floor, 1.0),
        color='tab:blue',
        lw=2.0,
        label=r'$x_{\rm HI}$ numerical',
    )
    ax.plot(
        radius.to_value(unyt.kpc),
        np.clip(xHII, plot_floor, 1.0),
        color='tab:red',
        lw=2.0,
        label=r'$x_{\rm HII}$ numerical',
    )
    ax.plot(
        radius.to_value(unyt.kpc),
        np.clip(xHI_analytic, plot_floor, 1.0),
        color='tab:blue',
        lw=1.6,
        ls='--',
        label=r'$x_{\rm HI}$ analytic',
    )
    ax.plot(
        radius.to_value(unyt.kpc),
        np.clip(xHII_analytic, plot_floor, 1.0),
        color='tab:red',
        lw=1.6,
        ls='--',
        label=r'$x_{\rm HII}$ analytic',
    )
    ax.axvline(
        radius_stromgren.to_value(unyt.kpc),
        color='black',
        lw=2.0,
        label=r'$R_{\rm S}=%.2f\ {\rm kpc}$' % radius_stromgren.to_value(unyt.kpc),
    )
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Hydrogen fraction')
    ax.set_xlim(0.0, boxsize.to_value(unyt.kpc))
    ax.set_yscale('log')
    ax.set_ylim(plot_floor, 1.2)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False, loc='center right')
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)


def main():
    par, mesh, fluid, solver = build_static_problem()
    evolve_static_chemistry(mesh, fluid, par, solver)
    save_plot(mesh, fluid, par)
    print('time = %s' % fluid.time)
    print('recombination time = %s' % recombination_time())
    print('stromgren radius = %s' % stromgren_radius().to(unyt.kpc))
    print('analytic front radius = %s' % analytic_front_radius(final_time).to(unyt.kpc))
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    main()
