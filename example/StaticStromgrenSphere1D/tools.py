"""Helper utilities for the static Stromgren sphere example."""

from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.fluid import Fluid
import radhydropy.hydrogen as rh
from radhydropy.mesh import Mesh
from radhydropy.solver import Solver
import stromgren_analytic as sa


def build_static_problem(config):
    par = SimpleNamespace(
        coordsys='spherical',
        boundcond='OpenSph',
        nogrid=config['number_of_cells'],
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
        hydrogen_collisional_ionization=False,
        hydrogen_alpha_B=config['alpha_B_coefficient'],
        hydrogen_beta=0.0 * unyt.cm**3 / unyt.s,
        hydrogen_radiation_field=False,
        hydrogen_radiation_evolution=False,
        hydrogen_ngamma_initial=0.0 / unyt.cm**3,
        hydrogen_sigma_gamma=config['sigma_gamma'],
        hydrogen_epsilon_gamma=0.0 * unyt.erg,
        radiative_transfer=True,
        radiative_transfer_method='long_characteristics',
        radiative_transfer_boundary_flux=0.0 / (unyt.cm**2 * unyt.s),
        radiative_transfer_source_photon_rate=config['source_photon_rate'],
        radiative_transfer_direction=1,
    )

    mesh = Mesh()
    mesh.boundary = np.linspace(
        0.0,
        config['boxsize'].to_value(unyt.cm),
        par.nogrid + 1,
    ) * unyt.cm
    mesh.SetUpMesh(par)

    fluid = Fluid()
    fluid.rho = (
        np.ones(par.nogrid)
        * config['hydrogen_number_density']
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


def ionization_front_position(mesh, fluid, par, neutral_fraction=0.5):
    interior = interior_slice(par)
    radius = mesh.coordinate[interior].to(unyt.kpc)
    xHI = np.asarray(fluid.xHI[interior])

    ionized = xHI <= neutral_fraction
    if not np.any(ionized):
        return 0.0 * unyt.kpc
    if np.all(ionized):
        return mesh.boundary[interior.stop].to(unyt.kpc)

    outer_ionized_index = np.where(ionized)[0][-1]
    left = outer_ionized_index
    right = outer_ionized_index + 1
    x_left = xHI[left]
    x_right = xHI[right]
    if x_right == x_left:
        return radius[left]

    weight = (neutral_fraction - x_left) / (x_right - x_left)
    return radius[left] + weight * (radius[right] - radius[left])


def append_front_history(history, mesh, fluid, par):
    history['time_Myr'].append(fluid.time.to_value(unyt.Myr))
    history['front_radius_kpc'].append(
        ionization_front_position(mesh, fluid, par).to_value(unyt.kpc)
    )


def evolve_static_chemistry(mesh, fluid, par, solver, final_time, chemistry_timestep):
    interior = interior_slice(par)
    elapsed = 0.0 * unyt.Myr
    history = {'time_Myr': [], 'front_radius_kpc': []}
    append_front_history(history, mesh, fluid, par)
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
        append_front_history(history, mesh, fluid, par)
    solver.SetBoundary(mesh, fluid, par)
    solver.ApplyRadiativeTransfer(mesh, fluid, par)
    return history


def save_plot(mesh, fluid, par, config, figure_filename):
    interior = interior_slice(par)
    radius = mesh.coordinate[interior].to(unyt.kpc)
    plot_radius_max = config.get('plot_radius_max', config['boxsize']).to_value(unyt.kpc)
    xHI = fluid.xHI[interior]
    xHII = 1.0 - xHI
    xHI_analytic = sa.neutral_fraction_profile(
        radius,
        config['hydrogen_number_density'],
        config['sigma_gamma'],
        config['alpha_B_coefficient'],
        config['source_photon_rate'],
        inner_radius=config['analytic_inner_radius'],
    )
    xHII_analytic = 1.0 - xHI_analytic
    radius_stromgren = sa.stromgren_radius(
        config['source_photon_rate'],
        config['hydrogen_number_density'],
        config['alpha_B_coefficient'],
    ).to(unyt.kpc)
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
    ax.set_xlim(0.0, plot_radius_max)
    ax.set_yscale('log')
    ax.set_ylim(plot_floor, 1.2)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False, loc='center right')
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)


def save_front_history_plot(history, config, figure_filename):
    time_Myr = np.asarray(history['time_Myr'])
    front_radius_kpc = np.asarray(history['front_radius_kpc'])
    plot_radius_max = config.get('plot_radius_max', config['boxsize']).to_value(unyt.kpc)
    time = time_Myr * unyt.Myr
    analytic_front = sa.ionization_front_radius(
        time,
        config['source_photon_rate'],
        config['hydrogen_number_density'],
        config['alpha_B_coefficient'],
    ).to_value(unyt.kpc)
    radius_stromgren = sa.stromgren_radius(
        config['source_photon_rate'],
        config['hydrogen_number_density'],
        config['alpha_B_coefficient'],
    ).to_value(unyt.kpc)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        time_Myr,
        front_radius_kpc,
        color='tab:blue',
        lw=2.0,
        label=r'$x_{\rm HI}=0.5$ numerical',
    )
    ax.plot(
        time_Myr,
        analytic_front,
        color='black',
        lw=1.8,
        ls='--',
        label=r'$R_I(t)=R_S[1-\exp(-t/\tau_r)]^{1/3}$',
    )
    ax.axhline(
        radius_stromgren,
        color='0.25',
        lw=1.2,
        ls=':',
        label=r'$R_{\rm S}=%.2f\ {\rm kpc}$' % radius_stromgren,
    )
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Ionization-front radius [kpc]')
    ax.set_xlim(0.0, time_Myr[-1])
    ax.set_ylim(0.0, plot_radius_max)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)
