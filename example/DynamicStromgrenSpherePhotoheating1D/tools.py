"""Utilities for the dynamic photoheated Stromgren sphere example."""

import os
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
from radhydropy.mesh import Mesh
from radhydropy.solver import Solver


def build_problem(config):
    par = SimpleNamespace(
        coordsys='spherical',
        boundcond='OpenSph',
        nogrid=config['number_of_cells'],
        noghost=2,
        area=1.0 * unyt.cm**2,
        EOStype='polytropic',
        gamma=5.0 / 3.0,
        CFL=config['hydro_cfl'],
        order=0,
        dtmin=1.0e-8 * unyt.Myr,
        dtmax=config['hydro_timestep_max'],
        hydrogen_chemistry=True,
        hydrogen_mass_fraction=1.0,
        hydrogen_xHI_initial=1.0,
        hydrogen_xHI_inflow=1.0,
        hydrogen_xHI_outflow=1.0,
        hydrogen_source_CFL=config['source_cfl'],
        hydrogen_source_dtmin=config['source_timestep_min'],
        hydrogen_update_mu=True,
        hydrogen_thermal_coupling=True,
        hydrogen_recombination=True,
        hydrogen_collisional_ionization=False,
        hydrogen_alpha_B=config['alpha_B_coefficient'],
        hydrogen_beta=0.0 * unyt.cm**3 / unyt.s,
        hydrogen_radiation_field=False,
        hydrogen_radiation_evolution=False,
        hydrogen_ngamma_initial=0.0 / unyt.cm**3,
        hydrogen_sigma_gamma=config['sigma_gamma'],
        hydrogen_epsilon_gamma=config['epsilon_gamma'],
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
    fluid.eos = EOS(par.EOStype, par.gamma)
    fluid.rho = (
        np.ones(par.nogrid)
        * config['hydrogen_number_density']
        * unyt.mp
    ).to(unyt.g / unyt.cm**3)
    fluid.vel = np.zeros(par.nogrid) * unyt.cm / unyt.s
    fluid.temp = np.ones(par.nogrid) * config['initial_temperature']
    fluid.mu = np.ones(par.nogrid)
    fluid.xHI = np.ones(par.nogrid)
    fluid.SetUpFluid(par)
    fluid.SetFluidTime(0.0 * unyt.Myr)

    solver = Solver()
    solver.SetBoundary(mesh, fluid, par)
    solver.SetConserved(mesh, fluid)
    solver.ApplyRadiativeTransfer(mesh, fluid, par)
    return par, mesh, fluid, solver


def interior_slice(par):
    first = par.noghost
    return slice(first, first + par.nogrid)


def ionization_front_position(mesh, fluid, par, neutral_fraction=0.5):
    interior = interior_slice(par)
    radius = mesh.coordinate[interior].to_value(unyt.kpc)
    xHI = np.asarray(fluid.xHI[interior], dtype=float)

    ionized = xHI <= neutral_fraction
    if not np.any(ionized):
        return 0.0
    if np.all(ionized):
        return radius[-1]

    outer_ionized_index = np.where(ionized)[0][-1]
    left = outer_ionized_index
    right = outer_ionized_index + 1
    x_left = xHI[left]
    x_right = xHI[right]
    if x_right == x_left:
        return radius[left]

    weight = (neutral_fraction - x_left) / (x_right - x_left)
    return radius[left] + weight * (radius[right] - radius[left])


def mean_ionized_temperature(fluid, par):
    interior = interior_slice(par)
    xHI = np.asarray(fluid.xHI[interior], dtype=float)
    temperature = fluid.temp[interior].to_value(unyt.K)
    ionized_weight = 1.0 - xHI
    if np.sum(ionized_weight) <= 0.0:
        return 0.0
    return float(np.sum(ionized_weight * temperature) / np.sum(ionized_weight))


def append_history(history, mesh, fluid, par):
    history['time_Myr'].append(fluid.time.to_value(unyt.Myr))
    history['front_radius_kpc'].append(ionization_front_position(mesh, fluid, par))
    history['mean_ionized_temperature_K'].append(mean_ionized_temperature(fluid, par))


def stromgren_radius(config):
    radius = (
        3.0
        * config['source_photon_rate']
        / (
            4.0
            * np.pi
            * config['alpha_B_coefficient']
            * config['hydrogen_number_density']**2
        )
    ) ** (1.0 / 3.0)
    return radius.to(unyt.kpc)


def recombination_time(config):
    return (
        1.0
        / (config['hydrogen_number_density'] * config['alpha_B_coefficient'])
    ).to(unyt.Myr)


def ionized_sound_speed_from_history(history, gamma):
    temperature = history['mean_ionized_temperature_K'][-1] * unyt.K
    mu_ionized = 0.5
    return np.sqrt(gamma * unyt.kboltz * temperature / (mu_ionized * unyt.mp)).to(
        unyt.km / unyt.s
    )


def spitzer_radius(time, config, ci):
    radius_stromgren = stromgren_radius(config)
    factor = (
        1.0
        + 7.0
        * ci.to(unyt.cm / unyt.s)
        * time.to(unyt.s)
        / (4.0 * radius_stromgren.to(unyt.cm))
    )
    return (radius_stromgren * factor**(4.0 / 7.0)).to(unyt.kpc)


def shifted_spitzer_radius(time, config, ci):
    time_since_recombination = time - recombination_time(config)
    return spitzer_radius(time_since_recombination, config, ci)


def load_reference_profile(filename, radius_unit, log_value=False):
    if filename is None or not os.path.exists(filename):
        return None
    data = np.loadtxt(filename, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    value = data[:, 1]
    if log_value:
        value = 10.0**value
    return {
        'radius_kpc': data[:, 0] * radius_unit.to_value(unyt.kpc),
        'value': value,
    }


def scatter_reference(ax, reference, label='ZEUS-MP'):
    if reference is None:
        return
    ax.scatter(
        reference['radius_kpc'],
        reference['value'],
        s=20,
        color='black',
        marker='o',
        facecolors='none',
        label=label,
    )


def save_front_plot(history, config, figure_filename):
    time = np.asarray(history['time_Myr']) * unyt.Myr
    front_radius = np.asarray(history['front_radius_kpc'])
    radius_stromgren = stromgren_radius(config)
    tau_recombination = recombination_time(config)
    ci = ionized_sound_speed_from_history(history, 5.0 / 3.0)
    spitzer_valid = time >= tau_recombination
    radius_spitzer = shifted_spitzer_radius(
        time[spitzer_valid],
        config,
        ci,
    ).to_value(unyt.kpc)
    plot_radius_max = config['plot_radius_max'].to_value(unyt.kpc)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        time.to_value(unyt.Myr),
        front_radius,
        color='tab:blue',
        lw=2.0,
        label=r'RadHydropy $x_{\rm HI}=0.5$',
    )
    ax.plot(
        time[spitzer_valid].to_value(unyt.Myr),
        radius_spitzer,
        color='black',
        lw=1.7,
        ls='--',
        label=(
            r'Spitzer after $\tau_{\rm rec}$, '
            r'$c_i=%.1f$ km s$^{-1}$'
            % ci.to_value(unyt.km / unyt.s)
        ),
    )
    ax.axvline(
        tau_recombination.to_value(unyt.Myr),
        color='0.45',
        lw=1.2,
        ls='-.',
        label=r'$\tau_{\rm rec}=%.1f$ Myr' % tau_recombination.to_value(unyt.Myr),
    )
    ax.axhline(
        radius_stromgren.to_value(unyt.kpc),
        color='0.3',
        lw=1.4,
        ls=':',
        label=r'$R_{\rm S}$',
    )
    ax.set_xlim(0.0, time[-1].to_value(unyt.Myr))
    ax.set_ylim(0.0, max(plot_radius_max, 1.05 * np.nanmax(radius_spitzer)))
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Ionization-front radius [kpc]')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_plot(mesh, fluid, par, config, figure_filename):
    interior = interior_slice(par)
    radius_kpc = mesh.coordinate[interior].to_value(unyt.kpc)
    number_density = (
        fluid.rho[interior] / unyt.mp
    ).to_value(1.0 / unyt.cm**3)
    velocity = fluid.vel[interior].to_value(unyt.km / unyt.s)
    neutral_fraction = np.asarray(fluid.xHI[interior], dtype=float)
    pressure = fluid.pre[interior].to_value(unyt.g / unyt.cm / unyt.s**2)
    plot_radius_max = config['plot_radius_max'].to_value(unyt.kpc)
    radius_unit = config.get('reference_radius_unit', 15.0 * unyt.kpc)
    density_reference = load_reference_profile(
        config.get('density_reference_filename', None),
        radius_unit,
        log_value=True,
    )
    velocity_reference = load_reference_profile(
        config.get('velocity_reference_filename', None),
        radius_unit,
        log_value=False,
    )
    pressure_reference = load_reference_profile(
        config.get('pressure_reference_filename', None),
        radius_unit,
        log_value=True,
    )
    neutral_fraction_reference = load_reference_profile(
        config.get('neutral_fraction_reference_filename', None),
        radius_unit,
        log_value=True,
    )

    fig, axes = plt.subplots(4, 1, figsize=(7.4, 9.0), sharex=True)
    axes[0].plot(radius_kpc, number_density, color='tab:blue', lw=1.8, label='RadHydropy')
    scatter_reference(axes[0], density_reference)
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'$n$ [cm$^{-3}$]')
    axes[0].legend(frameon=False, loc='best')

    axes[1].plot(radius_kpc, velocity, color='tab:orange', lw=1.8, label='RadHydropy')
    scatter_reference(axes[1], velocity_reference)
    axes[1].set_ylabel(r'$v_r$ [km s$^{-1}$]')
    axes[1].legend(frameon=False, loc='best')

    axes[2].plot(
        radius_kpc,
        np.clip(neutral_fraction, 1.0e-8, 1.0),
        color='tab:green',
        lw=1.8,
        label='RadHydropy',
    )
    scatter_reference(axes[2], neutral_fraction_reference)
    axes[2].set_yscale('log')
    axes[2].set_ylabel(r'$x_{\rm HI}$')
    axes[2].legend(frameon=False, loc='best')

    axes[3].plot(radius_kpc, pressure, color='tab:red', lw=1.8, label='RadHydropy')
    scatter_reference(axes[3], pressure_reference)
    axes[3].set_yscale('log')
    axes[3].set_ylabel(r'$P$ [g cm$^{-1}$ s$^{-2}$]')
    axes[3].set_xlabel('Radius [kpc]')
    axes[3].legend(frameon=False, loc='best')

    for ax in axes:
        ax.set_xlim(0.0, plot_radius_max)
        ax.grid(True, which='both', alpha=0.25)
    fig.suptitle('Dynamic photoheated Stromgren sphere at 200 Myr')
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
