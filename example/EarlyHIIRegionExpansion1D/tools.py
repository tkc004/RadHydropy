"""Utilities for the early isothermal H II region expansion example."""

import os
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

import radhydropy.hydrogen as rh
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
        EOStype='isothermal',
        gamma=1.0,
        CFL=config['hydro_cfl'],
        order=0,
        dtmin=1.0e-10 * unyt.Myr,
        dtmax=config['hydro_timestep_max'],
        hydrogen_chemistry=True,
        hydrogen_mass_fraction=1.0,
        hydrogen_xHI_initial=1.0,
        hydrogen_xHI_inflow=1.0,
        hydrogen_xHI_outflow=1.0,
        hydrogen_source_CFL=config['source_cfl'],
        hydrogen_source_dtmin=config['source_timestep_min'],
        hydrogen_update_mu=True,
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
    fluid.eos = EOS(par.EOStype, par.gamma)
    fluid.rho = np.ones(par.nogrid) * config['rho_initial']
    fluid.vel = np.zeros(par.nogrid) * unyt.cm / unyt.s
    fluid.temp = np.ones(par.nogrid) * config['neutral_temperature']
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
    return slice(par.noghost, par.noghost + par.nogrid)


def apply_piecewise_isothermal_state(mesh, fluid, par, solver, config):
    interior = interior_slice(par)
    ionized_fraction = 1.0 - np.clip(fluid.xHI[interior], 0.0, 1.0)
    fluid.temp[interior] = (
        config['neutral_temperature']
        + ionized_fraction
        * (config['ionized_temperature'] - config['neutral_temperature'])
    )
    fluid.SetHydrogenMu(
        hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0)
    )
    fluid.SetPressure()
    solver.SetBoundary(mesh, fluid, par)
    solver.SetConserved(mesh, fluid)


def ionization_front_position(mesh, fluid, par, ionized_fraction=0.5):
    interior = interior_slice(par)
    radius = mesh.coordinate[interior].to_value(unyt.pc)
    xHII = 1.0 - np.asarray(fluid.xHI[interior], dtype=float)

    ionized = xHII >= ionized_fraction
    if not np.any(ionized):
        return 0.0
    if np.all(ionized):
        return radius[-1]

    outer_ionized_index = np.where(ionized)[0][-1]
    left = outer_ionized_index
    right = outer_ionized_index + 1
    x_left = xHII[left]
    x_right = xHII[right]
    if x_right == x_left:
        return radius[left]

    weight = (ionized_fraction - x_left) / (x_right - x_left)
    return radius[left] + weight * (radius[right] - radius[left])


def append_history(history, mesh, fluid, par):
    history['time_Myr'].append(fluid.time.to_value(unyt.Myr))
    history['front_radius_pc'].append(ionization_front_position(mesh, fluid, par))


def front_radius_at_time(history, time):
    time_myr = np.asarray(history['time_Myr'])
    front_radius_pc = np.asarray(history['front_radius_pc'])
    target_time_myr = time.to_value(unyt.Myr)
    if target_time_myr < time_myr[0] or target_time_myr > time_myr[-1]:
        raise ValueError('requested time is outside the recorded history')

    return np.interp(target_time_myr, time_myr, front_radius_pc) * unyt.pc


def stromgren_radius(config):
    nH = rh.hydrogen_number_density(
        config['rho_initial'],
        hydrogen_mass_fraction=1.0,
    )
    radius = (
        3.0
        * config['source_photon_rate']
        / (4.0 * np.pi * config['alpha_B_coefficient'] * nH**2)
    ) ** (1.0 / 3.0)
    return radius.to(unyt.pc)


def spitzer_radius(time, config):
    radius_stromgren = stromgren_radius(config)
    ionized_sound_speed = config['ionized_sound_speed'].to(unyt.cm / unyt.s)
    factor = (
        1.0
        + 7.0
        * ionized_sound_speed
        * time.to(unyt.s)
        / (4.0 * radius_stromgren.to(unyt.cm))
    )
    return (radius_stromgren * factor**(4.0 / 7.0)).to(unyt.pc)


def save_front_plot(history, config, figure_filename):
    time = np.asarray(history['time_Myr']) * unyt.Myr
    time_myr = time.to_value(unyt.Myr)
    front_radius_pc = np.asarray(history['front_radius_pc'])
    stromgren_radius_pc = stromgren_radius(config).to_value(unyt.pc)
    radius_spitzer_pc = spitzer_radius(time, config).to_value(unyt.pc)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        time_myr,
        front_radius_pc,
        color='tab:blue',
        lw=2.0,
        label=r'RadHydropy $x_{\rm HII}=0.5$',
    )
    ax.plot(
        time_myr,
        radius_spitzer_pc,
        color='tab:orange',
        lw=1.8,
        ls='--',
        label=(
            r'Spitzer, $c_i=%.2f$ km s$^{-1}$'
            % config['ionized_sound_speed'].to_value(unyt.km / unyt.s)
        ),
    )
    ax.axhline(
        stromgren_radius_pc,
        color='black',
        lw=1.4,
        ls='--',
        label=r'$R_{\rm S}$',
    )
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Ionization-front radius [pc]')
    ax.set_xlim(0.0, config['final_time'].to_value(unyt.Myr))
    ax.set_ylim(
        0.0,
        max(
            1.05 * np.max(front_radius_pc),
            1.05 * np.max(radius_spitzer_pc),
            1.1 * stromgren_radius_pc,
        ),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)
