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
import radhydropy.hydrogen as rh
import radhydropy.utils as ru


K_BOLTZMANN = unyt.kboltz.to_value(unyt.erg / unyt.K)
M_PROTON = unyt.mp.to_value(unyt.g)


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


def attenuation_mean(tau):
    mean = np.ones_like(tau)
    valid = np.abs(tau) > 1.0e-10
    mean[valid] = -np.expm1(-tau[valid]) / tau[valid]
    return mean


def trace_spherical_ngamma(mesh, fluid, par, config):
    interior = interior_slice(par)
    boundary = mesh.boundary[interior.start : interior.stop + 1].to_value(unyt.cm)
    width = np.diff(boundary)
    volume = mesh.vol[interior].to_value(unyt.cm**3)
    rho = fluid.rho[interior].to_value(unyt.g / unyt.cm**3)
    nH = rho / M_PROTON
    xHI = np.clip(np.asarray(fluid.xHI[interior], dtype=float), 0.0, 1.0)
    sigma = config['sigma_gamma'].to_value(unyt.cm**2)
    source_rate = config['source_photon_rate'].to_value(1.0 / unyt.s)
    c_light = rh.SPEED_OF_LIGHT.to_value(unyt.cm / unyt.s)

    tau = sigma * nH * xHI * width
    attenuation = np.exp(-np.clip(tau, 0.0, 700.0))
    mean_attenuation = attenuation_mean(tau)
    face_rate = np.zeros(len(xHI) + 1)
    ngamma = np.zeros_like(xHI)
    face_rate[0] = source_rate
    for i in range(len(xHI)):
        face_rate[i + 1] = face_rate[i] * attenuation[i]
        ngamma[i] = (
            face_rate[i]
            * width[i]
            * mean_attenuation[i]
            / volume[i]
            / c_light
        )

    fluid.ngamma[interior] = ngamma / unyt.cm**3
    return ngamma


def neutral_fraction_rate(rho_g_cm3, xHI, ngamma, config):
    nH = rho_g_cm3 / M_PROTON
    recombination_rate = nH * config['alpha_B_coefficient'].to_value(
        unyt.cm**3 / unyt.s
    )
    photoionization_rate = (
        rh.SPEED_OF_LIGHT.to_value(unyt.cm / unyt.s)
        * config['sigma_gamma'].to_value(unyt.cm**2)
        * ngamma
    )
    ionized = 1.0 - xHI
    return recombination_rate * ionized**2 - photoionization_rate * xHI


def implicit_neutral_fraction_update(rho_g_cm3, xHI, ngamma, dt_s, config):
    xHI = np.clip(xHI, 1.0e-12, 1.0 - 1.0e-12)
    nH = rho_g_cm3 / M_PROTON
    recombination_rate = nH * config['alpha_B_coefficient'].to_value(
        unyt.cm**3 / unyt.s
    )
    photoionization_rate = (
        rh.SPEED_OF_LIGHT.to_value(unyt.cm / unyt.s)
        * config['sigma_gamma'].to_value(unyt.cm**2)
        * ngamma
    )

    a = dt_s * recombination_rate
    b = -(1.0 + dt_s * (photoionization_rate + 2.0 * recombination_rate))
    c = xHI + dt_s * recombination_rate
    discriminant = np.maximum(b**2 - 4.0 * a * c, 0.0)
    denominator = -b + np.sqrt(discriminant)
    updated = np.divide(
        2.0 * c,
        denominator,
        out=xHI.copy(),
        where=denominator != 0.0,
    )
    return np.clip(updated, 1.0e-12, 1.0 - 1.0e-12)


def thermal_rate_erg_cm3_s(rho_g_cm3, temperature_K, xHI, ngamma, config):
    rate = rh.hydrogen_thermal_rate(
        rho_g_cm3 * unyt.g / unyt.cm**3,
        temperature_K * unyt.K,
        xHI,
        hydrogen_mass_fraction=1.0,
        recombination=True,
        collisional_ionization=False,
        ngamma=ngamma / unyt.cm**3,
        sigma_gamma=config['sigma_gamma'],
        epsilon_gamma=config['epsilon_gamma'],
    )
    return rate.to_value(unyt.erg / unyt.cm**3 / unyt.s)


def adaptive_source_timestep(rho_g_cm3, temperature_K, xHI, specific_energy, ngamma, config, remaining_s):
    cfl = config['source_cfl']
    dtmin_s = config['source_timestep_min'].to_value(unyt.s)
    rate = neutral_fraction_rate(rho_g_cm3, xHI, ngamma, config)
    scale = np.where(rate < 0.0, xHI, 1.0 - xHI)
    valid = (np.abs(rate) > 0.0) & (scale > 0.0)
    candidates = []
    if np.any(valid):
        candidates.append(cfl * np.min(scale[valid] / np.abs(rate[valid])))

    thermal_rate = thermal_rate_erg_cm3_s(rho_g_cm3, temperature_K, xHI, ngamma, config)
    dudt = thermal_rate / rho_g_cm3
    valid = (np.abs(dudt) > 0.0) & (specific_energy > 0.0)
    if np.any(valid):
        candidates.append(cfl * np.min(specific_energy[valid] / np.abs(dudt[valid])))

    if len(candidates) == 0:
        dt_s = remaining_s
    else:
        dt_s = min(remaining_s, max(dtmin_s, min(candidates)))
    return dt_s, thermal_rate


def temperature_from_energy(specific_energy, xHI, gamma):
    mu = 1.0 / (2.0 - np.clip(xHI, 1.0e-12, 1.0))
    temperature = (gamma - 1.0) * mu * M_PROTON * specific_energy / K_BOLTZMANN
    return np.maximum(temperature, 1.0)


def apply_photoheating_sources(dt, mesh, fluid, par, config):
    interior = interior_slice(par)
    remaining_s = dt.to_value(unyt.s)
    source_steps = 0
    while remaining_s > 0.0:
        ngamma = trace_spherical_ngamma(mesh, fluid, par, config)
        rho = fluid.rho[interior].to_value(unyt.g / unyt.cm**3)
        xHI = np.asarray(fluid.xHI[interior], dtype=float)
        temperature = fluid.temp[interior].to_value(unyt.K)
        specific_energy = (
            fluid.pre[interior].to_value(unyt.erg / unyt.cm**3)
            / (par.gamma - 1.0)
            / rho
        )
        sub_dt_s, thermal_rate = adaptive_source_timestep(
            rho,
            temperature,
            xHI,
            specific_energy,
            ngamma,
            config,
            remaining_s,
        )
        specific_energy += thermal_rate / rho * sub_dt_s
        specific_energy = np.maximum(specific_energy, 1.0e6)
        xHI = implicit_neutral_fraction_update(rho, xHI, ngamma, sub_dt_s, config)
        temperature = temperature_from_energy(specific_energy, xHI, par.gamma)
        mu = 1.0 / (2.0 - xHI)

        fluid.xHI[interior] = xHI
        fluid.mu[interior] = mu
        fluid.temp[interior] = temperature * unyt.K
        fluid.pre[interior] = ru.CalPressure(
            fluid.rho[interior],
            fluid.temp[interior],
            fluid.mu[interior],
        )
        remaining_s -= sub_dt_s
        source_steps += 1

    fluid.SetHydrogenMu(hydrogen_mass_fraction=1.0)
    fluid.SetPressure()
    return source_steps


def advect_neutral_fraction(dt, mesh, fluid, par, old_mass, mass_flux):
    face_area = mesh.area
    x_left = np.roll(fluid.xHI, 1)
    x_right = fluid.xHI
    x_face = np.where(mass_flux >= 0.0 * mass_flux.units, x_left, x_right)
    neutral_mass = np.asarray(fluid.xHI) * old_mass
    neutral_flux = x_face * mass_flux
    neutral_mass += (
        neutral_flux * face_area
        - np.roll(neutral_flux * face_area, -1)
    ) * dt
    xHI = ru.SafeDivide(neutral_mass, fluid.Mass)
    fluid.xHI = rh.clip_neutral_fraction(xHI.to_value(''))


def run_hydro_step(mesh, fluid, par, solver, dt):
    solver.SetBoundary(mesh, fluid, par)
    solver.SetConserved(mesh, fluid)
    old_mass = fluid.Mass.copy()
    solver.SetInterFaceFlux(mesh, fluid, par.boundcond, order=par.order)
    mass_flux = fluid.Mass.flux.copy()
    solver.AddFluxes(dt, mesh, fluid, par.boundcond)
    advect_neutral_fraction(dt, mesh, fluid, par, old_mass, mass_flux)
    solver.SetPrimitive(mesh, fluid)
    fluid.SetHydrogenMu(hydrogen_mass_fraction=1.0)
    fluid.SetTemperature()
    fluid.SetPressure()
    solver.SetConserved(mesh, fluid)


def evolve(mesh, fluid, par, solver, config, final_time):
    hydro_steps = 0
    source_steps = 0
    while fluid.time < final_time:
        dt = solver.GetTimeStep(mesh, fluid, par)
        if fluid.time + dt > final_time:
            dt = final_time - fluid.time
        run_hydro_step(mesh, fluid, par, solver, dt)
        source_steps += apply_photoheating_sources(dt, mesh, fluid, par, config)
        solver.SetBoundary(mesh, fluid, par)
        solver.SetConserved(mesh, fluid)
        hydro_steps += 1
    trace_spherical_ngamma(mesh, fluid, par, config)
    solver.SetBoundary(mesh, fluid, par)
    return {'hydro_steps': hydro_steps, 'source_steps': source_steps}


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
