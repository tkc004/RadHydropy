"""Helper utilities for the photoheated static Stromgren sphere example."""

import os
import sys
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

static_stromgren_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'StaticStromgrenSphere1D')
)
if static_stromgren_dir not in sys.path:
    sys.path.append(static_stromgren_dir)

import stromgren_analytic as sa


SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
K_BOLTZMANN = unyt.kboltz.to_value(unyt.erg / unyt.K)
M_PROTON = unyt.mp.to_value(unyt.g)


def build_static_problem(config):
    par = SimpleNamespace(
        coordsys='spherical',
        boundcond='OpenSph',
        nogrid=config['number_of_cells'],
        noghost=2,
        area=1.0 * unyt.cm**2,
        EOStype='polytropic',
        gamma=5.0 / 3.0,
        dtmin=1.0e-6 * unyt.Myr,
        dtmax=1.0 * unyt.Myr,
        hydrogen_chemistry=True,
        hydrogen_mass_fraction=1.0,
        hydrogen_xHI_initial=1.0,
        hydrogen_xHI_inflow=1.0,
        hydrogen_xHI_outflow=1.0,
        hydrogen_source_CFL=0.1,
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


def static_float_state(mesh, fluid, par, config):
    interior = interior_slice(par)
    boundary = mesh.boundary[interior.start : interior.stop + 1].to_value(unyt.cm)
    xHI = np.asarray(fluid.xHI[interior], dtype=float).copy()
    temperature_K = fluid.temp[interior].to_value(unyt.K).copy()
    gamma = par.gamma
    mu = 1.0 / (2.0 - xHI)
    specific_energy = K_BOLTZMANN * temperature_K / ((gamma - 1.0) * mu * M_PROTON)
    return {
        'interior': interior,
        'boundary_cm': boundary,
        'width_cm': np.diff(boundary),
        'volume_cm3': mesh.vol[interior].to_value(unyt.cm**3),
        'radius_kpc': mesh.coordinate[interior].to_value(unyt.kpc),
        'xHI': xHI,
        'temperature_K': temperature_K,
        'specific_energy_erg_g': specific_energy,
        'nH_cm3': config['hydrogen_number_density'].to_value(1.0 / unyt.cm**3),
        'rho_g_cm3': (
            config['hydrogen_number_density'] * unyt.mp
        ).to_value(unyt.g / unyt.cm**3),
        'alpha_B_cm3_s': config['alpha_B_coefficient'].to_value(
            unyt.cm**3 / unyt.s
        ),
        'sigma_cm2': config['sigma_gamma'].to_value(unyt.cm**2),
        'epsilon_erg': config['epsilon_gamma'].to_value(unyt.erg),
        'source_rate_s': config['source_photon_rate'].to_value(1.0 / unyt.s),
        'c_cm_s': rh.SPEED_OF_LIGHT.to_value(unyt.cm / unyt.s),
        'gamma': gamma,
    }


def update_temperature_from_energy(state):
    xHI = np.clip(state['xHI'], 1.0e-12, 1.0)
    mu = 1.0 / (2.0 - xHI)
    state['temperature_K'] = (
        (state['gamma'] - 1.0)
        * mu
        * M_PROTON
        * state['specific_energy_erg_g']
        / K_BOLTZMANN
    )
    state['temperature_K'] = np.maximum(state['temperature_K'], 1.0)


def attenuation_mean(tau):
    mean = np.ones_like(tau)
    valid = np.abs(tau) > 1.0e-10
    mean[valid] = -np.expm1(-tau[valid]) / tau[valid]
    return mean


def trace_spherical_ngamma(state):
    tau = (
        state['sigma_cm2']
        * state['nH_cm3']
        * np.clip(state['xHI'], 0.0, 1.0)
        * state['width_cm']
    )
    attenuation = np.exp(-np.clip(tau, 0.0, 700.0))
    mean_attenuation = attenuation_mean(tau)
    face_rate = np.zeros(len(state['xHI']) + 1)
    ngamma = np.zeros_like(state['xHI'])
    face_rate[0] = state['source_rate_s']
    for i in range(len(state['xHI'])):
        face_rate[i + 1] = face_rate[i] * attenuation[i]
        ngamma[i] = (
            face_rate[i]
            * state['width_cm'][i]
            * mean_attenuation[i]
            / state['volume_cm3'][i]
            / state['c_cm_s']
        )
    return ngamma


def neutral_fraction_rate(xHI, ngamma, state):
    recombination_rate = state['nH_cm3'] * state['alpha_B_cm3_s']
    photoionization_rate = state['c_cm_s'] * state['sigma_cm2'] * ngamma
    ionized = 1.0 - xHI
    return recombination_rate * ionized**2 - photoionization_rate * xHI


def implicit_neutral_fraction_update(xHI, ngamma, dt_s, state):
    xHI = np.clip(xHI, 1.0e-12, 1.0 - 1.0e-12)
    recombination_rate = state['nH_cm3'] * state['alpha_B_cm3_s']
    photoionization_rate = state['c_cm_s'] * state['sigma_cm2'] * ngamma

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


def thermal_rate_erg_cm3_s(state, ngamma):
    rho = (
        np.ones_like(state['xHI'])
        * state['rho_g_cm3']
        * unyt.g
        / unyt.cm**3
    )
    temperature = state['temperature_K'] * unyt.K
    ngamma_unyt = ngamma / unyt.cm**3
    rate = rh.hydrogen_thermal_rate(
        rho,
        temperature,
        state['xHI'],
        hydrogen_mass_fraction=1.0,
        recombination=True,
        collisional_ionization=False,
        ngamma=ngamma_unyt,
        sigma_gamma=state['sigma_cm2'] * unyt.cm**2,
        epsilon_gamma=state['epsilon_erg'] * unyt.erg,
    )
    return rate.to_value(unyt.erg / unyt.cm**3 / unyt.s)


def adaptive_timestep_seconds(state, ngamma, dtmin_s, dtmax_s, cfl):
    neutral_rate = neutral_fraction_rate(state['xHI'], ngamma, state)
    neutral_scale = np.where(neutral_rate < 0.0, state['xHI'], 1.0 - state['xHI'])
    neutral_valid = (np.abs(neutral_rate) > 0.0) & (neutral_scale > 0.0)
    candidates = []
    if np.any(neutral_valid):
        candidates.append(
            cfl * np.min(neutral_scale[neutral_valid] / np.abs(neutral_rate[neutral_valid]))
        )

    thermal_rate = thermal_rate_erg_cm3_s(state, ngamma)
    dudt = thermal_rate / state['rho_g_cm3']
    thermal_valid = (np.abs(dudt) > 0.0) & (state['specific_energy_erg_g'] > 0.0)
    if np.any(thermal_valid):
        candidates.append(
            cfl
            * np.min(
                state['specific_energy_erg_g'][thermal_valid]
                / np.abs(dudt[thermal_valid])
            )
        )

    if len(candidates) == 0:
        return dtmax_s, thermal_rate
    dt_s = min(dtmax_s, max(dtmin_s, min(candidates)))
    return dt_s, thermal_rate


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


def mean_ionized_temperature(fluid, par):
    interior = interior_slice(par)
    xHI = np.asarray(fluid.xHI[interior])
    temperature = fluid.temp[interior].to_value(unyt.K)
    ionized = 1.0 - xHI
    if np.sum(ionized) <= 0.0:
        return 0.0
    return float(np.sum(ionized * temperature) / np.sum(ionized))


def append_history(history, mesh, fluid, par):
    history['time_Myr'].append(fluid.time.to_value(unyt.Myr))
    history['front_radius_kpc'].append(
        ionization_front_position(mesh, fluid, par).to_value(unyt.kpc)
    )
    history['mean_ionized_temp_K'].append(mean_ionized_temperature(fluid, par))


def ionization_front_position_from_arrays(radius_kpc, xHI, neutral_fraction=0.5):
    ionized = xHI <= neutral_fraction
    if not np.any(ionized):
        return 0.0
    if np.all(ionized):
        return radius_kpc[-1]

    outer_ionized_index = np.where(ionized)[0][-1]
    left = outer_ionized_index
    right = outer_ionized_index + 1
    x_left = xHI[left]
    x_right = xHI[right]
    if x_right == x_left:
        return radius_kpc[left]

    weight = (neutral_fraction - x_left) / (x_right - x_left)
    return radius_kpc[left] + weight * (radius_kpc[right] - radius_kpc[left])


def mean_ionized_temperature_from_arrays(state):
    ionized = 1.0 - state['xHI']
    if np.sum(ionized) <= 0.0:
        return 0.0
    return float(np.sum(ionized * state['temperature_K']) / np.sum(ionized))


def append_fast_history(history, state, time_s):
    history['time_Myr'].append(time_s / SECONDS_PER_MYR)
    history['front_radius_kpc'].append(
        ionization_front_position_from_arrays(state['radius_kpc'], state['xHI'])
    )
    history['mean_ionized_temp_K'].append(mean_ionized_temperature_from_arrays(state))


def snapshot_state(state, time_s):
    return {
        'time_Myr': time_s / SECONDS_PER_MYR,
        'radius_kpc': state['radius_kpc'].copy(),
        'xHI': state['xHI'].copy(),
        'temperature_K': state['temperature_K'].copy(),
    }


def apply_fast_state_to_fluid(state, fluid):
    interior = state['interior']
    fluid.xHI[interior] = state['xHI']
    fluid.ngamma[interior] = state['ngamma'] / unyt.cm**3
    fluid.temp[interior] = state['temperature_K'] * unyt.K
    fluid.time = (state['time_s'] / SECONDS_PER_MYR) * unyt.Myr
    fluid.SetHydrogenMu(hydrogen_mass_fraction=1.0)


def evolve_photoheating(mesh, fluid, par, solver, config, final_time, timestep):
    state = static_float_state(mesh, fluid, par, config)
    ngamma = trace_spherical_ngamma(state)
    time_s = 0.0
    final_time_s = final_time.to_value(unyt.s)
    dtmax_s = timestep.to_value(unyt.s)
    dtmin_s = config.get(
        'evolution_timestep_min',
        1.0e-6 * unyt.Myr,
    ).to_value(unyt.s)
    timestep_cfl = config.get('evolution_timestep_cfl', 0.1)
    rt_update_interval = max(
        1,
        int(config.get('radiative_transfer_update_interval', 1)),
    )
    history = {
        'time_Myr': [],
        'front_radius_kpc': [],
        'mean_ionized_temp_K': [],
    }
    reference_time = config.get('reference_time', None)
    reference_time_s = None
    if reference_time is not None:
        reference_time_s = reference_time.to_value(unyt.s)
    append_fast_history(history, state, time_s)
    step = 0
    rt_updates = 1
    while time_s < final_time_s:
        dtmax_step_s = min(dtmax_s, final_time_s - time_s)
        if (
            reference_time_s is not None
            and 'reference_snapshot' not in history
            and time_s < reference_time_s <= time_s + dtmax_step_s
        ):
            dtmax_step_s = reference_time_s - time_s
        dt_s, thermal_rate = adaptive_timestep_seconds(
            state,
            ngamma,
            dtmin_s,
            dtmax_step_s,
            timestep_cfl,
        )
        state['specific_energy_erg_g'] += thermal_rate / state['rho_g_cm3'] * dt_s
        state['specific_energy_erg_g'] = np.maximum(
            state['specific_energy_erg_g'],
            1.0e6,
        )
        update_temperature_from_energy(state)
        state['xHI'] = implicit_neutral_fraction_update(
            state['xHI'],
            ngamma,
            dt_s,
            state,
        )
        update_temperature_from_energy(state)
        time_s += dt_s
        step += 1
        if step % rt_update_interval == 0 or time_s >= final_time_s:
            ngamma = trace_spherical_ngamma(state)
            rt_updates += 1
        if (
            reference_time_s is not None
            and 'reference_snapshot' not in history
            and time_s >= reference_time_s
        ):
            history['reference_snapshot'] = snapshot_state(state, time_s)
        append_fast_history(history, state, time_s)

    state['ngamma'] = trace_spherical_ngamma(state)
    state['time_s'] = time_s
    apply_fast_state_to_fluid(state, fluid)
    solver.SetBoundary(mesh, fluid, par)
    history['evolution_steps'] = step
    history['radiative_transfer_updates'] = rt_updates
    return history


def load_log_reference_profile(filename, radius_unit):
    if filename is None or not os.path.exists(filename):
        return None
    data = np.loadtxt(filename, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        'radius_kpc': data[:, 0] * radius_unit.to_value(unyt.kpc),
        'value': 10.0**data[:, 1],
    }


def save_plot(mesh, fluid, par, history, config, figure_filename):
    interior = interior_slice(par)
    radius = mesh.coordinate[interior].to(unyt.kpc)
    snapshot = history.get('reference_snapshot', None)
    if snapshot is None:
        radius_kpc = radius.to_value(unyt.kpc)
        xHI = fluid.xHI[interior]
        temperature_K = fluid.temp[interior].to_value(unyt.K)
        profile_time_Myr = fluid.time.to_value(unyt.Myr)
    else:
        radius_kpc = snapshot['radius_kpc']
        xHI = snapshot['xHI']
        temperature_K = snapshot['temperature_K']
        profile_time_Myr = snapshot['time_Myr']
    xHII = 1.0 - xHI
    plot_radius_max = config.get('plot_radius_max', config['boxsize']).to_value(unyt.kpc)
    reference_radius_unit = config.get('reference_radius_unit', 5.4 * unyt.kpc)
    temperature_reference = load_log_reference_profile(
        config.get('temperature_reference_filename', None),
        reference_radius_unit,
    )
    neutral_fraction_reference = load_log_reference_profile(
        config.get('neutral_fraction_reference_filename', None),
        reference_radius_unit,
    )
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
    analytic_front = sa.ionization_front_radius(
        np.asarray(history['time_Myr']) * unyt.Myr,
        config['source_photon_rate'],
        config['hydrogen_number_density'],
        config['alpha_B_coefficient'],
    ).to_value(unyt.kpc)

    fig, (ax_frac, ax_temp, ax_front) = plt.subplots(
        3,
        1,
        figsize=(7.4, 8.0),
        gridspec_kw={'height_ratios': [1.4, 1.2, 1.2], 'hspace': 0.34},
    )
    ax_frac.plot(radius_kpc, np.clip(xHI, 1.0e-6, 1.0), label=r'$x_{\rm HI}$')
    ax_frac.plot(radius_kpc, np.clip(xHII, 1.0e-6, 1.0), label=r'$x_{\rm HII}$')
    ax_frac.plot(
        radius_kpc,
        np.clip(xHI_analytic, 1.0e-6, 1.0),
        color='tab:blue',
        lw=1.4,
        ls='--',
        label=r'$x_{\rm HI}$ analytic',
    )
    ax_frac.plot(
        radius_kpc,
        np.clip(xHII_analytic, 1.0e-6, 1.0),
        color='tab:orange',
        lw=1.4,
        ls='--',
        label=r'$x_{\rm HII}$ analytic',
    )
    if neutral_fraction_reference is not None:
        ax_frac.scatter(
            neutral_fraction_reference['radius_kpc'],
            np.clip(neutral_fraction_reference['value'], 1.0e-6, 1.0),
            s=18,
            color='black',
            marker='o',
            facecolors='none',
            label=r'$x_{\rm HI}$ 100 Myr ref.',
        )
    ax_frac.axvline(
        radius_stromgren.to_value(unyt.kpc),
        color='black',
        lw=1.5,
        ls=':',
        label=r'$R_{\rm S}$',
    )
    ax_frac.set_xlim(0.0, plot_radius_max)
    ax_frac.set_ylim(1.0e-6, 1.2)
    ax_frac.set_yscale('log')
    ax_frac.set_ylabel('Hydrogen fraction')
    ax_frac.set_title('Radial profiles at %.0f Myr' % profile_time_Myr)
    ax_frac.grid(True, which='both', alpha=0.25)
    ax_frac.legend(frameon=False, loc='center right')

    ax_temp.plot(radius_kpc, temperature_K, color='tab:red', lw=1.8)
    if temperature_reference is not None:
        ax_temp.scatter(
            temperature_reference['radius_kpc'],
            temperature_reference['value'],
            s=18,
            color='black',
            marker='o',
            facecolors='none',
            label='100 Myr ref.',
        )
    ax_temp.axvline(radius_stromgren.to_value(unyt.kpc), color='black', lw=1.5, ls=':')
    ax_temp.set_xlim(0.0, plot_radius_max)
    ax_temp.set_yscale('log')
    ax_temp.set_ylabel('Temperature [K]')
    ax_temp.grid(True, which='both', alpha=0.25)
    if temperature_reference is not None:
        ax_temp.legend(frameon=False, loc='upper right')

    ax_front.plot(
        history['time_Myr'],
        history['front_radius_kpc'],
        color='tab:blue',
        lw=2.0,
        label=r'$x_{\rm HI}=0.5$',
    )
    ax_front.plot(
        history['time_Myr'],
        analytic_front,
        color='black',
        lw=1.6,
        ls='--',
        label=r'$R_I(t)$ fixed-$T$ reference',
    )
    ax_front.axhline(radius_stromgren.to_value(unyt.kpc), color='0.25', lw=1.2, ls=':')
    ax_front.set_xlim(0.0, history['time_Myr'][-1])
    ax_front.set_ylim(0.0, plot_radius_max)
    ax_front.set_xlabel('Time [Myr]')
    ax_front.set_ylabel('I-front radius [kpc]')
    ax_front.grid(True, alpha=0.25)
    ax_front.legend(frameon=False, loc='lower right')

    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
