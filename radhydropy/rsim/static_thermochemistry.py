"""Rsim execution subsystem helpers."""

import numpy as np
import unyt

import radhydropy.radiative_transfer as rrt
import radhydropy.thermo_chemistry as rtc
from radhydropy.units import time_seconds


def _static_front_radius_from_state(sim, state, neutral_fraction=0.5):
    ionized = state['xHI'] <= neutral_fraction
    if not np.any(ionized):
        return 0.0
    if np.all(ionized):
        return state['radius_kpc'][-1]
    left = np.where(ionized)[0][-1]
    right = left + 1
    x_left = state['xHI'][left]
    x_right = state['xHI'][right]
    if x_right == x_left:
        return state['radius_kpc'][left]
    weight = (neutral_fraction - x_left) / (x_right - x_left)
    return state['radius_kpc'][left] + weight * (
        state['radius_kpc'][right] - state['radius_kpc'][left]
    )

def _append_static_history(
    sim,
    history,
    state,
    ngamma_cgs_cm3,
    time_s,
    recombined_photons,
    source_rate_s,
    seconds_to_myr,
):
    ionized = 1.0 - state['xHI']
    ionized_atoms = np.sum(ionized * state['nH_cgs_cm3'] * state['volume_cgs_cm3'])
    volume_photons = np.sum(ngamma_cgs_cm3 * state['volume_cgs_cm3'])
    history['time_Myr'].append(time_s * seconds_to_myr)
    history['front_radius_kpc'].append(sim._static_front_radius_from_state(state))
    history['injected_photons'].append(source_rate_s * time_s)
    history['ionized_atoms'].append(ionized_atoms)
    history['recombined_photons'].append(recombined_photons)
    history['volume_photons'].append(volume_photons)
    history['accounted_photons'].append(
        ionized_atoms + recombined_photons + volume_photons
    )
    if 'mean_ionized_temp_cgs_K' in history:
        ionized_weight = 1.0 - state['xHI']
        if np.sum(ionized_weight) > 0.0:
            mean_temp = np.sum(ionized_weight * state['temperature_cgs_K']) / np.sum(
                ionized_weight
            )
        else:
            mean_temp = 0.0
        history['mean_ionized_temp_cgs_K'].append(float(mean_temp))

def _snapshot_static_state(sim, state, time_s):
    return {
        'time_Myr': time_s / (1.0 * unyt.Myr).to_value(unyt.s),
        'radius_kpc': state['radius_kpc'].copy(),
        'xHI': state['xHI'].copy(),
        'temperature_cgs_K': state['temperature_cgs_K'].copy(),
    }

def _initial_static_history(sim, include_thermal_history=False):
    history = {
        'time_Myr': [],
        'front_radius_kpc': [],
        'injected_photons': [],
        'ionized_atoms': [],
        'recombined_photons': [],
        'volume_photons': [],
        'accounted_photons': [],
    }
    if include_thermal_history:
        history['mean_ionized_temp_cgs_K'] = []
    return history

def _static_reference_time_seconds(sim, reference_time):
    if reference_time is None:
        return None
    return time_seconds(
        reference_time,
        getattr(sim.par, 'CodeUnits', None),
    )

def _static_step_limit_seconds(sim, time_s, final_time_s, dtmax_s, reference_time_s, history):
    remaining_s = final_time_s - time_s
    dtmax_step_s = min(dtmax_s, remaining_s)
    if (
        reference_time_s is not None
        and 'reference_snapshot' not in history
        and time_s < reference_time_s <= time_s + dtmax_step_s
    ):
        dtmax_step_s = reference_time_s - time_s
    return remaining_s, dtmax_step_s

def _static_recombination_rate(sim, state):
    alpha = state.get('alpha_B_cgs_cm3_s', None)
    if alpha is None:
        alpha = getattr(sim.par, 'hydrogen_alpha_B', None)
    if alpha is None:
        return 0.0
    if hasattr(alpha, 'to_value'):
        alpha_value = alpha.to_value(unyt.cm**3 / unyt.s)
    else:
        alpha_value = float(alpha)
    ionized = 1.0 - state['xHI']
    return np.sum(
        alpha_value
        * ionized**2
        * state['nH_cgs_cm3']**2
        * state['volume_cgs_cm3']
    )

def _apply_static_thermal_update(sim, state, ngamma_cgs_cm3, thermal_rate, dt_s):
    if not getattr(sim.par, 'hydrogen_thermal_coupling', True):
        return
    if thermal_rate is None:
        thermal_rate = rtc.thermal_rate(state, ngamma_cgs_cm3, sim.par)
    active = np.asarray(
        state.get('active', np.asarray(state['rho_cgs_g_cm3']) > 0.0),
        dtype=bool,
    )
    rho = np.where(active, state['rho_cgs_g_cm3'], 1.0)
    energy_update = np.zeros_like(state['specific_energy_cgs_erg_g'])
    energy_update[active] = thermal_rate[active] / rho[active] * dt_s
    state['specific_energy_cgs_erg_g'] += energy_update
    state['specific_energy_cgs_erg_g'] = np.maximum(
        state['specific_energy_cgs_erg_g'],
        1.0e6,
    )
    rtc.update_temperature_from_energy(state)

def _advance_source_thermochemistry_state(sim, state, ngamma_cgs_cm3, dt_s, thermal_rate):
    recombination_rate_start = sim._static_recombination_rate(state)
    if getattr(sim.par, 'thermochemistry_network', 'hydrogen') == 'hydrogen_helium':
        rtc.coupled_implicit_update(state, ngamma_cgs_cm3, dt_s, sim.par)
    else:
        sim._apply_static_thermal_update(state, ngamma_cgs_cm3, thermal_rate, dt_s)
        rtc.ionization_fraction_implicit_update(state, ngamma_cgs_cm3, dt_s, sim.par)
    if getattr(sim.par, 'hydrogen_thermal_coupling', True):
        rtc.update_temperature_from_energy(state)
    recombination_rate_end = sim._static_recombination_rate(state)
    return 0.5 * (recombination_rate_start + recombination_rate_end) * dt_s

def _refresh_static_photon_density(sim, state, step, time_s, final_time_s):
    # The static thermo-chemistry path still needs the radiation field to
    # follow the evolving neutral fraction. Refresh every source step so
    # the next implicit update sees the current opacity.
    ngamma_cgs_cm3 = rrt.trace_photon_density(state, sim.par)
    return ngamma_cgs_cm3, 1

def _store_static_reference_snapshot(sim, history, state, time_s, reference_time_s):
    if (
        reference_time_s is not None
        and 'reference_snapshot' not in history
        and time_s >= reference_time_s
    ):
        history['reference_snapshot'] = sim._snapshot_static_state(state, time_s)

def _finish_static_thermochemistry(sim, state, time_s):
    if (
        getattr(sim.par, 'radiative_transfer_temporal_scheme', 'instantaneous')
        == 'c2ray'
    ):
        from radhydropy.thermo_networks import c2ray

        state['ngamma_cgs_cm3'] = state.get('ngamma_cgs_cm3')
        c2ray._ensure_fluid_photon_shape(sim.fluid, state['ngamma_cgs_cm3'])
    else:
        state['ngamma_cgs_cm3'] = rrt.trace_photon_density(state, sim.par)
    state['time_s'] = time_s
    rtc.apply_state(state, sim.fluid, sim.par)
    sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)

def EvolveStaticThermochemistry(
    sim,
    final_time,
    source_timestep,
    include_thermal_history=False,
    reference_time=None,
):
    """Evolve fixed-density thermo-chemistry/radiation source terms."""
    state = rtc.source_state(sim.mesh, sim.fluid, sim.par)
    code_units = getattr(sim.par, 'CodeUnits', None)
    final_time_s = time_seconds(final_time, code_units)
    dtmax_s = time_seconds(source_timestep, code_units)
    reference_time_s = sim._static_reference_time_seconds(reference_time)
    if (
        getattr(sim.par, 'radiative_transfer_temporal_scheme', 'instantaneous')
        == 'c2ray'
    ):
        history = rtc.evolve_static_source_state(
            state,
            sim.par,
            final_time_s=final_time_s,
            dtmax_s=dtmax_s,
            source_rate_s=getattr(sim.par, '_static_source_rate_s', 0.0),
            include_thermal_history=include_thermal_history,
            reference_time_s=reference_time_s,
        )
        sim._finish_static_thermochemistry(
            state,
            state.get('time_s', final_time_s),
        )
        return history
    ngamma_cgs_cm3 = rrt.trace_photon_density(state, sim.par)
    recombined_photons = 0.0
    time_s = 0.0
    source_rate_s = getattr(sim.par, '_static_source_rate_s', 0.0)
    seconds_to_myr = 1.0 / (1.0 * unyt.Myr).to_value(unyt.s)
    history = sim._initial_static_history(
        include_thermal_history=include_thermal_history
    )
    sim._append_static_history(
        history,
        state,
        ngamma_cgs_cm3,
        time_s,
        recombined_photons,
        source_rate_s,
        seconds_to_myr,
    )
    step = 0
    rt_updates = 1
    while time_s < final_time_s:
        remaining_s, dtmax_step_s = sim._static_step_limit_seconds(
            time_s,
            final_time_s,
            dtmax_s,
            reference_time_s,
            history,
        )
        dt_s, thermal_rate = rtc.get_timestep(
            state,
            ngamma_cgs_cm3,
            sim.par,
            remaining_s,
            dtmax_step_s,
        )
        recombined_photons += sim._advance_source_thermochemistry_state(
            state,
            ngamma_cgs_cm3,
            dt_s,
            thermal_rate,
        )
        time_s += dt_s
        step += 1
        updated_ngamma, updates = sim._refresh_static_photon_density(
            state,
            step,
            time_s,
            final_time_s
        )
        if updated_ngamma is not None:
            ngamma_cgs_cm3 = updated_ngamma
            rt_updates += updates
        sim._store_static_reference_snapshot(
            history,
            state,
            time_s,
            reference_time_s,
        )
        sim._append_static_history(
            history,
            state,
            ngamma_cgs_cm3,
            time_s,
            recombined_photons,
            source_rate_s,
            seconds_to_myr,
        )

    sim._finish_static_thermochemistry(state, time_s)
    history['chemistry_steps'] = step
    history['evolution_steps'] = step
    history['radiative_transfer_updates'] = rt_updates
    return history
