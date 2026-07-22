"""Hydrogen thermo-chemistry network.

This module contains the current hydrogen-only rate network.  The public
dispatcher in :mod:`radhydropy.thermo_chemistry` calls this through the
``HydrogenNetwork`` interface.
"""

import numpy as np
import unyt

import radhydropy.hydrogen as rh
import radhydropy.radiative_transfer as rrt
import radhydropy.utils as ru
from radhydropy.thermo_networks.base import ThermochemistryNetwork


def interior_slice(par):
    first = par.noghost
    return slice(first, first + par.nogrid)


def thermochemistry_enabled(fluid, par):
    return getattr(par, 'hydrogen_chemistry', False) and hasattr(fluid, 'xHI')


def thermochemistry_radiation_enabled(fluid, par):
    return (
        thermochemistry_enabled(fluid, par)
        and (
            getattr(par, 'hydrogen_radiation_field', False)
            or getattr(par, 'radiative_transfer', False)
        )
        and hasattr(fluid, 'ngamma')
    )


def thermochemistry_radiation_evolution_enabled(fluid, par):
    return (
        thermochemistry_radiation_enabled(fluid, par)
        and getattr(par, 'hydrogen_radiation_evolution', True)
        and not getattr(par, 'radiative_transfer', False)
    )


def spherical_center_cell_index(mesh):
    if getattr(mesh, 'coordsys', None) != 'spherical' or not hasattr(mesh, 'boundary'):
        return None
    origin = 0.0 * mesh.boundary.units
    origin_faces = np.where(mesh.boundary[:-1] == origin)[0]
    if len(origin_faces) > 0:
        return int(origin_faces[0])
    origin_cells = np.where(
        np.logical_and(mesh.boundary[:-1] < origin, mesh.boundary[1:] > origin)
    )[0]
    if len(origin_cells) > 0:
        return int(origin_cells[0])
    return None


def set_primitive(mesh, fluid):
    vol = mesh.vol
    fluid.rho = ru.SafeDivide(fluid.Mass, vol)
    fluid.vel = ru.SafeDivide(fluid.Mom, fluid.Mass)
    energy_density = ru.SafeDivide(fluid.Energy, vol)
    fluid.pre = (
        energy_density - 0.5 * fluid.rho * fluid.vel**2
    ) * (fluid.eos.gamma - 1.0)
    fluid.rho[np.logical_or(fluid.rho < 0.0, np.isnan(fluid.rho))] = 0.0
    fluid.vel[np.isnan(fluid.vel)] = 0.0 * fluid.vel.units
    fluid.pre[np.logical_or(fluid.pre < 0.0, np.isnan(fluid.pre))] = 0.0
    center_cell = spherical_center_cell_index(mesh)
    if center_cell is not None:
        fluid.vel[center_cell] = 0.0 * fluid.vel.units


def apply_radiative_transfer(mesh, fluid, par):
    return rrt.apply_long_characteristics_to_fluid(mesh, fluid, par)


def advect_ionization_fraction(dt, mesh, fluid, par, old_mass, mass_flux):
    """Advect the chemistry fraction consistently with the mass flux."""
    if not hasattr(fluid, 'xHI'):
        return
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


def trace_spherical_photon_density_fast(mesh, fluid, par):
    """Update ``ngamma`` with a lightweight spherical long-characteristic trace."""
    if getattr(mesh, 'coordsys', None) != 'spherical':
        apply_radiative_transfer(mesh, fluid, par)
        return fluid.ngamma[interior_slice(par)]
    if not hasattr(fluid, 'ngamma'):
        return None

    interior = interior_slice(par)
    boundary = mesh.boundary[interior.start : interior.stop + 1].to_value(unyt.cm)
    width = np.diff(boundary)
    volume = mesh.vol[interior].to_value(unyt.cm**3)
    rho = fluid.rho[interior].to_value(unyt.g / unyt.cm**3)
    nH = (
        rho
        * getattr(par, 'hydrogen_mass_fraction', 1.0)
        / unyt.mp.to_value(unyt.g)
    )
    xHI = np.clip(np.asarray(fluid.xHI[interior], dtype=float), 0.0, 1.0)
    sigma = getattr(par, 'hydrogen_sigma_gamma', rh.DEFAULT_SIGMA_GAMMA).to_value(
        unyt.cm**2
    )
    source_rate = getattr(
        par,
        'radiative_transfer_source_photon_rate',
        0.0 / unyt.s,
    ).to_value(1.0 / unyt.s)
    c_light = rh.SPEED_OF_LIGHT.to_value(unyt.cm / unyt.s)

    tau = sigma * nH * xHI * width
    attenuation = np.exp(-np.clip(tau, 0.0, 700.0))
    mean_attenuation = np.ones_like(tau)
    valid = np.abs(tau) > 1.0e-10
    mean_attenuation[valid] = -np.expm1(-tau[valid]) / tau[valid]

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
    return fluid.ngamma[interior]


def static_thermochemistry_state(mesh, fluid, par):
    """Return a float state for fixed-density static thermo-chemistry tests."""
    interior = interior_slice(par)
    boundary = mesh.boundary[interior.start : interior.stop + 1].to_value(unyt.cm)
    xHI = np.asarray(fluid.xHI[interior], dtype=float).copy()
    temperature = fluid.temp[interior].to_value(unyt.K).copy()
    rho = fluid.rho[interior].to_value(unyt.g / unyt.cm**3)
    gamma = getattr(
        getattr(fluid, 'eos', None),
        'gamma',
        getattr(par, 'gamma', 5.0 / 3.0),
    )
    mu = 1.0 / (2.0 - np.clip(xHI, 1.0e-12, 1.0))
    specific_energy = (
        unyt.kboltz.to_value(unyt.erg / unyt.K)
        * temperature
        / ((gamma - 1.0) * mu * unyt.mp.to_value(unyt.g))
    )
    return {
        'interior': interior,
        'boundary_cm': boundary,
        'width_cm': np.diff(boundary),
        'volume_cm3': mesh.vol[interior].to_value(unyt.cm**3),
        'radius_kpc': mesh.coordinate[interior].to_value(unyt.kpc),
        'xHI': xHI,
        'temperature_K': temperature,
        'specific_energy_erg_g': specific_energy,
        'rho_g_cm3': rho,
        'nH_cm3': (
            rho
            * getattr(par, 'hydrogen_mass_fraction', 1.0)
            / unyt.mp.to_value(unyt.g)
        ),
        'gamma': gamma,
    }


def trace_static_spherical_photon_density(state, par):
    """Trace a central source through a static spherical float state."""
    sigma = getattr(par, 'hydrogen_sigma_gamma', rh.DEFAULT_SIGMA_GAMMA).to_value(
        unyt.cm**2
    )
    source_rate = getattr(
        par,
        'radiative_transfer_source_photon_rate',
        0.0 / unyt.s,
    ).to_value(1.0 / unyt.s)
    c_light = rh.SPEED_OF_LIGHT.to_value(unyt.cm / unyt.s)
    tau = sigma * state['nH_cm3'] * np.clip(state['xHI'], 0.0, 1.0) * state['width_cm']
    attenuation = np.exp(-np.clip(tau, 0.0, 700.0))
    mean_attenuation = np.ones_like(tau)
    valid = np.abs(tau) > 1.0e-10
    mean_attenuation[valid] = -np.expm1(-tau[valid]) / tau[valid]
    face_rate = np.zeros(len(state['xHI']) + 1)
    ngamma = np.zeros_like(state['xHI'])
    face_rate[0] = source_rate
    for i in range(len(state['xHI'])):
        face_rate[i + 1] = face_rate[i] * attenuation[i]
        ngamma[i] = (
            face_rate[i]
            * state['width_cm'][i]
            * mean_attenuation[i]
            / state['volume_cm3'][i]
            / c_light
        )
    return ngamma


def static_ionization_fraction_rate(state, ngamma, par):
    """Return the chemistry fraction rate for a static float state."""
    recombination = getattr(par, 'hydrogen_recombination', True)
    collisional = getattr(par, 'hydrogen_collisional_ionization', True)
    alpha = getattr(par, 'hydrogen_alpha_B', None)
    beta = getattr(par, 'hydrogen_beta', None)
    if alpha is None:
        alpha = rh.alpha_B(state['temperature_K'] * unyt.K)
        alpha = alpha.to_value(unyt.cm**3 / unyt.s)
    else:
        alpha = alpha.to_value(unyt.cm**3 / unyt.s)
    if beta is None:
        beta = rh.beta(state['temperature_K'] * unyt.K)
        beta = beta.to_value(unyt.cm**3 / unyt.s)
    else:
        beta = beta.to_value(unyt.cm**3 / unyt.s)
    photoionization_rate = (
        rh.SPEED_OF_LIGHT.to_value(unyt.cm / unyt.s)
        * getattr(par, 'hydrogen_sigma_gamma', rh.DEFAULT_SIGMA_GAMMA).to_value(unyt.cm**2)
        * ngamma
    )
    xHI = state['xHI']
    ionized = 1.0 - xHI
    rate = -photoionization_rate * xHI
    if recombination:
        rate += state['nH_cm3'] * alpha * ionized**2
    if collisional:
        rate -= state['nH_cm3'] * beta * xHI * ionized
    return rate


def static_thermal_rate(state, ngamma, par):
    """Return thermal source rate for a static float state."""
    rate = rh.hydrogen_thermal_rate(
        state['rho_g_cm3'] * unyt.g / unyt.cm**3,
        state['temperature_K'] * unyt.K,
        state['xHI'],
        hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
        recombination=getattr(par, 'hydrogen_recombination', True),
        collisional_ionization=getattr(par, 'hydrogen_collisional_ionization', True),
        ngamma=ngamma / unyt.cm**3,
        sigma_gamma=getattr(par, 'hydrogen_sigma_gamma', rh.DEFAULT_SIGMA_GAMMA),
        epsilon_gamma=getattr(par, 'hydrogen_epsilon_gamma', rh.DEFAULT_EPSILON_GAMMA),
    )
    return rate.to_value(unyt.erg / unyt.cm**3 / unyt.s)


def get_static_thermochemistry_timestep(state, ngamma, par, remaining_s, dtmax_s):
    """Return a source substep for a static float thermo-chemistry state."""
    source_CFL = getattr(par, 'hydrogen_source_CFL', 0.1)
    dtmin_s = getattr(par, 'hydrogen_source_dtmin', 0.0 * unyt.s).to_value(unyt.s)
    candidates = []
    neutral_rate = static_ionization_fraction_rate(state, ngamma, par)
    scale = np.where(neutral_rate < 0.0, state['xHI'], 1.0 - state['xHI'])
    valid = (np.abs(neutral_rate) > 0.0) & (scale > 0.0)
    if np.any(valid):
        candidates.append(source_CFL * np.min(scale[valid] / np.abs(neutral_rate[valid])))

    thermal_rate = None
    if getattr(par, 'hydrogen_thermal_coupling', True):
        thermal_rate = static_thermal_rate(state, ngamma, par)
        dudt = thermal_rate / state['rho_g_cm3']
        valid = (np.abs(dudt) > 0.0) & (state['specific_energy_erg_g'] > 0.0)
        if np.any(valid):
            candidates.append(
                source_CFL
                * np.min(
                    state['specific_energy_erg_g'][valid]
                    / np.abs(dudt[valid])
                )
            )

    if len(candidates) == 0:
        return min(dtmax_s, remaining_s), thermal_rate
    return min(dtmax_s, remaining_s, max(dtmin_s, min(candidates))), thermal_rate


def update_static_temperature_from_energy(state):
    """Update temperature in a static float state from specific energy."""
    mu = 1.0 / (2.0 - np.clip(state['xHI'], 1.0e-12, 1.0))
    state['temperature_K'] = (
        (state['gamma'] - 1.0)
        * mu
        * unyt.mp.to_value(unyt.g)
        * state['specific_energy_erg_g']
        / unyt.kboltz.to_value(unyt.erg / unyt.K)
    )
    state['temperature_K'] = np.maximum(state['temperature_K'], 1.0)


def static_ionization_fraction_implicit_update(state, ngamma, dt_s, par):
    """Implicitly update the chemistry fraction for a static float state."""
    xHI = np.clip(state['xHI'], 1.0e-12, 1.0 - 1.0e-12)
    recombination = getattr(par, 'hydrogen_recombination', True)
    alpha = getattr(par, 'hydrogen_alpha_B', None)
    if alpha is None:
        alpha = rh.alpha_B(state['temperature_K'] * unyt.K)
        alpha = alpha.to_value(unyt.cm**3 / unyt.s)
    else:
        alpha = alpha.to_value(unyt.cm**3 / unyt.s)
    recombination_rate = state['nH_cm3'] * alpha if recombination else 0.0
    photoionization_rate = (
        rh.SPEED_OF_LIGHT.to_value(unyt.cm / unyt.s)
        * getattr(par, 'hydrogen_sigma_gamma', rh.DEFAULT_SIGMA_GAMMA).to_value(unyt.cm**2)
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
    state['xHI'] = np.clip(updated, 1.0e-12, 1.0 - 1.0e-12)


def apply_static_thermochemistry_state(state, fluid, par):
    """Copy a static float state back to a fluid object."""
    interior = state['interior']
    fluid.xHI[interior] = state['xHI']
    if hasattr(fluid, 'ngamma') and 'ngamma' in state:
        fluid.ngamma[interior] = state['ngamma'] / unyt.cm**3
    if hasattr(fluid, 'temp') and 'temperature_K' in state:
        fluid.temp[interior] = state['temperature_K'] * unyt.K
    if hasattr(fluid, 'xHI') and getattr(getattr(fluid, 'eos', None), 'gamma', None) is not None:
        fluid.SetHydrogenMu(
            hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0)
        )
        fluid.SetPressure()
    fluid.time = (state['time_s'] * unyt.s).to(unyt.Myr)


def get_thermochemistry_source_timestep_fast(mesh, fluid, par, remaining):
    """Return a source substep for RT-coupled heating/chemistry."""
    interior = interior_slice(par)
    source_CFL = getattr(par, 'hydrogen_source_CFL', 0.1)
    dtmin = getattr(par, 'hydrogen_source_dtmin', 0.0 * unyt.s).to(unyt.s)
    hydrogen_mass_fraction = getattr(par, 'hydrogen_mass_fraction', 1.0)
    recombination = getattr(par, 'hydrogen_recombination', True)
    collisional_ionization = getattr(par, 'hydrogen_collisional_ionization', True)
    thermal_coupling = getattr(par, 'hydrogen_thermal_coupling', True)
    sigma_gamma = getattr(par, 'hydrogen_sigma_gamma', rh.DEFAULT_SIGMA_GAMMA)
    epsilon_gamma = getattr(par, 'hydrogen_epsilon_gamma', rh.DEFAULT_EPSILON_GAMMA)
    recombination_coefficient = getattr(par, 'hydrogen_alpha_B', None)
    ionization_coefficient = getattr(par, 'hydrogen_beta', None)

    thermal_rate, neutral_fraction_rate = rh.hydrogen_source_terms(
        fluid.rho,
        fluid.temp,
        fluid.xHI,
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        recombination=recombination,
        collisional_ionization=collisional_ionization,
        ngamma=fluid.ngamma if hasattr(fluid, 'ngamma') else None,
        sigma_gamma=sigma_gamma,
        epsilon_gamma=epsilon_gamma,
        recombination_coefficient=recombination_coefficient,
        ionization_coefficient=ionization_coefficient,
    )
    candidates = []

    if thermal_coupling:
        thermal_energy_density = fluid.pre / (fluid.eos.gamma - 1.0)
        thermal_rate_abs = np.absolute(thermal_rate[interior])
        thermal_valid = np.logical_and(
            thermal_rate_abs > 0.0 * thermal_rate_abs.units,
            thermal_energy_density[interior] > 0.0 * thermal_energy_density.units,
        )
        if np.any(thermal_valid):
            thermal_times = (
                source_CFL
                * thermal_energy_density[interior][thermal_valid]
                / thermal_rate_abs[thermal_valid]
            ).to(unyt.s)
            thermal_times = thermal_times[
                np.logical_and(
                    np.isfinite(thermal_times.value),
                    thermal_times.value > 0.0,
                )
            ]
            if len(thermal_times) > 0:
                candidates.append(np.amin(thermal_times))

    xHI = rh.clip_neutral_fraction(fluid.xHI[interior])
    neutral_fraction_rate_values = neutral_fraction_rate[interior].to_value(1.0 / unyt.s)
    neutral_fraction_rate_abs = np.absolute(neutral_fraction_rate_values)
    scale = np.where(neutral_fraction_rate_values < 0.0, xHI, 1.0 - xHI)
    chemistry_valid = np.logical_and(neutral_fraction_rate_abs > 0.0, scale > 0.0)
    if np.any(chemistry_valid):
        chemistry_times = (
            source_CFL
            * scale[chemistry_valid]
            / neutral_fraction_rate_abs[chemistry_valid]
        ) * unyt.s
        chemistry_times = chemistry_times[
            np.logical_and(
                np.isfinite(chemistry_times.value),
                chemistry_times.value > 0.0,
            )
        ]
        if len(chemistry_times) > 0:
            candidates.append(np.amin(chemistry_times))

    if len(candidates) == 0:
        return remaining, thermal_rate
    sub_dt = min(remaining, max(dtmin, min(candidates)))
    return sub_dt, thermal_rate


def apply_thermal_source(dt, mesh, fluid, thermal_rate, par):
    interior = interior_slice(par)
    fluid.Energy[interior] += (
        thermal_rate[interior] * mesh.vol[interior] * dt
    ).to(fluid.Energy.units)
    kinetic_energy = 0.5 * fluid.rho * fluid.vel**2 * mesh.vol
    overcooled = fluid.Energy[interior] < kinetic_energy[interior]
    if np.any(overcooled):
        interior_indices = np.arange(len(fluid.Energy))[interior]
        overcooled_indices = interior_indices[overcooled]
        fluid.Energy[overcooled_indices] = kinetic_energy[overcooled_indices]


def apply_thermochemistry_fast(dt, mesh, fluid, par):
    """Fast source update for RT-coupled thermo-chemistry tests."""
    if not thermochemistry_enabled(fluid, par):
        return 0

    hydrogen_mass_fraction = getattr(par, 'hydrogen_mass_fraction', 1.0)
    recombination = getattr(par, 'hydrogen_recombination', True)
    collisional_ionization = getattr(par, 'hydrogen_collisional_ionization', True)
    thermal_coupling = getattr(par, 'hydrogen_thermal_coupling', True)
    sigma_gamma = getattr(par, 'hydrogen_sigma_gamma', rh.DEFAULT_SIGMA_GAMMA)
    recombination_coefficient = getattr(par, 'hydrogen_alpha_B', None)
    ionization_coefficient = getattr(par, 'hydrogen_beta', None)
    interior = interior_slice(par)
    remaining = dt.to(unyt.s)
    zero_time = 0.0 * unyt.s
    source_steps = 0
    while remaining > zero_time:
        if getattr(par, 'radiative_transfer', False):
            trace_spherical_photon_density_fast(mesh, fluid, par)
        if getattr(par, 'hydrogen_update_mu', False):
            fluid.SetHydrogenMu(hydrogen_mass_fraction=hydrogen_mass_fraction)
        fluid.SetTemperature()
        sub_dt, thermal_rate = get_thermochemistry_source_timestep_fast(
            mesh,
            fluid,
            par,
            remaining,
        )
        if not np.isfinite(sub_dt.to_value(unyt.s)) or sub_dt <= zero_time:
            sub_dt = remaining
        if sub_dt > remaining:
            sub_dt = remaining

        if thermal_coupling:
            apply_thermal_source(sub_dt, mesh, fluid, thermal_rate, par)
            set_primitive(mesh, fluid)
        if getattr(par, 'hydrogen_update_mu', False):
            fluid.SetHydrogenMu(hydrogen_mass_fraction=hydrogen_mass_fraction)
        fluid.SetTemperature()
        fluid.xHI[interior] = rh.hydrogen_neutral_fraction_implicit_update(
            fluid.rho[interior],
            fluid.temp[interior],
            fluid.xHI[interior],
            sub_dt,
            hydrogen_mass_fraction=hydrogen_mass_fraction,
            recombination=recombination,
            collisional_ionization=collisional_ionization,
            ngamma=fluid.ngamma[interior] if hasattr(fluid, 'ngamma') else None,
            sigma_gamma=sigma_gamma,
            recombination_coefficient=recombination_coefficient,
            ionization_coefficient=ionization_coefficient,
        )
        if getattr(par, 'hydrogen_update_mu', False):
            fluid.SetHydrogenMu(hydrogen_mass_fraction=hydrogen_mass_fraction)
        remaining -= sub_dt
        source_steps += 1
    if getattr(par, 'radiative_transfer', False):
        trace_spherical_photon_density_fast(mesh, fluid, par)
    return source_steps


def get_thermochemistry_timestep(mesh, fluid, par):
    """Return a thermo-chemistry source subcycle timestep."""
    if not thermochemistry_enabled(fluid, par):
        return par.dtmax

    source_CFL = getattr(par, 'hydrogen_source_CFL', 0.1)
    hydrogen_mass_fraction = getattr(par, 'hydrogen_mass_fraction', 1.0)
    recombination = getattr(par, 'hydrogen_recombination', True)
    collisional_ionization = getattr(par, 'hydrogen_collisional_ionization', True)
    thermal_coupling = getattr(par, 'hydrogen_thermal_coupling', True)
    radiation_evolution = thermochemistry_radiation_evolution_enabled(fluid, par)
    sigma_gamma = getattr(par, 'hydrogen_sigma_gamma', rh.DEFAULT_SIGMA_GAMMA)
    epsilon_gamma = getattr(par, 'hydrogen_epsilon_gamma', rh.DEFAULT_EPSILON_GAMMA)
    recombination_coefficient = getattr(par, 'hydrogen_alpha_B', None)
    ionization_coefficient = getattr(par, 'hydrogen_beta', None)
    ngamma = fluid.ngamma if thermochemistry_radiation_enabled(fluid, par) else None
    if getattr(par, 'hydrogen_update_mu', False):
        fluid.SetHydrogenMu(hydrogen_mass_fraction=hydrogen_mass_fraction)
    fluid.SetTemperature()
    thermal_rate, neutral_fraction_rate = rh.hydrogen_source_terms(
        fluid.rho,
        fluid.temp,
        fluid.xHI,
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        recombination=recombination,
        collisional_ionization=collisional_ionization,
        ngamma=ngamma,
        sigma_gamma=sigma_gamma,
        epsilon_gamma=epsilon_gamma,
        recombination_coefficient=recombination_coefficient,
        ionization_coefficient=ionization_coefficient,
    )
    interior = interior_slice(par)
    candidates = []

    if radiation_evolution:
        radiation_rate = rh.hydrogen_radiation_rate(
            fluid.rho,
            fluid.xHI,
            fluid.ngamma,
            hydrogen_mass_fraction=hydrogen_mass_fraction,
            sigma_gamma=sigma_gamma,
        )
        photon_density = fluid.ngamma[interior]
        radiation_rate_abs = np.absolute(radiation_rate[interior])
        radiation_valid = np.logical_and(
            radiation_rate_abs > 0.0 * radiation_rate_abs.units,
            photon_density > 0.0 * photon_density.units,
        )
        if np.any(radiation_valid):
            radiation_times = (
                source_CFL
                * photon_density[radiation_valid]
                / radiation_rate_abs[radiation_valid]
            ).to(unyt.s)
            radiation_times = radiation_times[
                np.logical_and(
                    np.isfinite(radiation_times.value),
                    radiation_times.value > 0.0,
                )
            ]
            if len(radiation_times) > 0:
                candidates.append(np.amin(radiation_times))

    thermal_energy_density = fluid.pre / (fluid.eos.gamma - 1.0)
    thermal_rate_abs = np.absolute(thermal_rate[interior])
    if thermal_coupling:
        cooling_valid = np.logical_and(
            thermal_rate_abs > 0.0 * thermal_rate_abs.units,
            thermal_energy_density[interior] > 0.0 * thermal_energy_density.units,
        )
    else:
        cooling_valid = np.zeros(len(thermal_rate_abs), dtype=bool)
    if np.any(cooling_valid):
        cooling_times = (
            source_CFL
            * thermal_energy_density[interior][cooling_valid]
            / thermal_rate_abs[cooling_valid]
        ).to(unyt.s)
        cooling_times = cooling_times[
            np.logical_and(np.isfinite(cooling_times.value), cooling_times.value > 0.0)
        ]
        if len(cooling_times) > 0:
            candidates.append(np.amin(cooling_times))

    xHI = rh.clip_neutral_fraction(fluid.xHI[interior])
    neutral_fraction_rate_abs = np.absolute(
        neutral_fraction_rate[interior].to_value(1.0 / unyt.s)
    )
    chemistry_valid = np.logical_and(neutral_fraction_rate_abs > 0.0, xHI > 0.0)
    if np.any(chemistry_valid):
        chemistry_times = (
            source_CFL
            * xHI[chemistry_valid]
            / neutral_fraction_rate_abs[chemistry_valid]
        ) * unyt.s
        chemistry_times = chemistry_times[
            np.logical_and(np.isfinite(chemistry_times.value), chemistry_times.value > 0.0)
        ]
        if len(chemistry_times) > 0:
            candidates.append(np.amin(chemistry_times))

    if len(candidates) == 0:
        return par.dtmax
    return min(candidates)


def apply_thermochemistry(dt, mesh, fluid, par):
    """Subcycle cooling explicitly and chemistry implicitly."""
    if not thermochemistry_enabled(fluid, par):
        return

    hydrogen_mass_fraction = getattr(par, 'hydrogen_mass_fraction', 1.0)
    recombination = getattr(par, 'hydrogen_recombination', True)
    collisional_ionization = getattr(par, 'hydrogen_collisional_ionization', True)
    thermal_coupling = getattr(par, 'hydrogen_thermal_coupling', True)
    sigma_gamma = getattr(par, 'hydrogen_sigma_gamma', rh.DEFAULT_SIGMA_GAMMA)
    epsilon_gamma = getattr(par, 'hydrogen_epsilon_gamma', rh.DEFAULT_EPSILON_GAMMA)
    recombination_coefficient = getattr(par, 'hydrogen_alpha_B', None)
    ionization_coefficient = getattr(par, 'hydrogen_beta', None)
    if getattr(par, 'radiative_transfer', False):
        apply_radiative_transfer(mesh, fluid, par)

    radiation_evolution = thermochemistry_radiation_evolution_enabled(fluid, par)
    interior = interior_slice(par)
    remaining = dt.to(unyt.s)
    zero_time = 0.0 * unyt.s
    while remaining > zero_time:
        if getattr(par, 'radiative_transfer', False):
            apply_radiative_transfer(mesh, fluid, par)
        if getattr(par, 'hydrogen_update_mu', False):
            fluid.SetHydrogenMu(hydrogen_mass_fraction=hydrogen_mass_fraction)
        fluid.SetTemperature()
        ngamma = fluid.ngamma if thermochemistry_radiation_enabled(fluid, par) else None
        thermal_rate, _ = rh.hydrogen_source_terms(
            fluid.rho,
            fluid.temp,
            fluid.xHI,
            hydrogen_mass_fraction=hydrogen_mass_fraction,
            recombination=recombination,
            collisional_ionization=collisional_ionization,
            ngamma=ngamma,
            sigma_gamma=sigma_gamma,
            epsilon_gamma=epsilon_gamma,
            recombination_coefficient=recombination_coefficient,
            ionization_coefficient=ionization_coefficient,
        )
        sub_dt = get_thermochemistry_timestep(mesh, fluid, par)
        if not np.isfinite(sub_dt.to_value(unyt.s)) or sub_dt <= zero_time:
            sub_dt = remaining
        if sub_dt > remaining:
            sub_dt = remaining

        if radiation_evolution:
            fluid.ngamma[interior] = rh.hydrogen_radiation_analytic_update(
                fluid.rho[interior],
                fluid.xHI[interior],
                fluid.ngamma[interior],
                sub_dt,
                hydrogen_mass_fraction=hydrogen_mass_fraction,
                sigma_gamma=sigma_gamma,
            ).to(fluid.ngamma.units)

        if thermal_coupling:
            apply_thermal_source(sub_dt, mesh, fluid, thermal_rate, par)
            set_primitive(mesh, fluid)
        if getattr(par, 'hydrogen_update_mu', False):
            fluid.SetHydrogenMu(hydrogen_mass_fraction=hydrogen_mass_fraction)
        fluid.SetTemperature()
        fluid.xHI[interior] = rh.hydrogen_neutral_fraction_implicit_update(
            fluid.rho[interior],
            fluid.temp[interior],
            fluid.xHI[interior],
            sub_dt,
            hydrogen_mass_fraction=hydrogen_mass_fraction,
            recombination=recombination,
            collisional_ionization=collisional_ionization,
            ngamma=(
                fluid.ngamma[interior]
                if thermochemistry_radiation_enabled(fluid, par)
                else None
            ),
            sigma_gamma=sigma_gamma,
            recombination_coefficient=recombination_coefficient,
            ionization_coefficient=ionization_coefficient,
        )
        if getattr(par, 'hydrogen_update_mu', False):
            fluid.SetHydrogenMu(hydrogen_mass_fraction=hydrogen_mass_fraction)
        remaining -= sub_dt
    if getattr(par, 'radiative_transfer', False):
        apply_radiative_transfer(mesh, fluid, par)


class HydrogenNetwork(ThermochemistryNetwork):
    """Hydrogen-only thermo-chemistry network."""

    name = "hydrogen"
    scalar_fields = ("xHI",)

    def enabled(self, fluid, par):
        return thermochemistry_enabled(fluid, par)

    def radiation_enabled(self, fluid, par):
        return thermochemistry_radiation_enabled(fluid, par)

    def radiation_evolution_enabled(self, fluid, par):
        return thermochemistry_radiation_evolution_enabled(fluid, par)

    def advect_ionization_fraction(self, dt, mesh, fluid, par, old_mass, mass_flux):
        return advect_ionization_fraction(dt, mesh, fluid, par, old_mass, mass_flux)

    def trace_spherical_photon_density_fast(self, mesh, fluid, par):
        return trace_spherical_photon_density_fast(mesh, fluid, par)

    def static_state(self, mesh, fluid, par):
        return static_thermochemistry_state(mesh, fluid, par)

    def trace_static_spherical_photon_density(self, state, par):
        return trace_static_spherical_photon_density(state, par)

    def static_ionization_fraction_rate(self, state, ngamma, par):
        return static_ionization_fraction_rate(state, ngamma, par)

    def static_thermal_rate(self, state, ngamma, par):
        return static_thermal_rate(state, ngamma, par)

    def get_static_timestep(self, state, ngamma, par, remaining_s, dtmax_s):
        return get_static_thermochemistry_timestep(
            state,
            ngamma,
            par,
            remaining_s,
            dtmax_s,
        )

    def update_static_temperature_from_energy(self, state):
        return update_static_temperature_from_energy(state)

    def static_ionization_fraction_implicit_update(self, state, ngamma, dt_s, par):
        return static_ionization_fraction_implicit_update(state, ngamma, dt_s, par)

    def apply_static_state(self, state, fluid, par):
        return apply_static_thermochemistry_state(state, fluid, par)

    def get_source_timestep_fast(self, mesh, fluid, par, remaining):
        return get_thermochemistry_source_timestep_fast(mesh, fluid, par, remaining)

    def apply_fast(self, dt, mesh, fluid, par):
        return apply_thermochemistry_fast(dt, mesh, fluid, par)

    def get_timestep(self, mesh, fluid, par):
        return get_thermochemistry_timestep(mesh, fluid, par)

    def apply(self, dt, mesh, fluid, par):
        return apply_thermochemistry(dt, mesh, fluid, par)
