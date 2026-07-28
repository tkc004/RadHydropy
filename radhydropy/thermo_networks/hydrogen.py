"""Hydrogen thermo-chemistry network.

This module contains the current hydrogen-only rate network.  The public
dispatcher in :mod:`radhydropy.thermo_chemistry` calls this through the
``HydrogenNetwork`` interface.
"""

import numpy as np
import unyt

import radhydropy.chemistry_species.hydrogen as rh
import radhydropy.radiative_transfer as rrt
import radhydropy.utils as ru
from radhydropy.thermo_networks.base import ThermochemistryNetwork


_PROTON_MASS_G = unyt.mp.to_value(unyt.g)
_BOLTZMANN_ERG_PER_K = unyt.kboltz.to_value(unyt.erg / unyt.K)
_SPEED_OF_LIGHT_CMS = rh.SPEED_OF_LIGHT.to_value(unyt.cm / unyt.s)
_CM2 = unyt.cm**2
_CM3_PER_S = unyt.cm**3 / unyt.s
_ERG = unyt.erg
_PER_S = 1.0 / unyt.s


def _as_float_array(value):
    if hasattr(value, 'to_value'):
        return np.asarray(value.to_value(), dtype=float)
    return np.asarray(value, dtype=float)


def _as_float_array_in_units(value, units):
    if hasattr(value, 'to_value'):
        return np.asarray(value.to_value(units), dtype=float)
    return np.asarray(value, dtype=float)


def _alpha_B_value(temperature_K):
    temperature_K = np.asarray(temperature_K, dtype=float)
    result = np.zeros_like(temperature_K, dtype=float)
    valid = temperature_K > 0.0
    if np.any(valid):
        lam = 315614.0 / temperature_K[valid]
        result[valid] = (
            2.753e-14
            * lam**1.5
            * (1.0 + (lam / 2.740) ** 0.407) ** -2.242
        )
    return result


def _beta_value(temperature_K):
    temperature_K = np.asarray(temperature_K, dtype=float)
    result = np.zeros_like(temperature_K, dtype=float)
    valid = temperature_K > 0.0
    if np.any(valid):
        temp = temperature_K[valid]
        temp5 = temp / 1.0e5
        result[valid] = (
            1.17e-10
            * temp**0.5
            * np.exp(-157809.1 / temp)
            / (1.0 + temp5**0.5)
        )
    return result


def _gamma_line_eHI_value(temperature_K):
    temperature_K = np.asarray(temperature_K, dtype=float)
    result = np.zeros_like(temperature_K, dtype=float)
    valid = temperature_K > 0.0
    if np.any(valid):
        temp = temperature_K[valid]
        temp5 = temp / 1.0e5
        result[valid] = (
            7.5e-19
            * np.exp(-118348.0 / temp)
            / (1.0 + temp5**0.5)
        )
    return result


def _gamma_ion_eHI_value(temperature_K):
    temperature_K = np.asarray(temperature_K, dtype=float)
    result = np.zeros_like(temperature_K, dtype=float)
    valid = temperature_K > 0.0
    if np.any(valid):
        temp = temperature_K[valid]
        temp5 = temp / 1.0e5
        result[valid] = (
            2.54e-21
            * temp**0.5
            * np.exp(-157809.1 / temp)
            / (1.0 + temp5**0.5)
        )
    return result


def _gamma_ff_eHII_value(temperature_K):
    temperature_K = np.asarray(temperature_K, dtype=float)
    result = np.zeros_like(temperature_K, dtype=float)
    valid = temperature_K > 0.0
    if np.any(valid):
        temp = temperature_K[valid]
        result[valid] = (
            1.42e-27
            * temp**0.5
            * (1.1 + 0.34 * np.exp(-(5.5 - np.log10(temp)) ** 2 / 3.0))
        )
    return result


def _gamma_B_eHII_value(temperature_K):
    temperature_K = np.asarray(temperature_K, dtype=float)
    result = np.zeros_like(temperature_K, dtype=float)
    valid = temperature_K > 0.0
    if np.any(valid):
        temp = temperature_K[valid]
        lam = 315614.0 / temp
        result[valid] = (
            3.435e-30
            * temp
            * lam**1.970
            * (1.0 + (lam / 2.250) ** 0.376) ** -3.720
        )
    return result


def _hydrogen_number_density_value(rho_g_cm3, hydrogen_mass_fraction=1.0):
    return hydrogen_mass_fraction * np.asarray(rho_g_cm3, dtype=float) / _PROTON_MASS_G


def _photoionization_frequency_value(ngamma_cm3, sigma_gamma_cm2):
    return _SPEED_OF_LIGHT_CMS * np.asarray(sigma_gamma_cm2, dtype=float) * np.asarray(
        ngamma_cm3,
        dtype=float,
    )


def _source_thermal_rate_value(
    rho_g_cm3,
    temperature_K,
    xHI,
    hydrogen_mass_fraction=1.0,
    recombination=True,
    collisional_ionization=True,
    ngamma_cm3=None,
    sigma_gamma_cm2=1.0,
    epsilon_gamma_erg=0.0,
):
    xHI = np.clip(np.asarray(xHI, dtype=float), 0.0, 1.0)
    ionized = 1.0 - xHI
    nH = _hydrogen_number_density_value(rho_g_cm3, hydrogen_mass_fraction)
    eHI_cooling = _gamma_line_eHI_value(temperature_K)
    if collisional_ionization:
        eHI_cooling += _gamma_ion_eHI_value(temperature_K)
    eHII_cooling = _gamma_ff_eHII_value(temperature_K)
    if recombination:
        eHII_cooling += _gamma_B_eHII_value(temperature_K)
    cooling = nH**2 * (xHI * ionized * eHI_cooling + ionized**2 * eHII_cooling)
    if ngamma_cm3 is None:
        heating = np.zeros_like(cooling, dtype=float)
    else:
        heating = (
            np.asarray(epsilon_gamma_erg, dtype=float)
            * _photoionization_frequency_value(ngamma_cm3, sigma_gamma_cm2)
        )
    return heating - cooling


def _static_neutral_fraction_rate_value(
    rho_g_cm3,
    temperature_K,
    xHI,
    hydrogen_mass_fraction=1.0,
    recombination=True,
    collisional_ionization=True,
    ngamma_cm3=None,
    sigma_gamma_cm2=1.0,
    recombination_coefficient_cm3_s=None,
    ionization_coefficient_cm3_s=None,
):
    xHI = np.clip(np.asarray(xHI, dtype=float), 0.0, 1.0)
    ionized = 1.0 - xHI
    nH = _hydrogen_number_density_value(rho_g_cm3, hydrogen_mass_fraction)
    if recombination_coefficient_cm3_s is None:
        recombination_coefficient_cm3_s = _alpha_B_value(temperature_K)
    else:
        recombination_coefficient_cm3_s = np.asarray(
            recombination_coefficient_cm3_s,
            dtype=float,
        )
    if not recombination:
        recombination_coefficient_cm3_s = np.zeros_like(
            recombination_coefficient_cm3_s,
            dtype=float,
        )
    if ionization_coefficient_cm3_s is None:
        ionization_coefficient_cm3_s = _beta_value(temperature_K)
    else:
        ionization_coefficient_cm3_s = np.asarray(
            ionization_coefficient_cm3_s,
            dtype=float,
        )
    if not collisional_ionization:
        ionization_coefficient_cm3_s = np.zeros_like(
            ionization_coefficient_cm3_s,
            dtype=float,
        )
    if ngamma_cm3 is None:
        photoionization_rate_s = np.zeros_like(xHI, dtype=float)
    else:
        photoionization_rate_s = _photoionization_frequency_value(
            ngamma_cm3,
            sigma_gamma_cm2,
        )
    return (
        ionized**2 * nH * recombination_coefficient_cm3_s
        - xHI * ionized * nH * ionization_coefficient_cm3_s
        - xHI * photoionization_rate_s
    )


def _static_neutral_fraction_implicit_update_value(
    rho_g_cm3,
    temperature_K,
    xHI,
    dt_s,
    hydrogen_mass_fraction=1.0,
    recombination=True,
    collisional_ionization=True,
    ngamma_cm3=None,
    sigma_gamma_cm2=1.0,
    recombination_coefficient_cm3_s=None,
    ionization_coefficient_cm3_s=None,
):
    xHI = np.clip(np.asarray(xHI, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    nH = _hydrogen_number_density_value(rho_g_cm3, hydrogen_mass_fraction)
    if recombination:
        if recombination_coefficient_cm3_s is None:
            recombination_coefficient_cm3_s = _alpha_B_value(temperature_K)
        else:
            recombination_coefficient_cm3_s = np.asarray(
                recombination_coefficient_cm3_s,
                dtype=float,
            )
        recombination_rate_s = nH * recombination_coefficient_cm3_s
    else:
        recombination_rate_s = np.zeros_like(xHI, dtype=float)
    if collisional_ionization:
        if ionization_coefficient_cm3_s is None:
            ionization_coefficient_cm3_s = _beta_value(temperature_K)
        else:
            ionization_coefficient_cm3_s = np.asarray(
                ionization_coefficient_cm3_s,
                dtype=float,
            )
        ionization_rate_s = nH * ionization_coefficient_cm3_s
    else:
        ionization_rate_s = np.zeros_like(recombination_rate_s, dtype=float)
    if ngamma_cm3 is None:
        photoionization_rate_s = np.zeros_like(recombination_rate_s, dtype=float)
    else:
        photoionization_rate_s = _photoionization_frequency_value(
            ngamma_cm3,
            sigma_gamma_cm2,
        )

    dt_value = float(np.asarray(dt_s, dtype=float))
    a = dt_value * (recombination_rate_s + ionization_rate_s)
    b = -(
        1.0
        + dt_value
        * (photoionization_rate_s + 2.0 * recombination_rate_s + ionization_rate_s)
    )
    c = xHI + dt_value * recombination_rate_s
    discriminant = np.maximum(b**2 - 4.0 * a * c, 0.0)
    denominator = -b + np.sqrt(discriminant)
    updated = np.array(xHI, copy=True, dtype=float)
    updated = np.divide(
        2.0 * c,
        denominator,
        out=updated,
        where=denominator != 0.0,
    )
    return np.clip(updated, 1.0e-12, 1.0 - 1.0e-12)


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


def source_state(mesh, fluid, par):
    """Return a float state for fixed-density thermo-chemistry tests."""
    interior = interior_slice(par)
    boundary = _as_float_array_in_units(
        mesh.boundary[interior.start : interior.stop + 1],
        unyt.cm,
    )
    xHI = np.asarray(fluid.xHI[interior], dtype=float).copy()
    temperature = _as_float_array_in_units(fluid.temp[interior], unyt.K).copy()
    rho = _as_float_array_in_units(fluid.rho[interior], unyt.g / unyt.cm**3)
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
    sigma_gamma = _as_float_array_in_units(
        getattr(par, 'hydrogen_sigma_gamma', rh.DEFAULT_SIGMA_GAMMA),
        _CM2,
    )
    source_rate = _as_float_array_in_units(
        getattr(par, 'radiative_transfer_source_photon_rate', 0.0 / unyt.s),
        _PER_S,
    )
    epsilon_gamma = _as_float_array_in_units(
        getattr(par, 'hydrogen_epsilon_gamma', rh.DEFAULT_EPSILON_GAMMA),
        _ERG,
    )
    alpha_B = getattr(par, 'hydrogen_alpha_B', None)
    if alpha_B is not None:
        alpha_B = _as_float_array_in_units(alpha_B, _CM3_PER_S)
    beta = getattr(par, 'hydrogen_beta', None)
    if beta is not None:
        beta = _as_float_array_in_units(beta, _CM3_PER_S)
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
        'nH_cm3': _hydrogen_number_density_value(
            rho,
            hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
        ),
        'gamma': gamma,
        'hydrogen_mass_fraction': getattr(par, 'hydrogen_mass_fraction', 1.0),
        'sigma_gamma_cm2': sigma_gamma,
        'source_rate_s': source_rate,
        'epsilon_gamma_erg': epsilon_gamma,
        'source_CFL': getattr(par, 'hydrogen_source_CFL', 0.1),
        'dtmin_s': _as_float_array_in_units(
            getattr(par, 'hydrogen_source_dtmin', 0.0 * unyt.s),
            unyt.s,
        ),
        'recombination': getattr(par, 'hydrogen_recombination', True),
        'collisional_ionization': getattr(
            par,
            'hydrogen_collisional_ionization',
            True,
        ),
        'thermal_coupling': getattr(par, 'hydrogen_thermal_coupling', True),
        'alpha_B_cm3_s': alpha_B,
        'beta_cm3_s': beta,
    }


def trace_spherical_photon_density(state):
    """Trace a central source through a float source state."""
    sigma = state['sigma_gamma_cm2']
    source_rate = state['source_rate_s']
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
            / _SPEED_OF_LIGHT_CMS
        )
    return ngamma


def ionization_fraction_rate(state, ngamma):
    """Return the chemistry fraction rate for a float source state."""
    hydrogen_mass_fraction = state['hydrogen_mass_fraction']
    recombination = state['recombination']
    collisional_ionization = state['collisional_ionization']
    sigma = state['sigma_gamma_cm2']
    alpha_value = state['alpha_B_cm3_s']
    beta_value = state['beta_cm3_s']
    return _static_neutral_fraction_rate_value(
        state['rho_g_cm3'],
        state['temperature_K'],
        state['xHI'],
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        recombination=recombination,
        collisional_ionization=collisional_ionization,
        ngamma_cm3=ngamma,
        sigma_gamma_cm2=sigma,
        recombination_coefficient_cm3_s=alpha_value,
        ionization_coefficient_cm3_s=beta_value,
    )


def thermal_rate(state, ngamma):
    """Return thermal source rate for a float source state."""
    hydrogen_mass_fraction = state['hydrogen_mass_fraction']
    recombination = state['recombination']
    collisional_ionization = state['collisional_ionization']
    sigma = state['sigma_gamma_cm2']
    epsilon_gamma = state['epsilon_gamma_erg']
    return _source_thermal_rate_value(
        state['rho_g_cm3'],
        state['temperature_K'],
        state['xHI'],
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        recombination=recombination,
        collisional_ionization=collisional_ionization,
        ngamma_cm3=ngamma,
        sigma_gamma_cm2=sigma,
        epsilon_gamma_erg=epsilon_gamma,
    )


def get_timestep(state, ngamma, remaining_s, dtmax_s):
    """Return a source substep for a float thermo-chemistry state."""
    source_CFL = state['source_CFL']
    dtmin_s = state['dtmin_s']
    candidates = []
    neutral_rate = ionization_fraction_rate(state, ngamma)
    scale = np.where(neutral_rate < 0.0, state['xHI'], 1.0 - state['xHI'])
    valid = (np.abs(neutral_rate) > 0.0) & (scale > 0.0)
    if np.any(valid):
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            neutral_times = scale[valid] / np.abs(neutral_rate[valid])
        neutral_times = neutral_times[np.isfinite(neutral_times) & (neutral_times > 0.0)]
        if len(neutral_times) > 0:
            candidates.append(source_CFL * np.min(neutral_times))

    source_thermal_rate = None
    if state['thermal_coupling']:
        source_thermal_rate = thermal_rate(state, ngamma)
        dudt = source_thermal_rate / state['rho_g_cm3']
        valid = (np.abs(dudt) > 0.0) & (state['specific_energy_erg_g'] > 0.0)
        if np.any(valid):
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                thermal_times = (
                    state['specific_energy_erg_g'][valid]
                    / np.abs(dudt[valid])
                )
            thermal_times = thermal_times[
                np.isfinite(thermal_times) & (thermal_times > 0.0)
            ]
            if len(thermal_times) > 0:
                candidates.append(source_CFL * np.min(thermal_times))

    if len(candidates) == 0:
        return float(min(dtmax_s, remaining_s)), source_thermal_rate
    return float(min(dtmax_s, remaining_s, max(dtmin_s, min(candidates)))), source_thermal_rate


def update_temperature_from_energy(state):
    """Update temperature in a float state from specific energy."""
    mu = 1.0 / (2.0 - np.clip(state['xHI'], 1.0e-12, 1.0))
    state['temperature_K'] = (
        (state['gamma'] - 1.0)
        * mu
        * _PROTON_MASS_G
        * state['specific_energy_erg_g']
        / _BOLTZMANN_ERG_PER_K
    )
    state['temperature_K'] = np.maximum(state['temperature_K'], 1.0)


def ionization_fraction_implicit_update(state, ngamma, dt_s):
    """Implicitly update the chemistry fraction for a float state."""
    xHI = np.clip(np.asarray(state['xHI'], dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    hydrogen_mass_fraction = state['hydrogen_mass_fraction']
    recombination = state['recombination']
    collisional_ionization = state['collisional_ionization']
    sigma = state['sigma_gamma_cm2']
    alpha_value = state['alpha_B_cm3_s']
    updated = _static_neutral_fraction_implicit_update_value(
        state['rho_g_cm3'],
        state['temperature_K'],
        xHI,
        dt_s,
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        recombination=recombination,
        collisional_ionization=collisional_ionization,
        ngamma_cm3=ngamma,
        sigma_gamma_cm2=sigma,
        recombination_coefficient_cm3_s=alpha_value,
    )
    state['xHI'] = np.clip(updated, 1.0e-12, 1.0 - 1.0e-12)


def apply_state(state, fluid, par):
    """Copy a float thermo-chemistry state back to a fluid object."""
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
    state = _fast_source_state(mesh, fluid, par)
    remaining_s = remaining.to_value(unyt.s) if hasattr(remaining, 'to_value') else float(remaining)
    if getattr(par, 'radiative_transfer', False):
        state['ngamma_cm3'] = trace_spherical_photon_density(state)
    sub_dt_s, thermal_rate = get_timestep(
        state,
        state.get('ngamma_cm3'),
        remaining_s,
        remaining_s,
    )
    if thermal_rate is None:
        return sub_dt_s * unyt.s, None
    return sub_dt_s * unyt.s, thermal_rate * (unyt.erg / unyt.cm**3 / unyt.s)


def _fast_source_state(mesh, fluid, par):
    """Return a cgs float snapshot for the fast thermo-chemistry path."""
    interior = slice(par.noghost, par.noghost + par.nogrid)
    state = {
        'interior': interior,
        'boundary_cm': np.asarray(
            mesh.boundary[interior.start : interior.stop + 1].to_value(unyt.cm),
            dtype=float,
        ),
        'width_cm': np.asarray(mesh.xdelta[interior].to_value(unyt.cm), dtype=float),
        'volume_cm3': np.asarray(mesh.vol[interior].to_value(unyt.cm**3), dtype=float),
        'rho_g_cm3': np.asarray(fluid.rho[interior].to_value(unyt.g / unyt.cm**3), dtype=float),
        'temperature_K': np.asarray(fluid.temp[interior].to_value(unyt.K), dtype=float),
        'xHI': np.asarray(
            fluid.xHI[interior] if hasattr(fluid, 'xHI') else np.ones(par.nogrid),
            dtype=float,
        ),
        'nH_cm3': _hydrogen_number_density_value(
            np.asarray(fluid.rho[interior].to_value(unyt.g / unyt.cm**3), dtype=float),
            hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
        ),
        'gamma': getattr(getattr(fluid, 'eos', None), 'gamma', getattr(par, 'gamma', 5.0 / 3.0)),
        'mu': (
            np.asarray(fluid.mu[interior], dtype=float)
            if hasattr(fluid, 'mu')
            else rh.pure_hydrogen_mu(
                np.asarray(
                    fluid.xHI[interior] if hasattr(fluid, 'xHI') else np.ones(par.nogrid),
                    dtype=float,
                ),
                hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
            )
        ),
        'hydrogen_mass_fraction': getattr(par, 'hydrogen_mass_fraction', 1.0),
        'sigma_gamma_cm2': _as_float_array_in_units(
            getattr(par, 'hydrogen_sigma_gamma', rh.DEFAULT_SIGMA_GAMMA),
            _CM2,
        ),
        'source_rate_s': _as_float_array_in_units(
            getattr(par, 'radiative_transfer_source_photon_rate', 0.0 / unyt.s),
            _PER_S,
        ),
        'epsilon_gamma_erg': _as_float_array_in_units(
            getattr(par, 'hydrogen_epsilon_gamma', rh.DEFAULT_EPSILON_GAMMA),
            _ERG,
        ),
        'source_CFL': getattr(par, 'hydrogen_source_CFL', 0.1),
        'dtmin_s': _as_float_array_in_units(
            getattr(par, 'hydrogen_source_dtmin', 0.0 * unyt.s),
            unyt.s,
        ),
        'recombination': getattr(par, 'hydrogen_recombination', True),
        'collisional_ionization': getattr(par, 'hydrogen_collisional_ionization', True),
        'thermal_coupling': getattr(par, 'hydrogen_thermal_coupling', True),
        'hydrogen_update_mu': getattr(par, 'hydrogen_update_mu', False),
        'alpha_B_cm3_s': _as_float_array_in_units(
            getattr(par, 'hydrogen_alpha_B', 0.0 * _CM3_PER_S),
            _CM3_PER_S,
        ),
        'beta_cm3_s': _as_float_array_in_units(
            getattr(par, 'hydrogen_beta', 0.0 * _CM3_PER_S),
            _CM3_PER_S,
        ),
    }
    if state['thermal_coupling']:
        state['vel_cm_s'] = np.asarray(
            fluid.vel[interior].to_value(unyt.cm / unyt.s),
            dtype=float,
        )
        state['specific_total_energy_erg_g'] = (
            np.asarray(fluid.Energy[interior].to_value(unyt.erg), dtype=float)
            / np.asarray(fluid.Mass[interior].to_value(unyt.g), dtype=float)
        )
        state['specific_kinetic_energy_erg_g'] = 0.5 * state['vel_cm_s']**2
        _fast_update_temperature_from_energy(state)
    return state


def _fast_update_temperature_from_energy(state):
    """Update float temperature from total specific energy and mean molecular weight."""
    internal_specific = np.maximum(
        state['specific_total_energy_erg_g'] - state['specific_kinetic_energy_erg_g'],
        0.0,
    )
    if state.get('hydrogen_update_mu', False):
        state['mu'] = rh.pure_hydrogen_mu(
            state['xHI'],
            hydrogen_mass_fraction=state['hydrogen_mass_fraction'],
        )
    state['temperature_K'] = (
        (state['gamma'] - 1.0)
        * state['mu']
        * _PROTON_MASS_G
        * internal_specific
        / _BOLTZMANN_ERG_PER_K
    )


def _fast_apply_thermal_source(state, thermal_rate_erg_cm3_s, dt_s):
    """Apply thermal source terms in float cgs units."""
    state['specific_total_energy_erg_g'] += (
        thermal_rate_erg_cm3_s / state['rho_g_cm3'] * dt_s
    )
    state['specific_total_energy_erg_g'] = np.maximum(
        state['specific_total_energy_erg_g'],
        state['specific_kinetic_energy_erg_g'],
    )
    _fast_update_temperature_from_energy(state)


def _fast_sync_state_to_fluid(state, fluid, par):
    """Copy a float thermo-chemistry state back to the fluid container."""
    interior = state['interior']
    fluid.xHI[interior] = state['xHI']
    if hasattr(fluid, 'ngamma') and state.get('ngamma_cm3') is not None:
        fluid.ngamma[interior] = state['ngamma_cm3'] / unyt.cm**3
    if hasattr(fluid, 'mu'):
        fluid.mu[interior] = state['mu']
    fluid.temp[interior] = state['temperature_K'] * unyt.K
    if state.get('thermal_coupling', False):
        fluid.pre[interior] = (
            state['specific_total_energy_erg_g'] - state['specific_kinetic_energy_erg_g']
        ) * fluid.rho[interior] * (fluid.eos.gamma - 1.0)
        fluid.Energy[interior] = (
            state['specific_total_energy_erg_g'] * fluid.Mass[interior]
        ).to(fluid.Energy.units)
    if state.get('hydrogen_update_mu', False) and hasattr(fluid, 'xHI') and getattr(getattr(fluid, 'eos', None), 'gamma', None) is not None:
        fluid.SetHydrogenMu(
            hydrogen_mass_fraction=state['hydrogen_mass_fraction']
        )
        if state.get('thermal_coupling', False):
            fluid.SetPressure()


def apply_thermochemistry_fast(dt, mesh, fluid, par):
    """Fast source update for RT-coupled thermo-chemistry tests."""
    if not thermochemistry_enabled(fluid, par):
        return 0

    state = _fast_source_state(mesh, fluid, par)
    interior = state['interior']
    remaining_s = dt.to_value(unyt.s)
    zero_time_s = 0.0
    source_steps = 0
    while remaining_s > zero_time_s:
        if getattr(par, 'radiative_transfer', False):
            state['ngamma_cm3'] = trace_spherical_photon_density(state)
        if state['hydrogen_update_mu']:
            state['mu'] = rh.pure_hydrogen_mu(
                state['xHI'],
                hydrogen_mass_fraction=state['hydrogen_mass_fraction'],
            )
        if state['thermal_coupling']:
            _fast_update_temperature_from_energy(state)
        sub_dt_s, thermal_rate = get_timestep(
            state,
            state.get('ngamma_cm3'),
            remaining_s,
            remaining_s,
        )
        if not np.isfinite(sub_dt_s) or sub_dt_s <= zero_time_s:
            sub_dt_s = remaining_s
        if sub_dt_s > remaining_s:
            sub_dt_s = remaining_s

        if state['thermal_coupling']:
            _fast_apply_thermal_source(state, thermal_rate, sub_dt_s)
        ionization_fraction_implicit_update(
            state,
            state.get('ngamma_cm3'),
            sub_dt_s,
        )
        if state['hydrogen_update_mu']:
            state['mu'] = rh.pure_hydrogen_mu(
                state['xHI'],
                hydrogen_mass_fraction=state['hydrogen_mass_fraction'],
            )
        if state['thermal_coupling']:
            _fast_update_temperature_from_energy(state)
        remaining_s -= sub_dt_s
        source_steps += 1
    if getattr(par, 'radiative_transfer', False):
        state['ngamma_cm3'] = trace_spherical_photon_density(state)
    _fast_sync_state_to_fluid(state, fluid, par)
    return source_steps


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

    def source_state(self, mesh, fluid, par):
        return source_state(mesh, fluid, par)

    def trace_spherical_photon_density(self, state):
        return trace_spherical_photon_density(state)

    def ionization_fraction_rate(self, state, ngamma):
        return ionization_fraction_rate(state, ngamma)

    def thermal_rate(self, state, ngamma):
        return thermal_rate(state, ngamma)

    def get_timestep(self, state, ngamma, remaining_s, dtmax_s):
        return get_timestep(
            state,
            ngamma,
            remaining_s,
            dtmax_s,
        )

    def update_temperature_from_energy(self, state):
        return update_temperature_from_energy(state)

    def ionization_fraction_implicit_update(self, state, ngamma, dt_s):
        return ionization_fraction_implicit_update(state, ngamma, dt_s)

    def apply_state(self, state, fluid, par):
        return apply_state(state, fluid, par)

    def get_source_timestep_fast(self, mesh, fluid, par, remaining):
        return get_thermochemistry_source_timestep_fast(mesh, fluid, par, remaining)

    def apply_fast(self, dt, mesh, fluid, par):
        return apply_thermochemistry_fast(dt, mesh, fluid, par)
