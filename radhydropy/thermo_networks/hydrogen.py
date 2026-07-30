"""Hydrogen thermo-chemistry network.

This module contains the current hydrogen-only rate network.  The public
dispatcher in :mod:`radhydropy.thermo_chemistry` calls this through the
``HydrogenNetwork`` interface.
"""

import numpy as np
import unyt

from radhydropy.constants import (
    BOLTZMANN_CONSTANT_CGS,
    DEFAULT_EPSILON_GAMMA,
    DEFAULT_SIGMA_GAMMA,
    PROTON_MASS_CGS,
    SPEED_OF_LIGHT_CGS,
)
import radhydropy.chemistry_species.hydrogen as rh
import radhydropy.radiative_transfer as rrt
import radhydropy.utils as ru
from radhydropy.units import _code_units, cgs_to_code_value, code_to_cgs_value, to_code_value
from radhydropy.arrays import as_named_array
from radhydropy.thermo_networks.base import ThermochemistryNetwork


SPEED_OF_LIGHT_CMS = SPEED_OF_LIGHT_CGS


def _code_quantity_to_cgs(value, unit):
    code_value = to_code_value(value, unit)
    return code_to_cgs_value(code_value, unit)


def _cgs_quantity_to_code(value, unit):
    return cgs_to_code_value(value, unit)


def _optional_code_quantity_to_cgs(value, unit, default=None):
    if value is None:
        if default is None:
            return None
        return code_to_cgs_value(default, unit)
    return _code_quantity_to_cgs(value, unit)


def _cgs_alpha_B(temperature_K):
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


def _cgs_beta(temperature_K):
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


def _cgs_gamma_line_eHI(temperature_K):
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


def _cgs_gamma_ion_eHI(temperature_K):
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


def _cgs_gamma_ff_eHII(temperature_K):
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


def _cgs_gamma_B_eHII(temperature_K):
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


def _cgs_hydrogen_number_density(rho_g_cm3, hydrogen_mass_fraction=1.0):
    return hydrogen_mass_fraction * np.asarray(rho_g_cm3, dtype=float) / PROTON_MASS_CGS


def _cgs_photoionization_frequency(ngamma_cm3, sigma_gamma_cm2):
    return SPEED_OF_LIGHT_CMS * np.asarray(sigma_gamma_cm2, dtype=float) * np.asarray(
        ngamma_cm3,
        dtype=float,
    )


def _cgs_source_thermal_rate(
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
    nH = _cgs_hydrogen_number_density(rho_g_cm3, hydrogen_mass_fraction)
    eHI_cooling = _cgs_gamma_line_eHI(temperature_K)
    if collisional_ionization:
        eHI_cooling += _cgs_gamma_ion_eHI(temperature_K)
    eHII_cooling = _cgs_gamma_ff_eHII(temperature_K)
    if recombination:
        eHII_cooling += _cgs_gamma_B_eHII(temperature_K)
    cooling = nH**2 * (xHI * ionized * eHI_cooling + ionized**2 * eHII_cooling)
    if ngamma_cm3 is None:
        heating = np.zeros_like(cooling, dtype=float)
    else:
        heating = (
            nH
            * xHI
            * np.asarray(epsilon_gamma_erg, dtype=float)
            * _cgs_photoionization_frequency(ngamma_cm3, sigma_gamma_cm2)
        )
    return heating - cooling


def _cgs_static_neutral_fraction_rate(
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
    nH = _cgs_hydrogen_number_density(rho_g_cm3, hydrogen_mass_fraction)
    if recombination_coefficient_cm3_s is None:
        recombination_coefficient_cm3_s = _cgs_alpha_B(temperature_K)
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
        ionization_coefficient_cm3_s = _cgs_beta(temperature_K)
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
        photoionization_rate_s = _cgs_photoionization_frequency(
            ngamma_cm3,
            sigma_gamma_cm2,
        )
    return (
        ionized**2 * nH * recombination_coefficient_cm3_s
        - xHI * ionized * nH * ionization_coefficient_cm3_s
        - xHI * photoionization_rate_s
    )


def _cgs_static_neutral_fraction_implicit_update(
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
    nH = _cgs_hydrogen_number_density(rho_g_cm3, hydrogen_mass_fraction)
    if recombination:
        if recombination_coefficient_cm3_s is None:
            recombination_coefficient_cm3_s = _cgs_alpha_B(temperature_K)
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
            ionization_coefficient_cm3_s = _cgs_beta(temperature_K)
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
        photoionization_rate_s = _cgs_photoionization_frequency(
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
    sqrt_discriminant = np.sqrt(discriminant)
    updated = np.array(xHI, copy=True, dtype=float)
    nonzero = a != 0.0
    if np.any(nonzero):
        root_low = np.divide(
            -b - sqrt_discriminant,
            2.0 * a,
            out=np.array(xHI, copy=True, dtype=float),
            where=nonzero,
        )
        root_high = np.divide(
            -b + sqrt_discriminant,
            2.0 * a,
            out=np.array(xHI, copy=True, dtype=float),
            where=nonzero,
        )
        selected = np.where(
            (root_low >= 0.0) & (root_low <= 1.0),
            root_low,
            root_high,
        )
        updated = np.where(nonzero, selected, updated)
    zero = ~nonzero
    if np.any(zero):
        linear_denominator = b[zero]
        linear_numerator = -c[zero]
        linear_updated = np.divide(
            linear_numerator,
            linear_denominator,
            out=np.array(xHI[zero], copy=True, dtype=float),
            where=linear_denominator != 0.0,
        )
        updated[zero] = linear_updated
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


def advect_ionization_fraction(dt, mesh, fluid, par, old_mass, mass_flux):
    """Advect the chemistry fraction consistently with the mass flux."""
    if not hasattr(fluid, 'xHI'):
        return
    face_area = mesh.area
    x_left = np.roll(fluid.xHI, 1)
    x_right = fluid.xHI
    x_face = np.where(np.asarray(mass_flux, dtype=float) >= 0.0, x_left, x_right)
    neutral_mass = np.asarray(fluid.xHI) * old_mass
    neutral_flux = x_face * mass_flux
    neutral_mass += (
        neutral_flux * face_area
        - np.roll(neutral_flux * face_area, -1)
    ) * dt
    xHI = ru.SafeDivide(neutral_mass, fluid.Mass)
    fluid.xHI = rh.clip_neutral_fraction(np.asarray(xHI, dtype=float))


def source_state(mesh, fluid, par):
    """Return a float state for fixed-density thermo-chemistry tests."""
    code = _code_units(par)
    if code is None:
        raise ValueError("hydrogen thermo-chemistry requires par.code_units")
    kpc_in_cm = float((1.0 * unyt.kpc).to_value(unyt.cm))
    interior = interior_slice(par)
    boundary = as_named_array(
        _code_quantity_to_cgs(
            mesh.boundary[interior.start : interior.stop + 1],
            code.length_unit,
        )
    )
    xHI = as_named_array(np.asarray(fluid.xHI[interior], dtype=float).copy())
    temperature = as_named_array(
        _code_quantity_to_cgs(fluid.temp[interior], code.temperature_unit).copy()
    )
    rho = as_named_array(_code_quantity_to_cgs(fluid.rho[interior], code.density_unit))
    gamma = getattr(
        getattr(fluid, 'eos', None),
        'gamma',
        getattr(par, 'gamma', 5.0 / 3.0),
    )
    mu = 1.0 / (2.0 - np.clip(xHI, 1.0e-12, 1.0))
    specific_energy = (
        BOLTZMANN_CONSTANT_CGS
        * temperature
        / ((gamma - 1.0) * mu * PROTON_MASS_CGS)
    )
    sigma_gamma = _optional_code_quantity_to_cgs(
        getattr(par, 'hydrogen_sigma_gamma', None),
        code.area_unit,
        default=DEFAULT_SIGMA_GAMMA,
    )
    source_rate = _optional_code_quantity_to_cgs(
        getattr(par, 'radiative_transfer_source_photon_rate', None),
        code.time_unit ** -1,
        default=0.0,
    )
    epsilon_gamma = _optional_code_quantity_to_cgs(
        getattr(par, 'hydrogen_epsilon_gamma', None),
        code.energy_unit,
        default=DEFAULT_EPSILON_GAMMA,
    )
    alpha_B = getattr(par, 'hydrogen_alpha_B', None)
    if alpha_B is not None:
        alpha_B = _code_quantity_to_cgs(alpha_B, code.volume_unit / code.time_unit)
    beta = getattr(par, 'hydrogen_beta', None)
    if beta is not None:
        beta = _code_quantity_to_cgs(beta, code.volume_unit / code.time_unit)
    return {
        'interior': interior,
        'boundary_cm': boundary,
        'width_cm': np.diff(boundary),
        'volume_cm3': as_named_array(
            _code_quantity_to_cgs(mesh.vol[interior], code.volume_unit)
        ),
        'radius_cm': as_named_array(
            _code_quantity_to_cgs(mesh.coordinate[interior], code.length_unit)
        ),
        'radius_kpc': np.asarray(
            _code_quantity_to_cgs(mesh.coordinate[interior], code.length_unit) / kpc_in_cm,
            dtype=float,
        ),
        'xHI': xHI,
        'temperature_K': temperature,
        'specific_energy_erg_g': specific_energy,
        'rho_g_cm3': rho,
        'nH_cm3': rho * getattr(par, 'hydrogen_mass_fraction', 1.0) / PROTON_MASS_CGS,
        'gamma': gamma,
        'hydrogen_mass_fraction': getattr(par, 'hydrogen_mass_fraction', 1.0),
        'sigma_gamma_cm2': sigma_gamma,
        'source_rate_s': source_rate,
        'epsilon_gamma_erg': epsilon_gamma,
        'source_CFL': getattr(par, 'hydrogen_source_CFL', 0.1),
        'dtmin_s': _optional_code_quantity_to_cgs(
            getattr(par, 'hydrogen_source_dtmin', None),
            code.time_unit,
            default=0.0,
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


def trace_spherical_tau(mesh, rho, xHI, hydrogen_mass_fraction, sigma_gamma):
    """Return the hydrogen optical depth per cell.

    This helper now requires ``mesh.code_units`` so the cgs conversion is
    explicit at the mesh boundary.
    """
    code = mesh.code_units
    rho_g_cm3 = _code_quantity_to_cgs(rho, code.density_unit)
    sigma_cm2 = _code_quantity_to_cgs(rh.photon_cross_section(sigma_gamma), code.area_unit)
    width_cm = _code_quantity_to_cgs(
        np.abs(mesh.boundary[1:] - mesh.boundary[:-1]),
        code.length_unit,
    )
    nH = _cgs_hydrogen_number_density(rho_g_cm3, hydrogen_mass_fraction)
    xHI = rh.clip_neutral_fraction(xHI)
    tau = sigma_cm2 * nH * xHI * width_cm
    return as_named_array(np.maximum(tau, 0.0))


def ionization_fraction_rate(state, ngamma):
    """Return the chemistry fraction rate for a float source state."""
    hydrogen_mass_fraction = state['hydrogen_mass_fraction']
    recombination = state['recombination']
    collisional_ionization = state['collisional_ionization']
    sigma = state['sigma_gamma_cm2']
    alpha_value = state['alpha_B_cm3_s']
    beta_value = state['beta_cm3_s']
    return _cgs_static_neutral_fraction_rate(
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
    return _cgs_source_thermal_rate(
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


def get_timestep(state, ngamma, remaining_s, dtmax_s, verbose=False):
    """Return a source substep for a float thermo-chemistry state."""
    source_CFL = state['source_CFL']
    dtmin_s = state['dtmin_s']
    candidates = []
    debug_lines = []
    neutral_rate = ionization_fraction_rate(state, ngamma)
    scale = np.where(neutral_rate < 0.0, state['xHI'], 1.0 - state['xHI'])
    valid = (np.abs(neutral_rate) > 0.0) & (scale > 0.0)
    if np.any(valid):
        valid_cells = np.where(valid)[0]
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            neutral_times = scale[valid] / np.abs(neutral_rate[valid])
        neutral_mask = np.isfinite(neutral_times) & (neutral_times > 0.0)
        neutral_times = neutral_times[neutral_mask]
        neutral_cells = valid_cells[neutral_mask]
        if len(neutral_times) > 0:
            neutral_min_index = int(np.argmin(neutral_times))
            neutral_dt = source_CFL * neutral_times[neutral_min_index]
            neutral_cell = int(neutral_cells[neutral_min_index])
            candidates.append(neutral_dt)
            if verbose:
                debug_lines.append(
                    '[source dt] neutral limiter cell=%d rate=%s scale=%s candidate=%s'
                    % (
                        neutral_cell,
                        neutral_rate[neutral_cell],
                        scale[neutral_cell],
                        neutral_dt,
                    )
                )

    source_thermal_rate = None
    if state['thermal_coupling']:
        source_thermal_rate = thermal_rate(state, ngamma)
        dudt = source_thermal_rate / state['rho_g_cm3']
        valid = (np.abs(dudt) > 0.0) & (state['specific_energy_erg_g'] > 0.0)
        if np.any(valid):
            valid_cells = np.where(valid)[0]
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                thermal_times = (
                    state['specific_energy_erg_g'][valid]
                    / np.abs(dudt[valid])
                )
            thermal_mask = np.isfinite(thermal_times) & (thermal_times > 0.0)
            thermal_times = thermal_times[thermal_mask]
            thermal_cells = valid_cells[thermal_mask]
            if len(thermal_times) > 0:
                thermal_min_index = int(np.argmin(thermal_times))
                thermal_dt = source_CFL * thermal_times[thermal_min_index]
                thermal_cell = int(thermal_cells[thermal_min_index])
                candidates.append(thermal_dt)
                if verbose:
                    debug_lines.append(
                        '[source dt] thermal limiter cell=%d dudt=%s energy=%s candidate=%s'
                        % (
                            thermal_cell,
                            dudt[thermal_cell],
                            state['specific_energy_erg_g'][thermal_cell],
                            thermal_dt,
                        )
                    )

    if len(candidates) == 0:
        dt = float(min(dtmax_s, remaining_s))
    else:
        dt = float(min(dtmax_s, remaining_s, max(dtmin_s, min(candidates))))
    if verbose:
        print(
            '[source dt] remaining=%s dtmax=%s dtmin=%s selected=%s'
            % (remaining_s, dtmax_s, dtmin_s, dt)
        )
        for line in debug_lines:
            print(line)
    return dt, source_thermal_rate


def update_temperature_from_energy(state):
    """Update temperature in a float state from specific energy."""
    mu = 1.0 / (2.0 - np.clip(state['xHI'], 1.0e-12, 1.0))
    state['temperature_K'] = (
        (state['gamma'] - 1.0)
        * mu
        * PROTON_MASS_CGS
        * state['specific_energy_erg_g']
        / BOLTZMANN_CONSTANT_CGS
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
    updated = _cgs_static_neutral_fraction_implicit_update(
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
    code = _code_units(par)
    if code is None:
        raise ValueError("hydrogen thermo-chemistry requires par.code_units")
    if hasattr(fluid, 'ngamma') and 'ngamma_cm3' in state:
        fluid.ngamma[interior] = _cgs_quantity_to_code(
            state['ngamma_cm3'],
            code.number_density_unit,
        )
    if hasattr(fluid, 'temp') and 'temperature_K' in state:
        fluid.temp[interior] = _cgs_quantity_to_code(
            state['temperature_K'],
            code.temperature_unit,
        )
    if hasattr(fluid, 'xHI') and getattr(getattr(fluid, 'eos', None), 'gamma', None) is not None:
        fluid.SetHydrogenMu(
            hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0)
        )
        fluid.SetPressure()
    fluid.time = _cgs_quantity_to_code(state['time_s'], code.time_unit)


def get_thermochemistry_source_timestep_fast(mesh, fluid, par, remaining):
    """Return a source substep for RT-coupled heating/chemistry."""
    state = _fast_source_state(mesh, fluid, par)
    code = _code_units(par)
    if code is None:
        raise ValueError("hydrogen thermo-chemistry requires par.code_units")
    remaining_s = _code_quantity_to_cgs(remaining, code.time_unit)
    if getattr(par, 'radiative_transfer', False):
        state['ngamma_cm3'] = rrt.trace_photon_density(state, par)
    sub_dt_s, thermal_rate = get_timestep(
        state,
        state.get('ngamma_cm3'),
        remaining_s,
        remaining_s,
    )
    if thermal_rate is None:
        return sub_dt_s, None
    return sub_dt_s, thermal_rate


def _fast_source_state(mesh, fluid, par):
    """Return a cgs float snapshot for the fast thermo-chemistry path."""
    code = _code_units(par)
    if code is None:
        raise ValueError("hydrogen thermo-chemistry requires par.code_units")
    interior = slice(par.noghost, par.noghost + par.nogrid)
    rho_g_cm3 = _code_quantity_to_cgs(fluid.rho[interior], code.density_unit)
    temperature_K = _code_quantity_to_cgs(fluid.temp[interior], code.temperature_unit)
    vel_cm_s = _code_quantity_to_cgs(fluid.vel[interior], code.velocity_unit)
    mass_g = _code_quantity_to_cgs(fluid.Mass[interior], code.mass_unit)
    energy_erg = _code_quantity_to_cgs(fluid.Energy[interior], code.energy_unit)
    state = {
        'interior': interior,
        'boundary_cm': as_named_array(
            _code_quantity_to_cgs(mesh.boundary[interior.start : interior.stop + 1], code.length_unit)
        ),
        'width_cm': as_named_array(_code_quantity_to_cgs(mesh.xdelta[interior], code.length_unit)),
        'volume_cm3': as_named_array(_code_quantity_to_cgs(mesh.vol[interior], code.volume_unit)),
        'rho_g_cm3': rho_g_cm3,
        'temperature_K': temperature_K,
        'xHI': as_named_array(
            fluid.xHI[interior] if hasattr(fluid, 'xHI') else np.ones(par.nogrid),
        ),
        'nH_cm3': rho_g_cm3 * getattr(par, 'hydrogen_mass_fraction', 1.0) / PROTON_MASS_CGS,
        'gamma': getattr(getattr(fluid, 'eos', None), 'gamma', getattr(par, 'gamma', 5.0 / 3.0)),
        'mu': (
            np.asarray(fluid.mu[interior], dtype=float)
            if hasattr(fluid, 'mu')
            else rh.mean_molecular_weight_mu(
                np.asarray(
                    fluid.xHI[interior] if hasattr(fluid, 'xHI') else np.ones(par.nogrid),
                    dtype=float,
                ),
                hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
            )
        ),
        'hydrogen_mass_fraction': getattr(par, 'hydrogen_mass_fraction', 1.0),
        'sigma_gamma_cm2': _optional_code_quantity_to_cgs(
            getattr(par, 'hydrogen_sigma_gamma', None),
            code.area_unit,
            default=DEFAULT_SIGMA_GAMMA,
        ),
        'ngamma_cm3': (
            _code_quantity_to_cgs(fluid.ngamma[interior], code.number_density_unit)
            if (
                getattr(par, 'hydrogen_radiation_field', False)
                or getattr(par, 'radiative_transfer', False)
            )
            and hasattr(fluid, 'ngamma')
            else None
        ),
        'source_rate_s': _optional_code_quantity_to_cgs(
            getattr(par, 'radiative_transfer_source_photon_rate', None),
            1.0 / code.time_unit,
            default=0.0,
        ),
        'epsilon_gamma_erg': _optional_code_quantity_to_cgs(
            getattr(par, 'hydrogen_epsilon_gamma', None),
            code.energy_unit,
            default=DEFAULT_EPSILON_GAMMA,
        ),
        'source_CFL': getattr(par, 'hydrogen_source_CFL', 0.1),
        'dtmin_s': _optional_code_quantity_to_cgs(
            getattr(par, 'hydrogen_source_dtmin', None),
            code.time_unit,
            default=0.0,
        ),
        'recombination': getattr(par, 'hydrogen_recombination', True),
        'collisional_ionization': getattr(par, 'hydrogen_collisional_ionization', True),
        'thermal_coupling': getattr(par, 'hydrogen_thermal_coupling', True),
        'hydrogen_update_mu': getattr(par, 'hydrogen_update_mu', False),
        'alpha_B_cm3_s': _optional_code_quantity_to_cgs(
            getattr(par, 'hydrogen_alpha_B', None),
            code.volume_unit / code.time_unit,
            default=None,
        ),
        'beta_cm3_s': _optional_code_quantity_to_cgs(
            getattr(par, 'hydrogen_beta', None),
            code.volume_unit / code.time_unit,
            default=None,
        ),
    }
    if state['thermal_coupling']:
        state['vel_cm_s'] = vel_cm_s
        state['specific_total_energy_erg_g'] = energy_erg / mass_g
        state['specific_kinetic_energy_erg_g'] = 0.5 * state['vel_cm_s']**2
        state['specific_energy_erg_g'] = np.maximum(
            state['specific_total_energy_erg_g'] - state['specific_kinetic_energy_erg_g'],
            0.0,
        )
        _fast_update_temperature_from_energy(state)
    return state


def _fast_update_temperature_from_energy(state):
    """Update float temperature from total specific energy and mean molecular weight."""
    internal_specific = np.maximum(
        state['specific_total_energy_erg_g'] - state['specific_kinetic_energy_erg_g'],
        0.0,
    )
    if state.get('hydrogen_update_mu', False):
        state['mu'] = rh.mean_molecular_weight_mu(
            state['xHI'],
            hydrogen_mass_fraction=state['hydrogen_mass_fraction'],
        )
    state['temperature_K'] = (
        (state['gamma'] - 1.0)
        * state['mu']
        * PROTON_MASS_CGS
        * internal_specific
        / BOLTZMANN_CONSTANT_CGS
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
        code = _code_units(par)
        fluid.ngamma[interior] = _cgs_quantity_to_code(
            state['ngamma_cm3'],
            code.number_density_unit,
        )
    if hasattr(fluid, 'mu'):
        fluid.mu[interior] = state['mu']
    code = _code_units(par)
    fluid.temp[interior] = _cgs_quantity_to_code(state['temperature_K'], code.temperature_unit)
    if state.get('thermal_coupling', False):
        specific_internal_energy = (
            state['specific_total_energy_erg_g']
            - state['specific_kinetic_energy_erg_g']
        )
        fluid.pre[interior] = (
            specific_internal_energy
            * np.asarray(fluid.rho[interior], dtype=float)
            * (fluid.eos.gamma - 1.0)
        )
        fluid.Energy[interior] = (
            state['specific_total_energy_erg_g']
            * np.asarray(fluid.Mass[interior], dtype=float)
        )
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
    code = _code_units(par)
    if code is None:
        raise ValueError("hydrogen thermo-chemistry requires par.code_units")
    remaining_s = _code_quantity_to_cgs(dt, code.time_unit)
    zero_time_s = 0.0
    source_steps = 0
    rt_update_interval = max(
        1,
        int(getattr(par, 'radiative_transfer_update_interval', 1)),
    )
    rt_step_counter = int(getattr(par, '_radiative_transfer_hydro_step', 0))
    while remaining_s > zero_time_s:
        if getattr(par, 'radiative_transfer', False) and (
            rt_step_counter % rt_update_interval == 0
            or 'ngamma_cm3' not in state
        ):
            state['ngamma_cm3'] = rrt.trace_photon_density(state, par)
        if state['hydrogen_update_mu']:
            state['mu'] = rh.mean_molecular_weight_mu(
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
            verbose=getattr(par, 'verbose', 0) >= 2,
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
            state['mu'] = rh.mean_molecular_weight_mu(
                state['xHI'],
                hydrogen_mass_fraction=state['hydrogen_mass_fraction'],
            )
        if state['thermal_coupling']:
            _fast_update_temperature_from_energy(state)
        remaining_s -= sub_dt_s
        source_steps += 1
    setattr(par, '_radiative_transfer_hydro_step', rt_step_counter + 1)
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

    def source_state(self, mesh, fluid, par):
        return source_state(mesh, fluid, par)

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
