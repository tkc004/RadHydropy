"""Hydrogen thermo-chemistry network.

This module contains the current hydrogen-only rate network.  The public
dispatcher in :mod:`radhydropy.thermo_chemistry` calls this through the
``HydrogenNetwork`` interface.
"""

import copy

import numpy as np
import unyt
from types import SimpleNamespace

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
from radhydropy.units import (
    _code_units,
    from_unit_value,
    to_unit_value,
)
from radhydropy.arrays import as_named_array
from radhydropy.thermo_networks.base import ThermochemistryNetwork
from radhydropy.thermo_networks.compton import cmb_compton_rate




def _require_numeric_array(value, label):
    if hasattr(value, "to_value"):
        raise TypeError(f"{label} must be a plain numeric array, not a unyt quantity")
    return np.asarray(value, dtype=float)

def _optional_numeric_value(value, unit, default=None):
    if value is None:
        if default is None:
            return None
        return to_unit_value(default, unit)
    return to_unit_value(value, unit)


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


def _cgs_alpha_A(temperature_K):
    """H II case-A recombination coefficient (Hui & Gnedin 1997)."""
    temperature_K = np.asarray(temperature_K, dtype=float)
    result = np.zeros_like(temperature_K, dtype=float)
    valid = temperature_K > 0.0
    if np.any(valid):
        lam = 315614.0 / temperature_K[valid]
        result[valid] = (
            1.269e-13
            * lam**1.503
            * (1.0 + (lam / 0.522) ** 0.470) ** -1.923
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


def collisional_equilibrium_neutral_fraction(temperature_K):
    """Return the H I fraction in collisional ionization equilibrium."""
    alpha = _cgs_alpha_B(temperature_K)
    beta = _cgs_beta(temperature_K)
    total = alpha + beta
    return np.divide(alpha, total, out=np.ones_like(alpha), where=total > 0.0)


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


def _cgs_gamma_A_eHII(temperature_K):
    """H II case-A recombination cooling coefficient."""
    temperature_K = np.asarray(temperature_K, dtype=float)
    result = np.zeros_like(temperature_K, dtype=float)
    valid = temperature_K > 0.0
    if np.any(valid):
        temp = temperature_K[valid]
        lam = 315614.0 / temp
        result[valid] = (
            1.778e-29
            * temp
            * lam**1.965
            * (1.0 + (lam / 0.541) ** 0.502) ** -2.697
        )
    return result


def _cgs_hydrogen_number_density(rho_g_cm3, hydrogen_mass_fraction=1.0):
    return hydrogen_mass_fraction * np.asarray(rho_g_cm3, dtype=float) / PROTON_MASS_CGS


def _cgs_photoionization_frequency(ngamma_cm3, sigma_gamma_cm2):
    ngamma = np.asarray(ngamma_cm3, dtype=float)
    sigma = np.asarray(sigma_gamma_cm2, dtype=float)
    if ngamma.ndim > 1 and sigma.ndim == 1:
        sigma = sigma[:, None]
    rate = SPEED_OF_LIGHT_CGS * sigma * ngamma
    return np.sum(rate, axis=0) if np.ndim(rate) > 1 else rate


def _cgs_source_thermal_rate(
    rho_g_cm3,
    temperature_K,
    xHI,
    hydrogen_mass_fraction=1.0,
    recombination=True,
    collisional_ionization=True,
    atomic_cooling=True,
    ngamma_cm3=None,
    sigma_gamma_cm2=1.0,
    epsilon_gamma_erg=0.0,
    compton_cmb_enabled=False,
    compton_cmb_redshift=0.0,
    cmb_temperature_0_K=2.7255,
):
    xHI = np.clip(np.asarray(xHI, dtype=float), 0.0, 1.0)
    ionized = 1.0 - xHI
    nH = _cgs_hydrogen_number_density(rho_g_cm3, hydrogen_mass_fraction)
    if atomic_cooling:
        eHI_cooling = _cgs_gamma_line_eHI(temperature_K)
        if collisional_ionization:
            eHI_cooling += _cgs_gamma_ion_eHI(temperature_K)
        eHII_cooling = _cgs_gamma_ff_eHII(temperature_K)
        if recombination:
            eHII_cooling += _cgs_gamma_B_eHII(temperature_K)
    else:
        eHI_cooling = np.zeros_like(temperature_K, dtype=float)
        eHII_cooling = np.zeros_like(temperature_K, dtype=float)
    cooling = nH**2 * (xHI * ionized * eHI_cooling + ionized**2 * eHII_cooling)
    if ngamma_cm3 is None:
        heating = np.zeros_like(cooling, dtype=float)
    else:
        ngamma = np.asarray(ngamma_cm3, dtype=float)
        sigma = np.asarray(sigma_gamma_cm2, dtype=float)
        epsilon = np.asarray(epsilon_gamma_erg, dtype=float)
        if ngamma.ndim > 1:
            if sigma.ndim == 1:
                sigma = sigma[:, None]
            if epsilon.ndim == 1:
                epsilon = epsilon[:, None]
        photoheating_per_atom = (
            SPEED_OF_LIGHT_CGS
            * epsilon
            * sigma
            * ngamma
        )
        if np.ndim(photoheating_per_atom) > 1:
            photoheating_per_atom = np.sum(photoheating_per_atom, axis=0)
        heating = nH * xHI * photoheating_per_atom
    electron_density = nH * ionized
    return heating - cooling + cmb_compton_rate(
        temperature_K,
        electron_density,
        enabled=compton_cmb_enabled,
        redshift=compton_cmb_redshift,
        cmb_temperature_0_K=cmb_temperature_0_K,
    )


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
    if not recombination:
        recombination_coefficient_cm3_s = np.zeros_like(
            temperature_K,
            dtype=float,
        )
    elif recombination_coefficient_cm3_s is None:
        recombination_coefficient_cm3_s = _cgs_alpha_B(temperature_K)
    else:
        recombination_coefficient_cm3_s = np.asarray(
            recombination_coefficient_cm3_s,
            dtype=float,
        )
    if not collisional_ionization:
        ionization_coefficient_cm3_s = np.zeros_like(
            temperature_K,
            dtype=float,
        )
    elif ionization_coefficient_cm3_s is None:
        ionization_coefficient_cm3_s = _cgs_beta(temperature_K)
    else:
        ionization_coefficient_cm3_s = np.asarray(
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
        raise ValueError("hydrogen thermo-chemistry requires par.CodeUnits")
    kpc_in_cm = float((1.0 * unyt.kpc).to_value(unyt.cm))
    interior = interior_slice(par)
    boundary = as_named_array(
        to_unit_value(
            mesh.boundary[interior.start : interior.stop + 1],
            code.length_unit,
        )
    )
    xHI = as_named_array(np.asarray(fluid.xHI[interior], dtype=float).copy())
    temperature = as_named_array(
        to_unit_value(fluid.temp[interior], code.temperature_unit).copy()
    )
    rho = as_named_array(to_unit_value(fluid.rho[interior], code.density_unit))
    gamma = getattr(
        getattr(fluid, 'eos', None),
        'gamma',
        getattr(par, 'gamma', 5.0 / 3.0),
    )
    scaling = _fast_source_scaling(fluid, par, gamma)
    mu = 1.0 / (2.0 - np.clip(xHI, 1.0e-12, 1.0))
    temperature_physical = temperature / scaling['temperature_factor']
    rho_physical = rho / scaling['density_factor']
    specific_energy = (
        BOLTZMANN_CONSTANT_CGS
        * temperature_physical
        / ((gamma - 1.0) * mu * PROTON_MASS_CGS)
    )
    sigma_parameter = getattr(
        par,
        'radiation_group_sigma_gamma',
        None,
    ) if getattr(par, 'radiation_group_edges_eV', None) is not None else getattr(
        par,
        'hydrogen_sigma_gamma',
        None,
    )
    sigma_gamma = _optional_numeric_value(
        sigma_parameter,
        code.area_unit,
        default=DEFAULT_SIGMA_GAMMA,
    )
    source_rate = _optional_numeric_value(
        getattr(par, 'radiative_transfer_source_photon_rate', None),
        code.time_unit ** -1,
        default=0.0,
    )
    epsilon_parameter = getattr(
        par,
        'radiation_group_epsilon_gamma',
        None,
    ) if getattr(par, 'radiation_group_edges_eV', None) is not None else getattr(
        par,
        'hydrogen_epsilon_gamma',
        None,
    )
    epsilon_gamma = _optional_numeric_value(
        epsilon_parameter,
        code.energy_unit,
        default=DEFAULT_EPSILON_GAMMA,
    )
    alpha_B = getattr(par, 'hydrogen_alpha_B', None)
    if alpha_B is not None:
        alpha_B = to_unit_value(alpha_B, code.volume_unit / code.time_unit)
    beta = getattr(par, 'hydrogen_beta', None)
    if beta is not None:
        beta = to_unit_value(beta, code.volume_unit / code.time_unit)
    return {
        'interior': interior,
        'boundary_cm': boundary * scaling['scale_factor'],
        'width_cm': np.diff(boundary) * scaling['scale_factor'],
        'volume_cm3': as_named_array(
            to_unit_value(mesh.vol[interior], code.volume_unit)
            * scaling['density_factor']
        ),
        'radius_cm': as_named_array(
            to_unit_value(mesh.coordinate[interior], code.length_unit)
            * scaling['scale_factor']
        ),
        'radius_kpc': np.asarray(
            to_unit_value(mesh.coordinate[interior], code.length_unit)
            * scaling['scale_factor'] / kpc_in_cm,
            dtype=float,
        ),
        'xHI': xHI,
        'temperature_K': temperature_physical,
        'specific_energy_erg_g': specific_energy,
        'rho_g_cm3': rho_physical,
        'nH_cm3': rho_physical * getattr(par, 'hydrogen_mass_fraction', 1.0) / PROTON_MASS_CGS,
        'gamma': gamma,
        'hydrogen_mass_fraction': getattr(par, 'hydrogen_mass_fraction', 1.0),
        'sigma_gamma_cm2': sigma_gamma,
        'source_rate_s': source_rate,
        'epsilon_gamma_erg': epsilon_gamma,
        'source_temperature_factor': scaling['temperature_factor'],
        'source_scale_factor': scaling['scale_factor'],
        'source_CFL': getattr(par, 'hydrogen_source_CFL', 0.1),
        'dtmin_s': _optional_numeric_value(
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
        'compton_cmb_enabled': getattr(par, 'compton_cmb_enabled', False),
        'compton_cmb_redshift': getattr(par, 'compton_cmb_redshift', 0.0),
        'cmb_temperature_0_K': _optional_numeric_value(
            getattr(par, 'cmb_temperature_0', None),
            code.temperature_unit,
            default=2.7255 * unyt.K,
        ),
        'alpha_B_cm3_s': alpha_B,
        'beta_cm3_s': beta,
    }


def trace_spherical_tau(mesh, rho, xHI, hydrogen_mass_fraction, sigma_gamma):
    """Return the hydrogen optical depth per cell.

    This helper now requires ``mesh.code_units`` so the cgs conversion is
    explicit at the mesh boundary.
    """
    code = mesh.code_units
    rho_g_cm3 = to_unit_value(rho, code.density_unit)
    sigma_cm2 = to_unit_value(rh.photon_cross_section(sigma_gamma), code.area_unit)
    width_cm = to_unit_value(
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
        atomic_cooling=state.get('atomic_cooling', True),
        ngamma_cm3=ngamma,
        sigma_gamma_cm2=sigma,
        epsilon_gamma_erg=epsilon_gamma,
        compton_cmb_enabled=state['compton_cmb_enabled'],
        compton_cmb_redshift=state['compton_cmb_redshift'],
        cmb_temperature_0_K=state['cmb_temperature_0_K'],
    )


def get_timestep(state, ngamma, remaining_s, dtmax_s, verbose=False):
    """Return a source substep for a float thermo-chemistry state."""
    source_CFL = state['source_CFL']
    dtmin_s = state['dtmin_s']
    candidates = []
    debug_lines = []
    ionization_limiter_enabled = (
        state['recombination']
        or state['collisional_ionization']
        or ngamma is not None
    )
    if ionization_limiter_enabled:
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
        raise ValueError("hydrogen thermo-chemistry requires par.CodeUnits")
    if hasattr(fluid, 'ngamma') and 'ngamma_cm3' in state:
        target = from_unit_value(state['ngamma_cm3'], code.number_density_unit)
        if np.ndim(target) == 2:
            fluid.ngamma[:, interior] = target
        else:
            fluid.ngamma[interior] = target
    if hasattr(fluid, 'temp') and 'temperature_K' in state:
        fluid.temp[interior] = from_unit_value(
            state['temperature_K'] * state.get('source_temperature_factor', 1.0),
            code.temperature_unit,
        )
    if hasattr(fluid, 'xHI') and getattr(getattr(fluid, 'eos', None), 'gamma', None) is not None:
        fluid.SetHydrogenMu(
            hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0)
        )
        fluid.SetPressure()
    fluid.time = from_unit_value(state['time_s'], code.time_unit)


def get_thermochemistry_source_timestep_fast(mesh, fluid, par, remaining):
    """Return a source substep for RT-coupled heating/chemistry."""
    state = _fast_source_state(mesh, fluid, par)
    code = _code_units(par)
    if code is None:
        raise ValueError("hydrogen thermo-chemistry requires par.CodeUnits")
    remaining_s = (
        to_unit_value(remaining, code.time_unit)
        * state['source_scale_factor']**2
    )
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
    return from_unit_value(
        sub_dt_s / state['source_scale_factor']**2,
        code.time_unit,
    ), thermal_rate


def _fast_source_scaling(fluid, par, gamma):
    """Return conversions from supercomoving state values to physical values."""
    identity = {
        'scale_factor': 1.0,
        'density_factor': 1.0,
        'temperature_factor': 1.0,
        'velocity_factor': 1.0,
        'time_factor': 1.0,
    }
    if not getattr(par, 'supercomoving_coordinates', False):
        return identity
    cosmology = getattr(par, 'cosmology', None)
    if cosmology is None:
        raise ValueError("supercomoving thermo-chemistry requires par.cosmology")
    time = getattr(par, 'fluid_time', None)
    if time is None:
        time = getattr(fluid, 'time', getattr(par, 'time', 0.0))
    tau = float(np.asarray(time, dtype=float).flat[0])
    scale_factor = float(cosmology.scale_factor_from_supercomoving(tau))
    return {
        'scale_factor': scale_factor,
        'density_factor': scale_factor**3,
        'temperature_factor': scale_factor**(3.0 * (gamma - 1.0)),
        'velocity_factor': scale_factor,
        'time_factor': scale_factor**2,
    }


def _fast_source_state(mesh, fluid, par):
    """Return a cgs float snapshot for the fast thermo-chemistry path."""
    code = _code_units(par)
    if code is None:
        raise ValueError("hydrogen thermo-chemistry requires par.CodeUnits")
    interior = slice(par.noghost, par.noghost + par.nogrid)
    gamma = getattr(getattr(fluid, 'eos', None), 'gamma', getattr(par, 'gamma', 5.0 / 3.0))
    scaling = _fast_source_scaling(fluid, par, gamma)
    rho_g_cm3 = (
        to_unit_value(fluid.rho[interior], code.density_unit)
        / scaling['density_factor']
    )
    temperature_K = (
        to_unit_value(fluid.temp[interior], code.temperature_unit)
        / scaling['temperature_factor']
    )
    velocity_supercomoving_cm_s = to_unit_value(
        fluid.vel[interior], code.velocity_unit
    )
    vel_cm_s = velocity_supercomoving_cm_s / scaling['velocity_factor']
    mass_g = to_unit_value(fluid.Mass[interior], code.mass_unit)
    energy_supercomoving_erg = to_unit_value(
        fluid.Energy[interior], code.energy_unit
    )
    state = {
        'interior': interior,
        'boundary_cm': as_named_array(
            to_unit_value(mesh.boundary[interior.start : interior.stop + 1], code.length_unit)
            * scaling['scale_factor']
        ),
        'width_cm': as_named_array(
            to_unit_value(mesh.xdelta[interior], code.length_unit)
            * scaling['scale_factor']
        ),
        'volume_cm3': as_named_array(
            to_unit_value(mesh.vol[interior], code.volume_unit)
            * scaling['density_factor']
        ),
        'rho_g_cm3': rho_g_cm3,
        'temperature_K': temperature_K,
        'xHI': as_named_array(
            fluid.xHI[interior] if hasattr(fluid, 'xHI') else np.ones(par.nogrid),
        ),
        'nH_cm3': rho_g_cm3 * getattr(par, 'hydrogen_mass_fraction', 1.0) / PROTON_MASS_CGS,
        'gamma': gamma,
        'source_scale_factor': scaling['scale_factor'],
        'source_temperature_factor': scaling['temperature_factor'],
        'source_density_factor': scaling['density_factor'],
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
        'sigma_gamma_cm2': _optional_numeric_value(
            getattr(par, 'hydrogen_sigma_gamma', None),
            code.area_unit,
            default=DEFAULT_SIGMA_GAMMA,
        ),
        'ngamma_cm3': (
            to_unit_value(
                fluid.ngamma[:, interior]
                if np.ndim(fluid.ngamma) == 2
                else fluid.ngamma[interior],
                code.number_density_unit,
            )
            / scaling['density_factor']
            if (
                getattr(par, 'hydrogen_radiation_field', False)
                or getattr(par, 'radiative_transfer', False)
            )
            and hasattr(fluid, 'ngamma')
            else None
        ),
        'source_rate_s': _optional_numeric_value(
            getattr(par, 'radiative_transfer_source_photon_rate', None),
            1.0 / code.time_unit,
            default=0.0,
        ),
        'epsilon_gamma_erg': _optional_numeric_value(
            getattr(par, 'hydrogen_epsilon_gamma', None),
            code.energy_unit,
            default=DEFAULT_EPSILON_GAMMA,
        ),
        'source_CFL': getattr(par, 'hydrogen_source_CFL', 0.1),
        'dtmin_s': _optional_numeric_value(
            getattr(par, 'hydrogen_source_dtmin', None),
            code.time_unit,
            default=0.0,
        ),
        'recombination': getattr(par, 'hydrogen_recombination', True),
        'collisional_ionization': getattr(par, 'hydrogen_collisional_ionization', True),
        'atomic_cooling': getattr(par, 'hydrogen_atomic_cooling', True),
        'thermal_coupling': getattr(par, 'hydrogen_thermal_coupling', True),
        'hydrogen_update_mu': getattr(par, 'hydrogen_update_mu', False),
        'compton_cmb_enabled': getattr(par, 'compton_cmb_enabled', False),
        'compton_cmb_redshift': getattr(par, 'compton_cmb_redshift', 0.0),
        'cmb_temperature_0_K': _optional_numeric_value(
            getattr(par, 'cmb_temperature_0', None),
            code.temperature_unit,
            default=2.7255 * unyt.K,
        ),
        'alpha_B_cm3_s': _optional_numeric_value(
            getattr(par, 'hydrogen_alpha_B', None),
            code.volume_unit / code.time_unit,
            default=None,
        ),
        'beta_cm3_s': _optional_numeric_value(
            getattr(par, 'hydrogen_beta', None),
            code.volume_unit / code.time_unit,
            default=None,
        ),
    }
    if state['thermal_coupling']:
        state['vel_cm_s'] = vel_cm_s
        specific_total_supercomoving = energy_supercomoving_erg / mass_g
        specific_kinetic_supercomoving = (
            0.5 * velocity_supercomoving_cm_s**2
        )
        specific_internal_physical = np.maximum(
            specific_total_supercomoving - specific_kinetic_supercomoving,
            0.0,
        ) / scaling['temperature_factor']
        state['specific_kinetic_energy_supercomoving_erg_g'] = (
            specific_kinetic_supercomoving
        )
        state['specific_kinetic_energy_erg_g'] = 0.5 * state['vel_cm_s']**2
        state['specific_total_energy_erg_g'] = (
            specific_internal_physical
            + state['specific_kinetic_energy_erg_g']
        )
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


def _apply_compton_only_source(state, dt_s):
    """Advance the fixed-composition Compton relaxation exactly."""
    temperature = np.asarray(state['temperature_K'], dtype=float)
    xHI = np.clip(np.asarray(state['xHI'], dtype=float), 0.0, 1.0)
    nH = _cgs_hydrogen_number_density(
        state['rho_g_cm3'], state['hydrogen_mass_fraction']
    )
    electron_density = nH * (1.0 - xHI)
    specific_heat = (
        BOLTZMANN_CONSTANT_CGS
        / ((state['gamma'] - 1.0) * state['mu'] * PROTON_MASS_CGS)
    )
    cmb_temperature = (
        state['cmb_temperature_0_K']
        * (1.0 + float(state['compton_cmb_redshift']))
    )
    zero_temperature_rate = cmb_compton_rate(
        np.zeros_like(temperature),
        electron_density,
        enabled=True,
        redshift=state['compton_cmb_redshift'],
        cmb_temperature_0_K=state['cmb_temperature_0_K'],
    )
    coupling_rate = np.divide(
        zero_temperature_rate,
        state['rho_g_cm3'] * specific_heat * cmb_temperature,
        out=np.zeros_like(temperature),
        where=(state['rho_g_cm3'] > 0.0) & (specific_heat > 0.0),
    )
    temperature = cmb_temperature + (
        temperature - cmb_temperature
    ) * np.exp(-coupling_rate * dt_s)
    state['specific_total_energy_erg_g'] = (
        specific_heat * temperature
        + state['specific_kinetic_energy_erg_g']
    )
    state['specific_energy_erg_g'] = specific_heat * temperature
    _fast_update_temperature_from_energy(state)


def _coupled_implicit_source_update(
    state,
    dt_s,
    ngamma=None,
    tolerance=1.0e-6,
    max_iterations=32,
):
    """Advance hydrogen energy and ``xHI`` with a coupled backward-Euler solve.

    The solve is performed in ``(log e, logit xHI)`` coordinates.  This keeps
    the internal energy positive and the neutral fraction inside its physical
    interval while allowing the temperature-dependent rates to be evaluated
    at the same new-time state.  The function is deliberately limited to the
    local source problem: the photon field is held fixed during this update.

    Returns ``True`` only when every cell converges.  The caller can then use
    the existing source subcycler as a safe fallback for a failed solve.
    """
    if not state.get('thermal_coupling', False):
        return False

    rho = np.asarray(state['rho_g_cm3'], dtype=float)
    kinetic = np.asarray(state['specific_kinetic_energy_erg_g'], dtype=float)
    energy_old = np.asarray(state['specific_energy_erg_g'], dtype=float).copy()
    x_old = np.asarray(state['xHI'], dtype=float).copy()
    if not (
        np.all(np.isfinite(rho))
        and np.all(rho > 0.0)
        and np.all(np.isfinite(energy_old))
        and np.all(np.isfinite(x_old))
    ):
        return False

    # The floor is only a coordinate safeguard.  It is many orders of
    # magnitude below normal gas internal energies and is never written back
    # unless the input itself is below it.
    energy_floor = 1.0e-30
    x_floor = 1.0e-12
    energy_old = np.maximum(energy_old, energy_floor)
    x_old = np.clip(x_old, x_floor, 1.0 - x_floor)
    energy_scale = np.maximum(energy_old, energy_floor)
    dt_value = float(np.asarray(dt_s, dtype=float))
    if not np.isfinite(dt_value) or dt_value < 0.0:
        return False
    if dt_value == 0.0:
        state['specific_energy_erg_g'] = energy_old
        state['specific_total_energy_erg_g'] = energy_old + kinetic
        state['xHI'] = x_old
        _fast_update_temperature_from_energy(state)
        return True

    def _logit(value):
        value = np.clip(value, x_floor, 1.0 - x_floor)
        return np.log(value / (1.0 - value))

    def _sigmoid(value):
        value = np.clip(value, -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(-value))

    def _residual(log_energy, logit_x):
        energy = np.exp(np.clip(log_energy, np.log(energy_floor), 700.0))
        xhi = _sigmoid(logit_x)
        trial = dict(state)
        trial['specific_energy_erg_g'] = energy
        trial['specific_total_energy_erg_g'] = energy + kinetic
        trial['xHI'] = xhi
        if trial.get('hydrogen_update_mu', False):
            trial['mu'] = rh.mean_molecular_weight_mu(
                xhi,
                hydrogen_mass_fraction=trial['hydrogen_mass_fraction'],
            )
        _fast_update_temperature_from_energy(trial)
        thermal = thermal_rate(trial, ngamma)
        chemistry = ionization_fraction_rate(trial, ngamma)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            energy_residual = (
                energy - energy_old - dt_value * thermal / rho
            ) / energy_scale
        chemistry_residual = xhi - x_old - dt_value * chemistry
        return energy_residual, chemistry_residual, energy, xhi, trial

    log_energy = np.log(energy_old)
    logit_x = _logit(x_old)
    residual_energy, residual_x, _, _, _ = _residual(log_energy, logit_x)
    converged = np.zeros_like(energy_old, dtype=bool)
    finite = np.isfinite(residual_energy) & np.isfinite(residual_x)
    converged[finite] = np.maximum(
        np.abs(residual_energy[finite]), np.abs(residual_x[finite])
    ) <= tolerance

    finite_difference_step = 1.0e-5
    for _ in range(int(max_iterations)):
        active = ~converged
        if not np.any(active):
            break

        energy_plus, x_plus = _residual(
            log_energy + finite_difference_step,
            logit_x,
        )[:2]
        energy_x_plus, x_x_plus = _residual(
            log_energy,
            logit_x + finite_difference_step,
        )[:2]
        jacobian_11 = (energy_plus - residual_energy) / finite_difference_step
        jacobian_21 = (x_plus - residual_x) / finite_difference_step
        jacobian_12 = (energy_x_plus - residual_energy) / finite_difference_step
        jacobian_22 = (x_x_plus - residual_x) / finite_difference_step
        determinant = jacobian_11 * jacobian_22 - jacobian_12 * jacobian_21
        good = (
            active
            & np.isfinite(determinant)
            & (np.abs(determinant) > 1.0e-30)
            & np.isfinite(jacobian_11)
            & np.isfinite(jacobian_12)
            & np.isfinite(jacobian_21)
            & np.isfinite(jacobian_22)
        )
        # If thermal sources are disabled, the energy residual is an exactly
        # decoupled zero equation and the full 2x2 Jacobian is singular.  The
        # chemistry equation is still a valid scalar Newton problem.  Apply
        # that reduction (and the analogous energy reduction) rather than
        # incorrectly treating a solvable source state as a failed solve.
        scalar_chemistry = (
            active
            & ~good
            & (np.abs(residual_energy) <= tolerance)
            & np.isfinite(jacobian_22)
            & (np.abs(jacobian_22) > 1.0e-30)
        )
        scalar_energy = (
            active
            & ~good
            & ~scalar_chemistry
            & (np.abs(residual_x) <= tolerance)
            & np.isfinite(jacobian_11)
            & (np.abs(jacobian_11) > 1.0e-30)
        )
        solvable = good | scalar_chemistry | scalar_energy
        if not np.any(solvable):
            break

        delta_energy = np.zeros_like(residual_energy)
        delta_x = np.zeros_like(residual_x)
        delta_energy[good] = (
            -residual_energy[good] * jacobian_22[good]
            + jacobian_12[good] * residual_x[good]
        ) / determinant[good]
        delta_x[good] = (
            jacobian_21[good] * residual_energy[good]
            - jacobian_11[good] * residual_x[good]
        ) / determinant[good]
        delta_x[scalar_chemistry] = (
            -residual_x[scalar_chemistry] / jacobian_22[scalar_chemistry]
        )
        delta_energy[scalar_energy] = (
            -residual_energy[scalar_energy] / jacobian_11[scalar_energy]
        )
        finite_delta = (
            solvable & np.isfinite(delta_energy) & np.isfinite(delta_x)
        )
        accepted = np.zeros_like(active, dtype=bool)
        current_norm = np.maximum(np.abs(residual_energy), np.abs(residual_x))
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
            trial_log_energy = log_energy + damping * delta_energy
            trial_logit_x = logit_x + damping * delta_x
            trial_energy_residual, trial_x_residual = _residual(
                trial_log_energy,
                trial_logit_x,
            )[:2]
            trial_norm = np.maximum(
                np.abs(trial_energy_residual), np.abs(trial_x_residual)
            )
            improve = (
                finite_delta
                & ~accepted
                & np.isfinite(trial_norm)
                & (trial_norm <= current_norm)
            )
            if np.any(improve):
                log_energy[improve] = trial_log_energy[improve]
                logit_x[improve] = trial_logit_x[improve]
                residual_energy[improve] = trial_energy_residual[improve]
                residual_x[improve] = trial_x_residual[improve]
                accepted[improve] = True

        converged |= accepted & (
            np.maximum(np.abs(residual_energy), np.abs(residual_x)) <= tolerance
        )
        # A cell whose Newton step cannot reduce the residual is left for the
        # explicit fallback rather than allowing a source update to diverge.
        if not np.any(accepted & active):
            break

    if not np.all(converged):
        return False

    _, _, energy, xhi, trial = _residual(log_energy, logit_x)
    state['specific_energy_erg_g'] = energy
    state['specific_total_energy_erg_g'] = energy + kinetic
    state['xHI'] = np.clip(xhi, x_floor, 1.0 - x_floor)
    if state.get('hydrogen_update_mu', False):
        state['mu'] = trial['mu']
    _fast_update_temperature_from_energy(state)
    return True


def _copy_fast_source_state(state):
    """Copy a numeric source state for a trial implicit update."""
    return copy.deepcopy(state)


def _implicit_state_difference(coarse, fine):
    """Return the normalized difference between two implicit source states."""
    differences = []
    for key, floor in (
        ('specific_energy_erg_g', 1.0e-30),
        ('temperature_K', 1.0),
        ('xHI', 1.0e-8),
    ):
        coarse_value = np.asarray(coarse[key], dtype=float)
        fine_value = np.asarray(fine[key], dtype=float)
        scale = np.maximum(np.abs(fine_value), floor)
        with np.errstate(divide='ignore', invalid='ignore'):
            difference = np.abs(coarse_value - fine_value) / scale
        differences.append(np.max(difference))
    result = max(differences)
    return float(result) if np.isfinite(result) else np.inf


def _set_fast_source_state(target, source):
    """Replace a source state with a converged trial state."""
    target.clear()
    target.update(copy.deepcopy(source))


def _adaptive_coupled_implicit_source_update(
    state,
    dt_s,
    ngamma=None,
    tolerance=1.0e-6,
    convergence_tolerance=None,
    max_iterations=32,
    max_refinements=4,
):
    """Advance coupled sources with factor-of-two timestep convergence.

    Each accepted interval compares one backward-Euler step with two
    half-sized backward-Euler steps.  If the difference is too large, the
    interval is halved and the comparison is repeated.  The returned count
    is the number of fine implicit substeps actually applied.
    """
    remaining_s = float(np.asarray(dt_s, dtype=float))
    if not np.isfinite(remaining_s) or remaining_s < 0.0:
        return False, 0
    if convergence_tolerance is None:
        convergence_tolerance = tolerance
    if remaining_s == 0.0:
        return True, 0

    total_source_steps = 0
    trial_dt_s = remaining_s
    zero_time_s = 0.0
    while remaining_s > zero_time_s:
        accepted = False
        candidate_dt_s = min(trial_dt_s, remaining_s)
        for refinement in range(int(max_refinements) + 1):
            coarse = _copy_fast_source_state(state)
            fine = _copy_fast_source_state(state)
            coarse_ok = _coupled_implicit_source_update(
                coarse,
                candidate_dt_s,
                ngamma=ngamma,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            half_dt_s = 0.5 * candidate_dt_s
            fine_ok = _coupled_implicit_source_update(
                fine,
                half_dt_s,
                ngamma=ngamma,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            if fine_ok:
                fine_ok = _coupled_implicit_source_update(
                    fine,
                    half_dt_s,
                    ngamma=ngamma,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            difference = (
                _implicit_state_difference(coarse, fine)
                if coarse_ok and fine_ok
                else np.inf
            )
            if (
                coarse_ok
                and fine_ok
                and difference <= convergence_tolerance
            ):
                _set_fast_source_state(state, fine)
                remaining_s -= candidate_dt_s
                total_source_steps += 2
                trial_dt_s = candidate_dt_s
                accepted = True
                break
            candidate_dt_s *= 0.5

        if not accepted:
            return False, total_source_steps
    return True, total_source_steps


def _fast_sync_state_to_fluid(state, fluid, par):
    """Copy a float thermo-chemistry state back to the fluid container."""
    interior = state['interior']
    fluid.xHI[interior] = state['xHI']
    if hasattr(fluid, 'ngamma') and state.get('ngamma_cm3') is not None:
        code = _code_units(par)
        target = from_unit_value(state['ngamma_cm3'], code.number_density_unit)
        if np.ndim(target) == 2:
            fluid.ngamma[:, interior] = target
        else:
            fluid.ngamma[interior] = target
    if hasattr(fluid, 'mu'):
        fluid.mu[interior] = state['mu']
    code = _code_units(par)
    fluid.temp[interior] = from_unit_value(
        state['temperature_K'] * state.get('source_temperature_factor', 1.0),
        code.temperature_unit,
    )
    if state.get('thermal_coupling', False):
        specific_internal_energy_physical = (
            state['specific_total_energy_erg_g']
            - state['specific_kinetic_energy_erg_g']
        )
        specific_internal_energy = (
            specific_internal_energy_physical
            * state.get('source_temperature_factor', 1.0)
        )
        specific_total_energy = (
            specific_internal_energy
            + state.get('specific_kinetic_energy_supercomoving_erg_g', 0.0)
        )
        fluid.pre[interior] = (
            specific_internal_energy
            * np.asarray(fluid.rho[interior], dtype=float)
            * (fluid.eos.gamma - 1.0)
        )
        fluid.Energy[interior] = (
            specific_total_energy
            * np.asarray(fluid.Mass[interior], dtype=float)
        )
    if state.get('hydrogen_update_mu', False) and hasattr(fluid, 'xHI') and getattr(getattr(fluid, 'eos', None), 'gamma', None) is not None:
        fluid.SetHydrogenMu(
            hydrogen_mass_fraction=state['hydrogen_mass_fraction']
        )
        if state.get('thermal_coupling', False):
            fluid.SetPressure()


def c2ray_source_state(mesh, fluid, par):
    """Return the numeric source state used by the C²-Ray integrator."""
    return _fast_source_state(mesh, fluid, par)


def sync_c2ray_state(state, fluid, par):
    """Copy a C²-Ray numeric source state back to the runtime fluid."""
    return _fast_sync_state_to_fluid(state, fluid, par)


def apply_thermochemistry_fast(dt, mesh, fluid, par, transport_result=None):
    """Fast source update for RT-coupled thermo-chemistry tests."""
    if not thermochemistry_enabled(fluid, par):
        return 0

    state = _fast_source_state(mesh, fluid, par)
    code = _code_units(par)
    if code is None:
        raise ValueError("hydrogen thermo-chemistry requires par.CodeUnits")
    remaining_s = (
        to_unit_value(dt, code.time_unit)
        * state['source_scale_factor']**2
    )
    total_dt_s = remaining_s
    zero_time_s = 0.0
    source_steps = 0
    absorbed_integral = None
    source_solver = str(getattr(par, 'hydrogen_source_solver', 'explicit')).lower()
    if source_solver not in ('explicit', 'coupled_implicit'):
        raise ValueError(
            "hydrogen_source_solver must be 'explicit' or 'coupled_implicit'"
        )
    compton_only = (
        source_solver == 'explicit'
        and state['thermal_coupling']
        and state['compton_cmb_enabled']
        and not state['recombination']
        and not state['collisional_ionization']
        and not state.get('atomic_cooling', True)
        and not getattr(par, 'radiative_transfer', False)
        and not getattr(par, 'hydrogen_radiation_field', False)
        and state.get('ngamma_cm3') is None
    )
    if compton_only:
        _apply_compton_only_source(state, remaining_s)
        _fast_sync_state_to_fluid(state, fluid, par)
        return {
            'source_steps': 1,
            'absorbed_photon_rate': None,
            'photon_energy_erg': np.atleast_1d(
                _optional_numeric_value(
                    getattr(par, 'ionizing_photon_energy_erg',
                            getattr(par, 'hydrogen_photon_energy', 0.0)),
                    code.energy_unit,
                    default=0.0,
                )
            ),
            'direction': int(getattr(par, 'radiative_transfer_direction', 1)),
        }
    if source_solver == 'coupled_implicit' and remaining_s > zero_time_s:
        # A ray-traced photon field can change during the source step.  Keep
        # that operator split on the established path; the coupled solver is
        # for a local, fixed photon field (including no photon field).
        can_solve_coupled = not getattr(par, 'radiative_transfer', False)
        solved = False
        if can_solve_coupled:
            solved, implicit_source_steps = _adaptive_coupled_implicit_source_update(
                state,
                remaining_s,
                ngamma=state.get('ngamma_cm3'),
                tolerance=float(
                    getattr(par, 'hydrogen_implicit_tolerance', 1.0e-6)
                ),
                    max_iterations=int(
                        getattr(par, 'hydrogen_implicit_max_iterations', 32)
                    ),
                    convergence_tolerance=float(
                        getattr(
                            par,
                            'hydrogen_implicit_convergence_tolerance',
                            1.0e-3,
                        )
                    ),
                    max_refinements=int(
                    getattr(par, 'hydrogen_implicit_max_refinements', 4)
                ),
            )
        if solved:
            _fast_sync_state_to_fluid(state, fluid, par)
            return {
                'source_steps': implicit_source_steps,
                'absorbed_photon_rate': None,
                'photon_energy_erg': np.atleast_1d(
                    _optional_numeric_value(
                        getattr(par, 'ionizing_photon_energy_erg',
                                getattr(par, 'hydrogen_photon_energy', 0.0)),
                        code.energy_unit,
                        default=0.0,
                    )
                ),
                'direction': int(getattr(par, 'radiative_transfer_direction', 1)),
            }
        fallback = str(
            getattr(par, 'hydrogen_implicit_fallback', 'explicit')
        ).lower()
        if fallback not in ('explicit', 'error'):
            raise ValueError(
                "hydrogen_implicit_fallback must be 'explicit' or 'error'"
            )
        if fallback == 'error':
            raise RuntimeError(
                'coupled implicit hydrogen source solve did not converge'
            )
    while remaining_s > zero_time_s:
        absorbed = None
        # first update the photon density if RT is enabled and we are on the right step
        if getattr(par, 'radiative_transfer', False):
            if transport_result is not None:
                transport = transport_result
                transport_result = None
            else:
                boundary_flux = getattr(
                    par,
                    'radiative_transfer_boundary_flux_groups',
                    getattr(par, 'radiative_transfer_boundary_flux', 0.0),
                )
                if hasattr(boundary_flux, 'to_value'):
                    boundary_flux = boundary_flux.to_value(
                        1.0 / (unyt.cm**2 * unyt.s)
                    )
                else:
                    boundary_flux = np.asarray(boundary_flux, dtype=float) * float(
                        (1.0 / (code.length_unit**2 * code.time_unit)).to_value(
                            1.0 / (unyt.cm**2 * unyt.s)
                        )
                    )
                transport = rrt.trace_long_characteristics(
                    SimpleNamespace(
                        boundary=state['boundary_cm'],
                        vol=state['volume_cm3'],
                        coordsys=getattr(par, 'coordsys', 'cartesian'),
                    ),
                    rho=state['rho_g_cm3'],
                    xHI=state['xHI'],
                    hydrogen_mass_fraction=state['hydrogen_mass_fraction'],
                    sigma_gamma=np.asarray(state['sigma_gamma_cm2'], dtype=float)
                    * float((1.0 * code.area_unit).to_value(unyt.cm**2)),
                    boundary_flux=boundary_flux,
                    source_photon_rate=np.asarray(state['source_rate_s'], dtype=float)
                    * float((1.0 / code.time_unit).to_value(1.0 / unyt.s)),
                    direction=getattr(par, 'radiative_transfer_direction', 1),
                    group_edges_eV=getattr(par, 'radiation_group_edges_eV', None),
                )
            state['ngamma_cm3'] = transport.cell_photon_density
            absorbed = np.asarray(transport.absorbed_photon_rate, dtype=float)
            if absorbed.ndim == 1:
                absorbed = absorbed[None, :]
            if absorbed_integral is None:
                absorbed_integral = np.zeros_like(absorbed)

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
        # With no recombination, collisional ionization, or radiation field,
        # the ionization fraction is an intentionally fixed residual input.
        # Avoid the expensive implicit solve in the Compton-only thermal
        # network; Compton scattering changes energy, not ionization.
        if (
            state['recombination']
            or state['collisional_ionization']
            or getattr(par, 'radiative_transfer', False)
            or getattr(par, 'hydrogen_radiation_field', False)
        ):
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
        if absorbed is not None:
            absorbed_integral += absorbed * sub_dt_s
        remaining_s -= sub_dt_s
        source_steps += 1

    _fast_sync_state_to_fluid(state, fluid, par)
    absorbed_rate = None
    if absorbed_integral is not None:
        if total_dt_s == 0.0:
            absorbed_rate = np.zeros_like(absorbed_integral)
        else:
            absorbed_rate = absorbed_integral / total_dt_s
    energy = getattr(par, 'ionizing_photon_energy_erg', None)
    if energy is None:
        energy = getattr(par, 'hydrogen_photon_energy', 0.0)
    if hasattr(energy, 'to_value'):
        energy = np.asarray(energy.to_value('erg'), dtype=float)
    return {
        'source_steps': source_steps,
        'absorbed_photon_rate': absorbed_rate,
        'photon_energy_erg': np.atleast_1d(energy),
        'direction': int(getattr(par, 'radiative_transfer_direction', 1)),
    }


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

    def apply_fast(self, dt, mesh, fluid, par, transport_result=None):
        return apply_thermochemistry_fast(
            dt,
            mesh,
            fluid,
            par,
            transport_result=transport_result,
        )
