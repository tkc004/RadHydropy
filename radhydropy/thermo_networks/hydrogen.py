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
    DEFAULT_EPSILON_GAMMA_CGS_ERG,
    DEFAULT_SIGMA_GAMMA_CGS_CM2,
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
from radhydropy.diagnostics import (
    check_source_temperature,
    thermochemistry_active_mask,
)
from radhydropy.state_boundaries import (
    CgsSourceState,
    CodeFluidState,
    cgs_source_state_from_code,
)




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


def _parameter_value(par, name, default=None):
    """Read a parameter from the flat store or nested parameter group."""
    value = getattr(par, name, None)
    if value is not None:
        return value
    parameter = getattr(par, '_parameter', None)
    return parameter(name, default) if parameter is not None else default


def _cgs_alpha_B(temperature_cgs_K):
    temperature_cgs_K = np.asarray(temperature_cgs_K, dtype=float)
    result = np.zeros_like(temperature_cgs_K, dtype=float)
    valid = temperature_cgs_K > 0.0
    if np.any(valid):
        lam = 315614.0 / temperature_cgs_K[valid]
        result[valid] = (
            2.753e-14
            * lam**1.5
            * (1.0 + (lam / 2.740) ** 0.407) ** -2.242
        )
    return result


def _cgs_alpha_A(temperature_cgs_K):
    """H II case-A recombination coefficient (Hui & Gnedin 1997)."""
    temperature_cgs_K = np.asarray(temperature_cgs_K, dtype=float)
    result = np.zeros_like(temperature_cgs_K, dtype=float)
    valid = temperature_cgs_K > 0.0
    if np.any(valid):
        lam = 315614.0 / temperature_cgs_K[valid]
        result[valid] = (
            1.269e-13
            * lam**1.503
            * (1.0 + (lam / 0.522) ** 0.470) ** -1.923
        )
    return result


def _cgs_beta(temperature_cgs_K):
    temperature_cgs_K = np.asarray(temperature_cgs_K, dtype=float)
    result = np.zeros_like(temperature_cgs_K, dtype=float)
    valid = temperature_cgs_K > 0.0
    if np.any(valid):
        temp = temperature_cgs_K[valid]
        temp5 = temp / 1.0e5
        result[valid] = (
            1.17e-10
            * temp**0.5
            * np.exp(-157809.1 / temp)
            / (1.0 + temp5**0.5)
        )
    return result


def collisional_equilibrium_neutral_fraction(temperature_cgs_K):
    """Return the H I fraction in collisional ionization equilibrium."""
    alpha = _cgs_alpha_B(temperature_cgs_K)
    beta = _cgs_beta(temperature_cgs_K)
    total = alpha + beta
    return np.divide(alpha, total, out=np.ones_like(alpha), where=total > 0.0)


def _cgs_gamma_line_eHI(temperature_cgs_K):
    temperature_cgs_K = np.asarray(temperature_cgs_K, dtype=float)
    result = np.zeros_like(temperature_cgs_K, dtype=float)
    valid = temperature_cgs_K > 0.0
    if np.any(valid):
        temp = temperature_cgs_K[valid]
        temp5 = temp / 1.0e5
        result[valid] = (
            7.5e-19
            * np.exp(-118348.0 / temp)
            / (1.0 + temp5**0.5)
        )
    return result


def _cgs_gamma_ion_eHI(temperature_cgs_K):
    temperature_cgs_K = np.asarray(temperature_cgs_K, dtype=float)
    result = np.zeros_like(temperature_cgs_K, dtype=float)
    valid = temperature_cgs_K > 0.0
    if np.any(valid):
        temp = temperature_cgs_K[valid]
        temp5 = temp / 1.0e5
        result[valid] = (
            2.54e-21
            * temp**0.5
            * np.exp(-157809.1 / temp)
            / (1.0 + temp5**0.5)
        )
    return result


def _cgs_gamma_ff_eHII(temperature_cgs_K):
    temperature_cgs_K = np.asarray(temperature_cgs_K, dtype=float)
    result = np.zeros_like(temperature_cgs_K, dtype=float)
    valid = temperature_cgs_K > 0.0
    if np.any(valid):
        temp = temperature_cgs_K[valid]
        result[valid] = (
            1.42e-27
            * temp**0.5
            * (1.1 + 0.34 * np.exp(-(5.5 - np.log10(temp)) ** 2 / 3.0))
        )
    return result


def _cgs_gamma_B_eHII(temperature_cgs_K):
    temperature_cgs_K = np.asarray(temperature_cgs_K, dtype=float)
    result = np.zeros_like(temperature_cgs_K, dtype=float)
    valid = temperature_cgs_K > 0.0
    if np.any(valid):
        temp = temperature_cgs_K[valid]
        lam = 315614.0 / temp
        result[valid] = (
            3.435e-30
            * temp
            * lam**1.970
            * (1.0 + (lam / 2.250) ** 0.376) ** -3.720
        )
    return result


def _cgs_gamma_A_eHII(temperature_cgs_K):
    """H II case-A recombination cooling coefficient."""
    temperature_cgs_K = np.asarray(temperature_cgs_K, dtype=float)
    result = np.zeros_like(temperature_cgs_K, dtype=float)
    valid = temperature_cgs_K > 0.0
    if np.any(valid):
        temp = temperature_cgs_K[valid]
        lam = 315614.0 / temp
        result[valid] = (
            1.778e-29
            * temp
            * lam**1.965
            * (1.0 + (lam / 0.541) ** 0.502) ** -2.697
        )
    return result


def _cgs_hydrogen_number_density(rho_cgs_g_cm3, hydrogen_mass_fraction=1.0):
    return hydrogen_mass_fraction * np.asarray(rho_cgs_g_cm3, dtype=float) / PROTON_MASS_CGS


def _cgs_photoionization_frequency(ngamma_cgs_cm3, sigma_gamma_cgs_cm2):
    ngamma_cgs_cm3 = np.asarray(ngamma_cgs_cm3, dtype=float)
    sigma_gamma_cgs_cm2 = np.asarray(sigma_gamma_cgs_cm2, dtype=float)
    if ngamma_cgs_cm3.ndim > 1 and sigma_gamma_cgs_cm2.ndim == 1:
        sigma_gamma_cgs_cm2 = sigma_gamma_cgs_cm2[:, None]
    rate_cgs_s = SPEED_OF_LIGHT_CGS * sigma_gamma_cgs_cm2 * ngamma_cgs_cm3
    return np.sum(rate_cgs_s, axis=0) if np.ndim(rate_cgs_s) > 1 else rate_cgs_s


def _cgs_source_thermal_rate(
    rho_cgs_g_cm3,
    temperature_cgs_K,
    xHI,
    hydrogen_mass_fraction=1.0,
    recombination=True,
    collisional_ionization=True,
    atomic_cooling=True,
    ngamma_cgs_cm3=None,
    sigma_gamma_cgs_cm2=1.0,
    epsilon_gamma_cgs_erg=0.0,
    compton_cmb_enabled=False,
    compton_cmb_redshift=0.0,
    cmb_temperature_0_cgs_K=2.7255,
):
    xHI = np.clip(np.asarray(xHI, dtype=float), 0.0, 1.0)
    ionized = 1.0 - xHI
    nH = _cgs_hydrogen_number_density(rho_cgs_g_cm3, hydrogen_mass_fraction)
    if atomic_cooling:
        eHI_cooling = _cgs_gamma_line_eHI(temperature_cgs_K)
        if collisional_ionization:
            eHI_cooling += _cgs_gamma_ion_eHI(temperature_cgs_K)
        eHII_cooling = _cgs_gamma_ff_eHII(temperature_cgs_K)
        if recombination:
            eHII_cooling += _cgs_gamma_B_eHII(temperature_cgs_K)
    else:
        eHI_cooling = np.zeros_like(temperature_cgs_K, dtype=float)
        eHII_cooling = np.zeros_like(temperature_cgs_K, dtype=float)
    cooling = nH**2 * (xHI * ionized * eHI_cooling + ionized**2 * eHII_cooling)
    if ngamma_cgs_cm3 is None:
        heating_cgs_erg_cm3_s = np.zeros_like(cooling, dtype=float)
    else:
        ngamma_cgs_cm3 = np.asarray(ngamma_cgs_cm3, dtype=float)
        sigma_gamma_cgs_cm2 = np.asarray(sigma_gamma_cgs_cm2, dtype=float)
        epsilon_gamma_cgs_erg = np.asarray(epsilon_gamma_cgs_erg, dtype=float)
        if ngamma_cgs_cm3.ndim > 1:
            if sigma_gamma_cgs_cm2.ndim == 1:
                sigma_gamma_cgs_cm2 = sigma_gamma_cgs_cm2[:, None]
            if epsilon_gamma_cgs_erg.ndim == 1:
                epsilon_gamma_cgs_erg = epsilon_gamma_cgs_erg[:, None]
        photoheating_cgs_erg_s_per_atom = (
            SPEED_OF_LIGHT_CGS
            * epsilon_gamma_cgs_erg
            * sigma_gamma_cgs_cm2
            * ngamma_cgs_cm3
        )
        if np.ndim(photoheating_cgs_erg_s_per_atom) > 1:
            photoheating_cgs_erg_s_per_atom = np.sum(photoheating_cgs_erg_s_per_atom, axis=0)
        heating_cgs_erg_cm3_s = nH * xHI * photoheating_cgs_erg_s_per_atom
    electron_density = nH * ionized
    return heating_cgs_erg_cm3_s - cooling + cmb_compton_rate(
        temperature_cgs_K,
        electron_density,
        enabled=compton_cmb_enabled,
        redshift=compton_cmb_redshift,
        cmb_temperature_0_cgs_K=cmb_temperature_0_cgs_K,
    )


def _cgs_static_neutral_fraction_rate(
    rho_cgs_g_cm3,
    temperature_cgs_K,
    xHI,
    hydrogen_mass_fraction=1.0,
    recombination=True,
    collisional_ionization=True,
    ngamma_cgs_cm3=None,
    sigma_gamma_cgs_cm2=1.0,
    recombination_coefficient_cgs_cm3_s=None,
    ionization_coefficient_cgs_cm3_s=None,
):
    xHI = np.clip(np.asarray(xHI, dtype=float), 0.0, 1.0)
    ionized = 1.0 - xHI
    nH = _cgs_hydrogen_number_density(rho_cgs_g_cm3, hydrogen_mass_fraction)
    if not recombination:
        recombination_coefficient_cgs_cm3_s = np.zeros_like(
            temperature_cgs_K,
            dtype=float,
        )
    elif recombination_coefficient_cgs_cm3_s is None:
        recombination_coefficient_cgs_cm3_s = _cgs_alpha_B(temperature_cgs_K)
    else:
        recombination_coefficient_cgs_cm3_s = np.asarray(
            recombination_coefficient_cgs_cm3_s,
            dtype=float,
        )
    if not collisional_ionization:
        ionization_coefficient_cgs_cm3_s = np.zeros_like(
            temperature_cgs_K,
            dtype=float,
        )
    elif ionization_coefficient_cgs_cm3_s is None:
        ionization_coefficient_cgs_cm3_s = _cgs_beta(temperature_cgs_K)
    else:
        ionization_coefficient_cgs_cm3_s = np.asarray(
            ionization_coefficient_cgs_cm3_s,
            dtype=float,
        )
    if ngamma_cgs_cm3 is None:
        photoionization_rate_cgs_s = np.zeros_like(xHI, dtype=float)
    else:
        photoionization_rate_cgs_s = _cgs_photoionization_frequency(
            ngamma_cgs_cm3,
            sigma_gamma_cgs_cm2,
        )
    return (
        ionized**2 * nH * recombination_coefficient_cgs_cm3_s
        - xHI * ionized * nH * ionization_coefficient_cgs_cm3_s
        - xHI * photoionization_rate_cgs_s
    )


def _cgs_static_neutral_fraction_implicit_update(
    rho_cgs_g_cm3,
    temperature_cgs_K,
    xHI,
    dt_s,
    hydrogen_mass_fraction=1.0,
    recombination=True,
    collisional_ionization=True,
    ngamma_cgs_cm3=None,
    sigma_gamma_cgs_cm2=1.0,
    recombination_coefficient_cgs_cm3_s=None,
    ionization_coefficient_cgs_cm3_s=None,
):
    xHI = np.clip(np.asarray(xHI, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    nH = _cgs_hydrogen_number_density(rho_cgs_g_cm3, hydrogen_mass_fraction)
    if recombination:
        if recombination_coefficient_cgs_cm3_s is None:
            recombination_coefficient_cgs_cm3_s = _cgs_alpha_B(temperature_cgs_K)
        else:
            recombination_coefficient_cgs_cm3_s = np.asarray(
                recombination_coefficient_cgs_cm3_s,
                dtype=float,
            )
        recombination_rate_s = nH * recombination_coefficient_cgs_cm3_s
    else:
        recombination_rate_s = np.zeros_like(xHI, dtype=float)
    if collisional_ionization:
        if ionization_coefficient_cgs_cm3_s is None:
            ionization_coefficient_cgs_cm3_s = _cgs_beta(temperature_cgs_K)
        else:
            ionization_coefficient_cgs_cm3_s = np.asarray(
                ionization_coefficient_cgs_cm3_s,
                dtype=float,
            )
        ionization_rate_s = nH * ionization_coefficient_cgs_cm3_s
    else:
        ionization_rate_s = np.zeros_like(recombination_rate_s, dtype=float)
    if ngamma_cgs_cm3 is None:
        photoionization_rate_cgs_s = np.zeros_like(recombination_rate_s, dtype=float)
    else:
        photoionization_rate_cgs_s = _cgs_photoionization_frequency(
            ngamma_cgs_cm3,
            sigma_gamma_cgs_cm2,
        )

    dt_value = float(np.asarray(dt_s, dtype=float))
    a = dt_value * (recombination_rate_s + ionization_rate_s)
    b = -(
        1.0
        + dt_value
        * (photoionization_rate_cgs_s + 2.0 * recombination_rate_s + ionization_rate_s)
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
    first = int(par.mesh.ghost_cells)
    grid_cells = int(par.mesh.grid_cells)
    return slice(first, first + grid_cells)


def thermochemistry_enabled(fluid, par):
    return getattr(par, 'hydrogen_chemistry', False) and hasattr(fluid, 'xHI')


def thermochemistry_radiation_enabled(fluid, par):
    return (
        thermochemistry_enabled(fluid, par)
        and (
            getattr(par, 'hydrogen_radiation_field', False)
            or getattr(par, 'radiative_transfer', False)
        )
        and hasattr(fluid, 'ngamma_code')
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
    xHI = ru.SafeDivide(neutral_mass, fluid.Mass_code)
    fluid.xHI = rh.clip_neutral_fraction(np.asarray(xHI, dtype=float))


def source_state(mesh, fluid, par):
    """Return the hydrogen source state through the typed cgs boundary."""
    code = _code_units(par)
    if code is None:
        raise ValueError("hydrogen thermo-chemistry requires configured code units")
    kpc_in_cm = float((1.0 * unyt.kpc).to_value(unyt.cm))
    interior = interior_slice(par)
    runtime = fluid.code_state
    xHI = as_named_array(runtime.xHI_dimensionless[interior].copy())
    gamma = getattr(
        getattr(fluid, 'eos', None),
        'gamma',
        getattr(par, 'gamma', 5.0 / 3.0),
    )
    scaling = _fast_source_scaling(fluid, par, gamma)
    mu = 1.0 / (2.0 - np.clip(xHI, 1.0e-12, 1.0))

    # The constructor below is the single primitive code-to-cgs conversion
    # for this source state.  Specific internal energy is supplied in code
    # units after applying the same EOS relation used by the existing source
    # equations; the source-state scaling is applied only after this boundary.
    temperature_code = runtime.temp_code[interior]
    temperature_cgs_K = temperature_code * code.unit_conversion['temperature_cgs_K']
    specific_energy_cgs_erg_g = (
        BOLTZMANN_CONSTANT_CGS
        * temperature_cgs_K
        / ((gamma - 1.0) * mu * PROTON_MASS_CGS)
    )
    interior_code = CodeFluidState(
        rho_code=runtime.rho_code[interior],
        vel_code=runtime.vel_code[interior],
        temp_code=temperature_code,
        specific_energy_code=(
            specific_energy_cgs_erg_g
            / code.unit_conversion['specific_energy_cgs_erg_g']
        ),
        xHI_dimensionless=xHI,
    )
    primitive_cgs = cgs_source_state_from_code(
        code_units=code,
        fluid=interior_code,
        boundary_code=mesh.boundary[interior.start : interior.stop + 1],
        volume_code=mesh.vol[interior],
    )
    source = CgsSourceState(
        boundary_cgs_cm=primitive_cgs.boundary_cgs_cm * scaling['scale_factor'],
        volume_cgs_cm3=primitive_cgs.volume_cgs_cm3 * scaling['density_factor'],
        rho_cgs_g_cm3=primitive_cgs.rho_cgs_g_cm3 / scaling['density_factor'],
        velocity_cgs_cm_s=primitive_cgs.velocity_cgs_cm_s,
        temperature_cgs_K=primitive_cgs.temperature_cgs_K / scaling['temperature_factor'],
        specific_energy_cgs_erg_g=specific_energy_cgs_erg_g,
        xHI_dimensionless=xHI,
    )
    temperature_physical = source.temperature_cgs_K
    rho_physical = source.rho_cgs_g_cm3
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
        default=DEFAULT_SIGMA_GAMMA_CGS_CM2,
    )
    source_rate = _optional_numeric_value(
        _parameter_value(par, 'radiative_transfer_source_photon_rate'),
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
        default=DEFAULT_EPSILON_GAMMA_CGS_ERG,
    )
    alpha_B = getattr(par, 'hydrogen_alpha_B', None)
    if alpha_B is not None:
        alpha_B = to_unit_value(alpha_B, code.volume_unit / code.time_unit)
    beta = getattr(par, 'hydrogen_beta', None)
    if beta is not None:
        beta = to_unit_value(beta, code.volume_unit / code.time_unit)
    return {
        'interior': interior,
        'boundary_cgs_cm': source.boundary_cgs_cm,
        'width_cgs_cm': np.diff(source.boundary_cgs_cm),
        'volume_cgs_cm3': source.volume_cgs_cm3,
        'radius_cgs_cm': as_named_array(
            to_unit_value(mesh.coordinate[interior], code.length_unit)
            * scaling['scale_factor']
        ),
        'radius_kpc': np.asarray(
            to_unit_value(mesh.coordinate[interior], code.length_unit)
            * scaling['scale_factor'] / kpc_in_cm,
            dtype=float,
        ),
        'xHI': source.xHI_dimensionless,
        'temperature_cgs_K': source.temperature_cgs_K,
        'specific_energy_cgs_erg_g': source.specific_energy_cgs_erg_g,
        'rho_cgs_g_cm3': source.rho_cgs_g_cm3,
        'active': thermochemistry_active_mask(
            rho_physical, par, scaling['density_factor']
        ),
        'nH_cgs_cm3': rho_physical * getattr(par, 'hydrogen_mass_fraction', 1.0) / PROTON_MASS_CGS,
        'gamma': gamma,
        'hydrogen_mass_fraction': getattr(par, 'hydrogen_mass_fraction', 1.0),
        'sigma_gamma_cgs_cm2': sigma_gamma,
        'source_rate_s': source_rate,
        'epsilon_gamma_cgs_erg': epsilon_gamma,
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
        'cmb_temperature_0_cgs_K': _optional_numeric_value(
            getattr(par, 'cmb_temperature_0', None),
            code.temperature_unit,
            default=2.7255 * unyt.K,
        ),
        'alpha_B_cgs_cm3_s': alpha_B,
        'beta_cgs_cm3_s': beta,
    }


def trace_spherical_tau(mesh, rho, xHI, hydrogen_mass_fraction, sigma_gamma):
    """Return the hydrogen optical depth per cell.

    This helper now requires ``mesh.code_units`` so the cgs conversion is
    explicit at the mesh boundary.
    """
    code = mesh.code_units
    rho_cgs_g_cm3 = to_unit_value(rho, code.density_unit)
    sigma_gamma_cgs_cm2 = to_unit_value(
        rh.photon_cross_section(sigma_gamma), code.area_unit
    )
    width_cgs_cm = to_unit_value(
        np.abs(mesh.boundary[1:] - mesh.boundary[:-1]),
        code.length_unit,
    )
    nH_cgs_cm3 = _cgs_hydrogen_number_density(rho_cgs_g_cm3, hydrogen_mass_fraction)
    xHI = rh.clip_neutral_fraction(xHI)
    tau = sigma_gamma_cgs_cm2 * nH_cgs_cm3 * xHI * width_cgs_cm
    return as_named_array(np.maximum(tau, 0.0))


def ionization_fraction_rate(state, ngamma_cgs_cm3):
    """Return the chemistry fraction rate for a float source state."""
    hydrogen_mass_fraction = state['hydrogen_mass_fraction']
    recombination = state['recombination']
    collisional_ionization = state['collisional_ionization']
    sigma_gamma_cgs_cm2 = state['sigma_gamma_cgs_cm2']
    alpha_B_cgs_cm3_s = state['alpha_B_cgs_cm3_s']
    beta_cgs_cm3_s = state['beta_cgs_cm3_s']
    return _cgs_static_neutral_fraction_rate(
        state['rho_cgs_g_cm3'],
        state['temperature_cgs_K'],
        state['xHI'],
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        recombination=recombination,
        collisional_ionization=collisional_ionization,
        ngamma_cgs_cm3=ngamma_cgs_cm3,
        sigma_gamma_cgs_cm2=sigma_gamma_cgs_cm2,
        recombination_coefficient_cgs_cm3_s=alpha_B_cgs_cm3_s,
        ionization_coefficient_cgs_cm3_s=beta_cgs_cm3_s,
    )


def thermal_rate(state, ngamma_cgs_cm3):
    """Return thermal source rate for a float source state."""
    hydrogen_mass_fraction = state['hydrogen_mass_fraction']
    recombination = state['recombination']
    collisional_ionization = state['collisional_ionization']
    sigma_gamma_cgs_cm2 = state['sigma_gamma_cgs_cm2']
    epsilon_gamma_cgs_erg = state['epsilon_gamma_cgs_erg']
    return _cgs_source_thermal_rate(
        state['rho_cgs_g_cm3'],
        state['temperature_cgs_K'],
        state['xHI'],
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        recombination=recombination,
        collisional_ionization=collisional_ionization,
        atomic_cooling=state.get('atomic_cooling', True),
        ngamma_cgs_cm3=ngamma_cgs_cm3,
        sigma_gamma_cgs_cm2=sigma_gamma_cgs_cm2,
        epsilon_gamma_cgs_erg=epsilon_gamma_cgs_erg,
        compton_cmb_enabled=state['compton_cmb_enabled'],
        compton_cmb_redshift=state['compton_cmb_redshift'],
        cmb_temperature_0_cgs_K=state['cmb_temperature_0_cgs_K'],
    )


def get_timestep(state, ngamma_cgs_cm3, remaining_s, dtmax_s, verbose=False):
    """Return a source substep for a float thermo-chemistry state."""
    source_CFL = state['source_CFL']
    dtmin_s = state['dtmin_s']
    candidates = []
    debug_lines = []
    ionization_limiter_enabled = (
        state['recombination']
        or state['collisional_ionization']
        or ngamma_cgs_cm3 is not None
    )
    if ionization_limiter_enabled:
        neutral_rate = ionization_fraction_rate(state, ngamma_cgs_cm3)
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
        source_thermal_rate = thermal_rate(state, ngamma_cgs_cm3)
        active = np.asarray(
            state.get('active', np.asarray(state['rho_cgs_g_cm3']) > 0.0),
            dtype=bool,
        )
        rho = np.where(active, state['rho_cgs_g_cm3'], 1.0)
        dudt = np.zeros_like(source_thermal_rate, dtype=float)
        dudt[active] = np.asarray(source_thermal_rate)[active] / rho[active]
        valid = (
            active
            & (np.abs(dudt) > 0.0)
            & (state['specific_energy_cgs_erg_g'] > 0.0)
        )
        if np.any(valid):
            valid_cells = np.where(valid)[0]
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                thermal_times = (
                    state['specific_energy_cgs_erg_g'][valid]
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
                            state['specific_energy_cgs_erg_g'][thermal_cell],
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
    state['temperature_cgs_K'] = (
        (state['gamma'] - 1.0)
        * mu
        * PROTON_MASS_CGS
        * state['specific_energy_cgs_erg_g']
        / BOLTZMANN_CONSTANT_CGS
    )
    state['temperature_cgs_K'] = np.maximum(state['temperature_cgs_K'], 1.0)


def ionization_fraction_implicit_update(state, ngamma_cgs_cm3, dt_s):
    """Implicitly update the chemistry fraction for a float state."""
    xHI = np.clip(np.asarray(state['xHI'], dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    hydrogen_mass_fraction = state['hydrogen_mass_fraction']
    recombination = state['recombination']
    collisional_ionization = state['collisional_ionization']
    sigma = state['sigma_gamma_cgs_cm2']
    alpha_value = state['alpha_B_cgs_cm3_s']
    updated = _cgs_static_neutral_fraction_implicit_update(
        state['rho_cgs_g_cm3'],
        state['temperature_cgs_K'],
        xHI,
        dt_s,
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        recombination=recombination,
        collisional_ionization=collisional_ionization,
        ngamma_cgs_cm3=ngamma_cgs_cm3,
        sigma_gamma_cgs_cm2=sigma,
        recombination_coefficient_cgs_cm3_s=alpha_value,
    )
    state['xHI'] = np.clip(updated, 1.0e-12, 1.0 - 1.0e-12)


def apply_state(state, fluid, par):
    """Copy a float thermo-chemistry state back to a fluid object."""
    interior = state['interior']
    fluid.xHI[interior] = state['xHI']
    code = _code_units(par)
    if code is None:
        raise ValueError("hydrogen thermo-chemistry requires configured code units")
    if hasattr(fluid, 'ngamma_code') and 'ngamma_cgs_cm3' in state:
        target = from_unit_value(state['ngamma_cgs_cm3'], code.number_density_unit)
        if np.ndim(target) == 2:
            fluid.ngamma_code[:, interior] = target
        else:
            fluid.ngamma_code[interior] = target
    if hasattr(fluid, 'temp_code') and 'temperature_cgs_K' in state:
        fluid.temp_code[interior] = from_unit_value(
            state['temperature_cgs_K'] * state.get('source_temperature_factor', 1.0),
            code.temperature_unit,
        )
    if hasattr(fluid, 'xHI') and getattr(getattr(fluid, 'eos', None), 'gamma', None) is not None:
        fluid.SetHydrogenMu(
            hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0)
        )
        fluid.SetPressure()
    fluid.time_code = from_unit_value(state['time_s'], code.time_unit)


def get_thermochemistry_source_timestep_fast(mesh, fluid, par, remaining):
    """Return a source substep for RT-coupled heating/chemistry."""
    state = _fast_source_state(mesh, fluid, par)
    code = _code_units(par)
    if code is None:
        raise ValueError("hydrogen thermo-chemistry requires configured code units")
    remaining_s = (
        to_unit_value(remaining, code.time_unit)
        * state['source_scale_factor']**2
    )
    if getattr(par, 'radiative_transfer', False):
        state['ngamma_cgs_cm3'] = rrt.trace_photon_density(state, par)
    sub_dt_s, thermal_rate = get_timestep(
        state,
        state.get('ngamma_cgs_cm3'),
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
    if hasattr(getattr(par, 'simulation', None), 'current_time'):
        time = getattr(fluid, 'time', None)
        if time is None:
            time = par.simulation.current_time
    else:
        # Lightweight source-test objects predate grouped parameters.
        time = getattr(
            par, 'fluid_time', getattr(par, 'time', getattr(fluid, 'time', 0.0))
        )
    tau = float(np.asarray(time, dtype=float).flat[0])
    scale_factor = float(cosmology.scale_factor_from_supercomoving(tau))
    return {
        'scale_factor': scale_factor,
        'density_factor': scale_factor**3,
        'temperature_factor': scale_factor**(3.0 * (gamma - 1.0)),
        'velocity_factor': scale_factor,
        'time_factor': scale_factor**2,
    }


def _rotational_specific_energy_code(mesh, fluid, par):
    """Return rotational specific energy in the conserved code units."""
    if not getattr(par, 'gas_rotational_energy', False):
        return np.zeros(
            int(par.mesh.grid_cells),
            dtype=float,
        )
    ghost_cells = int(par.mesh.ghost_cells)
    grid_cells = int(par.mesh.grid_cells)
    interior = slice(ghost_cells, ghost_cells + grid_cells)
    mass = np.asarray(fluid.Mass_code[interior], dtype=float)
    angular = np.asarray(fluid.AngularMomentum_code[interior], dtype=float)
    radius = np.abs(np.asarray(mesh.coordinate[interior], dtype=float))
    j = np.zeros_like(mass)
    np.divide(angular, mass, out=j, where=mass > 0.0)
    result = np.zeros_like(mass)
    valid = (mass > 0.0) & (radius > 0.0) & np.isfinite(radius)
    result[valid] = 0.5 * j[valid]**2 / radius[valid]**2
    return result


def _fast_source_state(mesh, fluid, par):
    """Return a cgs float snapshot for the fast thermo-chemistry path."""
    code = _code_units(par)
    if code is None:
        raise ValueError("hydrogen thermo-chemistry requires configured code units")
    unit_conversion = code.unit_conversion
    ghost_cells = int(par.mesh.ghost_cells)
    grid_cells = int(par.mesh.grid_cells)
    interior = slice(ghost_cells, ghost_cells + grid_cells)
    gamma = getattr(getattr(fluid, 'eos', None), 'gamma', par.hydrodynamics.gamma)
    scaling = _fast_source_scaling(fluid, par, gamma)
    rho_cgs_g_cm3 = (
        np.asarray(fluid.rho_code[interior], dtype=float)
        * unit_conversion['density_cgs_g_cm3']
        / scaling['density_factor']
    )
    temperature_cgs_K = (
        np.asarray(fluid.temp_code[interior], dtype=float)
        * unit_conversion['temperature_cgs_K']
        / scaling['temperature_factor']
    )
    velocity_supercomoving_cgs_cm_s = (
        np.asarray(fluid.vel_code[interior], dtype=float)
        * unit_conversion['velocity_cgs_cm_s']
    )
    vel_cgs_cm_s = velocity_supercomoving_cgs_cm_s / scaling['velocity_factor']
    mass_g = np.asarray(fluid.Mass_code[interior], dtype=float) * unit_conversion['mass_g']
    energy_supercomoving_cgs_erg = (
        np.asarray(fluid.Energy_code[interior], dtype=float)
        * unit_conversion['energy_cgs_erg']
    )
    rotational_specific_code = _rotational_specific_energy_code(mesh, fluid, par)
    state = {
        'interior': interior,
        'boundary_cgs_cm': as_named_array(
            np.asarray(
                mesh.boundary[interior.start : interior.stop + 1], dtype=float
            ) * unit_conversion['length_cgs_cm']
            * scaling['scale_factor']
        ),
        'width_cgs_cm': as_named_array(
            np.asarray(mesh.xdelta[interior], dtype=float)
            * unit_conversion['length_cgs_cm']
            * scaling['scale_factor']
        ),
        'volume_cgs_cm3': as_named_array(
            np.asarray(mesh.vol[interior], dtype=float)
            * unit_conversion['volume_cgs_cm3']
            * scaling['density_factor']
        ),
        'rho_cgs_g_cm3': rho_cgs_g_cm3,
        'active': thermochemistry_active_mask(
            rho_cgs_g_cm3, par, scaling['density_factor']
        ),
        'temperature_cgs_K': temperature_cgs_K,
        'xHI': as_named_array(
            fluid.xHI[interior]
            if hasattr(fluid, 'xHI')
            else np.ones(int(par.mesh.grid_cells)),
        ),
        'nH_cgs_cm3': rho_cgs_g_cm3 * getattr(par, 'hydrogen_mass_fraction', 1.0) / PROTON_MASS_CGS,
        'gamma': gamma,
        'source_scale_factor': scaling['scale_factor'],
        'source_temperature_factor': scaling['temperature_factor'],
        'source_density_factor': scaling['density_factor'],
        'mu': (
            np.asarray(fluid.mu[interior], dtype=float)
            if hasattr(fluid, 'mu')
            else rh.mean_molecular_weight_mu(
                np.asarray(
                    fluid.xHI[interior]
                    if hasattr(fluid, 'xHI')
                    else np.ones(int(par.mesh.grid_cells)),
                    dtype=float,
                ),
                hydrogen_mass_fraction=getattr(par, 'hydrogen_mass_fraction', 1.0),
            )
        ),
        'hydrogen_mass_fraction': getattr(par, 'hydrogen_mass_fraction', 1.0),
        'sigma_gamma_cgs_cm2': _optional_numeric_value(
            getattr(par, 'hydrogen_sigma_gamma', None),
            code.area_unit,
            default=DEFAULT_SIGMA_GAMMA_CGS_CM2,
        ),
        'ngamma_cgs_cm3': (
                (
                    np.asarray(
                        fluid.ngamma_code[:, interior]
                        if np.ndim(fluid.ngamma_code) == 2
                        else fluid.ngamma_code[interior],
                        dtype=float,
                    ) * unit_conversion['number_density_cgs_cm3']
                )
            / scaling['density_factor']
            if (
                getattr(par, 'hydrogen_radiation_field', False)
                or getattr(par, 'radiative_transfer', False)
            )
            and hasattr(fluid, 'ngamma_code')
            else None
        ),
        'source_rate_s': _optional_numeric_value(
            _parameter_value(par, 'radiative_transfer_source_photon_rate'),
            1.0 / code.time_unit,
            default=0.0,
        ),
        'epsilon_gamma_cgs_erg': _optional_numeric_value(
            getattr(par, 'hydrogen_epsilon_gamma', None),
            code.energy_unit,
            default=DEFAULT_EPSILON_GAMMA_CGS_ERG,
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
        'cmb_temperature_0_cgs_K': _optional_numeric_value(
            getattr(par, 'cmb_temperature_0', None),
            code.temperature_unit,
            default=2.7255 * unyt.K,
        ),
        'alpha_B_cgs_cm3_s': _optional_numeric_value(
            getattr(par, 'hydrogen_alpha_B', None),
            code.volume_unit / code.time_unit,
            default=None,
        ),
        'beta_cgs_cm3_s': _optional_numeric_value(
            getattr(par, 'hydrogen_beta', None),
            code.volume_unit / code.time_unit,
            default=None,
        ),
    }
    hydro_temperature_floor = getattr(par, 'hydro_temperature_floor', None)
    state['temperature_floor_cgs_K'] = (
        0.0
        if hydro_temperature_floor is None
        else max(
            0.0,
            float(hydro_temperature_floor)
            * unit_conversion['temperature_cgs_K']
            / scaling['temperature_factor'],
        )
    )
    state['temperature_floor_tolerance'] = max(
        0.0,
        float(getattr(par, 'hydrogen_source_floor_temperature_tolerance', 1.0e-2)),
    )
    skip_floor_cells = bool(
        getattr(par, 'hydrogen_source_skip_floor_cells', False)
    )
    source_density_floor = _optional_numeric_value(
        getattr(par, 'hydrogen_source_density_floor', None),
        unyt.g / unyt.cm**3,
        default=None,
    )
    if skip_floor_cells and source_density_floor is not None:
        if state['temperature_floor_cgs_K'] > 0.0:
            at_temperature_floor = (
                temperature_cgs_K
                <= state['temperature_floor_cgs_K']
                * (1.0 + state['temperature_floor_tolerance'])
            )
            skip = (
                np.asarray(state['active'], dtype=bool)
                & (rho_cgs_g_cm3 <= float(source_density_floor))
                & at_temperature_floor
            )
            if getattr(par, 'hydrogen_implicit_debug', False):
                diagnostic_cells = np.where(
                    rho_cgs_g_cm3 <= float(source_density_floor)
                )[0]
                for cell in diagnostic_cells:
                    floor = state['temperature_floor_cgs_K']
                    print(
                        '[hydrogen source floor mask] '
                        'cell=%d rho_cgs_g_cm3=% .6e temperature_cgs_K=% .6e '
                        'temperature_floor_cgs_K=% .6e temperature_ratio=% .6e '
                        'skip=%s'
                        % (
                            int(cell),
                            float(rho_cgs_g_cm3[cell]),
                            float(temperature_cgs_K[cell]),
                            float(floor),
                            float(temperature_cgs_K[cell] / floor),
                            bool(skip[cell]),
                        )
                    )
            state['active'] = np.asarray(state['active'], dtype=bool) & ~skip
        # Apply this mask before the source energy/temperature floor. These
        # cells retain their hydro state and do not enter thermo-chemistry.
    if state['thermal_coupling']:
        state['vel_cgs_cm_s'] = vel_cgs_cm_s
        # Vacuum cells do not participate in chemistry.  Keep their specific
        # energy finite so the network can carry them through the source
        # update without generating NaNs or wasting convergence iterations.
        specific_total_supercomoving = np.divide(
            energy_supercomoving_cgs_erg,
            mass_g,
            out=np.zeros_like(energy_supercomoving_cgs_erg, dtype=float),
            where=mass_g > 0.0,
        )
        specific_kinetic_supercomoving = (
            0.5 * velocity_supercomoving_cgs_cm_s**2
        )
        # Conserved Energy includes rotational kinetic energy when enabled.
        # Remove it before sending the thermal state to the chemistry solver.
        rotational_specific_cgs_erg_g = (
            rotational_specific_code
            * unit_conversion['velocity_cgs_cm_s']**2
            / scaling['temperature_factor']
        )
        specific_internal_physical = np.maximum(
            specific_total_supercomoving
            - specific_kinetic_supercomoving
            - rotational_specific_code
            * unit_conversion['velocity_cgs_cm_s']**2,
            0.0,
        ) / scaling['temperature_factor']
        state['specific_rotational_energy_cgs_erg_g'] = rotational_specific_cgs_erg_g
        state['specific_rotational_energy_code'] = rotational_specific_code
        state['specific_kinetic_energy_supercomoving_cgs_erg_g'] = (
            specific_kinetic_supercomoving
        )
        state['specific_kinetic_energy_cgs_erg_g'] = 0.5 * state['vel_cgs_cm_s']**2
        state['specific_total_energy_cgs_erg_g'] = (
            specific_internal_physical
            + state['specific_kinetic_energy_cgs_erg_g']
        )
        state['specific_energy_cgs_erg_g'] = np.maximum(
            state['specific_total_energy_cgs_erg_g'] - state['specific_kinetic_energy_cgs_erg_g'],
            0.0,
        )
        _fast_update_temperature_from_energy(state)
    return state


def _fast_update_temperature_from_energy(state):
    """Update float temperature from total specific energy and mean molecular weight."""
    internal_specific = np.maximum(
        state['specific_total_energy_cgs_erg_g'] - state['specific_kinetic_energy_cgs_erg_g'],
        0.0,
    )
    if state.get('hydrogen_update_mu', False):
        state['mu'] = rh.mean_molecular_weight_mu(
            state['xHI'],
            hydrogen_mass_fraction=state['hydrogen_mass_fraction'],
        )
    temperature_floor = float(state.get('temperature_floor_cgs_K', 0.0) or 0.0)
    if temperature_floor > 0.0:
        energy_floor = (
            BOLTZMANN_CONSTANT_CGS
            * temperature_floor
            / (
                (state['gamma'] - 1.0)
                * np.maximum(state['mu'], 1.0e-99)
                * PROTON_MASS_CGS
            )
        )
        active = np.asarray(
            state.get('active', np.asarray(state['rho_cgs_g_cm3']) > 0.0),
            dtype=bool,
        )
        internal_specific = np.where(
            active,
            np.maximum(internal_specific, energy_floor),
            internal_specific,
        )
        # Keep the conserved source state consistent with the temperature
        # floor. A pressure-only floor would leave E-K equal to zero.
        state['specific_energy_cgs_erg_g'] = np.where(
            active,
            internal_specific,
            state['specific_energy_cgs_erg_g'],
        )
        state['specific_total_energy_cgs_erg_g'] = (
            state['specific_energy_cgs_erg_g']
            + state['specific_kinetic_energy_cgs_erg_g']
        )
    state['temperature_cgs_K'] = (
        (state['gamma'] - 1.0)
        * state['mu']
        * PROTON_MASS_CGS
        * internal_specific
        / BOLTZMANN_CONSTANT_CGS
    )


def _fast_apply_thermal_source(state, thermal_rate_cgs_erg_cm3_s, dt_s):
    """Apply thermal source terms in float cgs units."""
    active = np.asarray(
        state.get('active', np.asarray(state['rho_cgs_g_cm3']) > 0.0),
        dtype=bool,
    )
    rho = np.where(active, state['rho_cgs_g_cm3'], 1.0)
    energy_update = np.zeros_like(state['specific_total_energy_cgs_erg_g'])
    energy_update[active] = (
        np.asarray(thermal_rate_cgs_erg_cm3_s)[active] / rho[active] * dt_s
    )
    state['specific_total_energy_cgs_erg_g'] += energy_update
    state['specific_total_energy_cgs_erg_g'] = np.maximum(
        state['specific_total_energy_cgs_erg_g'],
        state['specific_kinetic_energy_cgs_erg_g'],
    )
    _fast_update_temperature_from_energy(state)


def _apply_compton_only_source(state, dt_s):
    """Advance the fixed-composition Compton relaxation exactly."""
    temperature = np.asarray(state['temperature_cgs_K'], dtype=float)
    old_temperature = temperature.copy()
    active = np.asarray(
        state.get('active', np.asarray(state['rho_cgs_g_cm3']) > 0.0),
        dtype=bool,
    )
    xHI = np.clip(np.asarray(state['xHI'], dtype=float), 0.0, 1.0)
    nH = _cgs_hydrogen_number_density(
        state['rho_cgs_g_cm3'], state['hydrogen_mass_fraction']
    )
    electron_density = nH * (1.0 - xHI)
    specific_heat = (
        BOLTZMANN_CONSTANT_CGS
        / ((state['gamma'] - 1.0) * state['mu'] * PROTON_MASS_CGS)
    )
    cmb_temperature = (
        state['cmb_temperature_0_cgs_K']
        * (1.0 + float(state['compton_cmb_redshift']))
    )
    zero_temperature_rate = cmb_compton_rate(
        np.zeros_like(temperature),
        electron_density,
        enabled=True,
        redshift=state['compton_cmb_redshift'],
        cmb_temperature_0_cgs_K=state['cmb_temperature_0_cgs_K'],
    )
    coupling_rate = np.divide(
        zero_temperature_rate,
        state['rho_cgs_g_cm3'] * specific_heat * cmb_temperature,
        out=np.zeros_like(temperature),
        where=(state['rho_cgs_g_cm3'] > 0.0) & (specific_heat > 0.0),
    )
    updated_temperature = cmb_temperature + (
        temperature - cmb_temperature
    ) * np.exp(-coupling_rate * dt_s)
    temperature = np.where(active, updated_temperature, old_temperature)
    state['specific_total_energy_cgs_erg_g'] = (
        specific_heat * temperature
        + state['specific_kinetic_energy_cgs_erg_g']
    )
    state['specific_energy_cgs_erg_g'] = specific_heat * temperature
    _fast_update_temperature_from_energy(state)


def _coupled_implicit_source_update(
    state,
    dt_s,
    ngamma_cgs_cm3=None,
    tolerance=1.0e-6,
    max_iterations=32,
    trust_region=False,
    absolute_temperature_tolerance=0.0,
    absolute_xhi_tolerance=0.0,
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

    rho = np.asarray(state['rho_cgs_g_cm3'], dtype=float)
    active_cells = np.asarray(state.get('active', rho > 0.0), dtype=bool)
    rho_for_update = np.where(active_cells, rho, 1.0)
    kinetic = np.asarray(state['specific_kinetic_energy_cgs_erg_g'], dtype=float)
    energy_old = np.asarray(state['specific_energy_cgs_erg_g'], dtype=float).copy()
    x_old = np.asarray(state['xHI'], dtype=float).copy()
    dt_value = float(np.asarray(dt_s, dtype=float))
    if not (
        np.all(np.isfinite(rho))
        and np.all(rho_for_update > 0.0)
        and np.all(np.isfinite(energy_old))
        and np.all(np.isfinite(x_old))
    ):
        return False

    def _record_failure(reason, determinant=None):
        active = active_cells & np.isfinite(residual_energy) & np.isfinite(residual_x)
        residual_norm = np.maximum(
            np.abs(residual_energy), np.abs(residual_x)
        )
        unconverged = active_cells & (
            ~np.isfinite(residual_energy)
            | ~np.isfinite(residual_x)
            | (residual_norm > tolerance)
        )
        unconverged_cells = [
            {
                'cell': int(cell),
                'residual_energy': float(residual_energy[cell]),
                'residual_xHI': float(residual_x[cell]),
            }
            for cell in np.where(unconverged)[0]
        ]
        _, _, _, _, failure_trial = _residual(log_energy, logit_x)
        heating_rate = thermal_rate(
            dict(failure_trial, atomic_cooling=False), ngamma_cgs_cm3
        )
        total_thermal_rate = thermal_rate(failure_trial, ngamma_cgs_cm3)
        cooling_rate = np.asarray(heating_rate) - np.asarray(total_thermal_rate)
        alpha_rate = state.get('alpha_B_cgs_cm3_s')
        if alpha_rate is None:
            alpha_rate = _cgs_alpha_B(failure_trial['temperature_cgs_K'])
        else:
            alpha_rate = np.full_like(
                np.asarray(failure_trial['temperature_cgs_K'], dtype=float),
                float(alpha_rate),
            )
        n_hydrogen = _cgs_hydrogen_number_density(
            failure_trial['rho_cgs_g_cm3'],
            failure_trial['hydrogen_mass_fraction'],
        )
        electron_density = n_hydrogen * (1.0 - failure_trial['xHI'])
        compton_rate = cmb_compton_rate(
            failure_trial['temperature_cgs_K'],
            electron_density,
            enabled=failure_trial['compton_cmb_enabled'],
            redshift=failure_trial['compton_cmb_redshift'],
            cmb_temperature_0_cgs_K=failure_trial['cmb_temperature_0_cgs_K'],
        )
        photoheating_rate = np.asarray(heating_rate) - np.asarray(compton_rate)
        failed_cells = []
        if determinant is not None:
            singular = active & (np.abs(determinant) <= 1.0e-30)
            for cell in np.where(singular)[0]:
                failed_cells.append({
                    'cell': int(cell),
                    'temperature_cgs_K': float(np.asarray(state['temperature_cgs_K'])[cell]),
                    'specific_energy_cgs_erg_g': float(energy_old[cell]),
                    'xHI': float(x_old[cell]),
                    'residual_energy': float(residual_energy[cell]),
                    'residual_xHI': float(residual_x[cell]),
                    'jacobian_determinant': float(determinant[cell]),
                    'alpha_B_cgs_cm3_s': float(np.asarray(alpha_rate)[cell]),
                    'heating_cgs_erg_cm3_s': float(np.asarray(heating_rate)[cell]),
                    'cooling_cgs_erg_cm3_s': float(np.asarray(cooling_rate)[cell]),
                    'nH_cgs_cm3': float(np.asarray(n_hydrogen)[cell]),
                    'ne_cgs_cm3': float(np.asarray(electron_density)[cell]),
                    'compton_heating_cgs_erg_cm3_s': float(np.asarray(compton_rate)[cell]),
                    'photoheating_cgs_erg_cm3_s': float(np.asarray(photoheating_rate)[cell]),
                })
        if not np.any(active):
            state['_implicit_failure'] = {
                'reason': reason, 'dt_s': dt_value,
                'unconverged_cells': unconverged_cells,
                'failed_cells': failed_cells,
            }
            return False
        norm = np.maximum(np.abs(residual_energy), np.abs(residual_x))
        norm = np.where(active, norm, -np.inf)
        index = int(np.argmax(norm))
        state['_implicit_failure'] = {
            'reason': reason,
            'dt_s': dt_value,
            'cell': index,
            'rho_cgs_g_cm3': float(np.asarray(state['rho_cgs_g_cm3'])[index]),
            'temperature_cgs_K': float(np.asarray(state['temperature_cgs_K'])[index]),
            'specific_energy_cgs_erg_g': float(energy_old[index]),
            'xHI': float(x_old[index]),
            'trial_temperature_cgs_K': float(
                np.asarray(failure_trial['temperature_cgs_K'])[index]
            ),
            'trial_specific_energy_cgs_erg_g': float(
                np.asarray(failure_trial['specific_energy_cgs_erg_g'])[index]
            ),
            'trial_xHI': float(np.asarray(failure_trial['xHI'])[index]),
            'residual_energy': float(residual_energy[index]),
            'residual_xHI': float(residual_x[index]),
            'jacobian_determinant': (
                None if determinant is None else float(determinant[index])
            ),
            'alpha_B_cgs_cm3_s': float(np.asarray(alpha_rate)[index]),
            'heating_cgs_erg_cm3_s': float(np.asarray(heating_rate)[index]),
            'cooling_cgs_erg_cm3_s': float(np.asarray(cooling_rate)[index]),
            'nH_cgs_cm3': float(np.asarray(n_hydrogen)[index]),
            'ne_cgs_cm3': float(np.asarray(electron_density)[index]),
            'compton_heating_cgs_erg_cm3_s': float(np.asarray(compton_rate)[index]),
            'photoheating_cgs_erg_cm3_s': float(np.asarray(photoheating_rate)[index]),
            'failed_cells': failed_cells,
            'unconverged_cells': unconverged_cells,
        }
        return False

    # Keep a tiny numerical floor even when no physical floor is configured.
    # The physical floor is enforced by _fast_update_temperature_from_energy
    # for the initial state and every Newton trial.
    energy_floor = 1.0e-30
    x_floor = 1.0e-12
    energy_old = np.maximum(energy_old, energy_floor)
    x_old = np.clip(x_old, x_floor, 1.0 - x_floor)
    temperature_floor = float(state.get('temperature_floor_cgs_K', 0.0) or 0.0)
    physical_energy_floor = np.zeros_like(energy_old)
    if temperature_floor > 0.0:
        physical_energy_floor = (
            BOLTZMANN_CONSTANT_CGS
            * temperature_floor
            / (
                (state['gamma'] - 1.0)
                * np.maximum(state['mu'], 1.0e-99)
                * PROTON_MASS_CGS
            )
        )
        energy_old = np.where(
            active_cells,
            np.maximum(energy_old, physical_energy_floor),
            energy_old,
        )
    relative_tolerance = float(tolerance)
    absolute_energy_tolerance = (
        BOLTZMANN_CONSTANT_CGS * max(float(absolute_temperature_tolerance), 0.0)
        / (
            (state['gamma'] - 1.0)
            * np.maximum(np.asarray(state['mu'], dtype=float), 1.0e-99)
            * PROTON_MASS_CGS
        )
    )
    # Use R_E / e_old when the old energy is resolved.  When e_old is below
    # the energy represented by the absolute temperature tolerance, disable
    # the relative term and compare R_E directly with that absolute scale.
    energy_reference = np.abs(energy_old)
    small_energy = (
        (absolute_energy_tolerance > 0.0)
        & (energy_reference <= absolute_energy_tolerance)
    )
    energy_residual_scale = np.where(
        small_energy,
        absolute_energy_tolerance,
        energy_reference,
    )
    energy_residual_scale = np.maximum(
        energy_residual_scale, np.finfo(float).tiny
    )
    energy_residual_tolerance = np.where(
        small_energy,
        1.0,
        relative_tolerance + np.divide(
            absolute_energy_tolerance,
            energy_reference,
            out=np.zeros_like(energy_reference),
            where=energy_reference > 0.0,
        ),
    )
    xhi_residual_tolerance = (
        relative_tolerance * np.maximum(np.abs(x_old), 1.0)
        + max(float(absolute_xhi_tolerance), 0.0)
    )
    # Finite-difference Jacobians can leave a residual a few ulps above the
    # requested normalized threshold.  Allow a small numerical margin while
    # retaining the user-specified relative/absolute scale.
    energy_residual_acceptance = 1.5 * energy_residual_tolerance
    # The transformed variables are logarithmic.  A unit initial radius only
    # permits a factor-e energy change and can spend the entire Newton budget
    # expanding away from a cold temperature floor during stiff Compton
    # heating.  Start at the established cap; the line search still rejects
    # any step that does not reduce the coupled residual.
    trust_radius = np.full_like(energy_old, 4.0, dtype=float)
    trust_radius_min = 1.0e-6
    if not np.isfinite(dt_value) or dt_value < 0.0:
        return False
    if dt_value == 0.0:
        state['specific_energy_cgs_erg_g'] = np.where(
            active_cells, energy_old, state['specific_energy_cgs_erg_g']
        )
        state['specific_total_energy_cgs_erg_g'] = (
            state['specific_energy_cgs_erg_g'] + kinetic
        )
        state['xHI'] = np.where(active_cells, x_old, state['xHI'])
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
        trial['specific_energy_cgs_erg_g'] = energy
        trial['specific_total_energy_cgs_erg_g'] = energy + kinetic
        trial['xHI'] = xhi
        if trial.get('hydrogen_update_mu', False):
            trial['mu'] = rh.mean_molecular_weight_mu(
                xhi,
                hydrogen_mass_fraction=trial['hydrogen_mass_fraction'],
            )
        _fast_update_temperature_from_energy(trial)
        # _fast_update_temperature_from_energy may impose the physical floor;
        # use the clamped value in the coupled residual as well.
        energy = np.asarray(trial['specific_energy_cgs_erg_g'], dtype=float)
        xhi = np.asarray(trial['xHI'], dtype=float)
        thermal = thermal_rate(trial, ngamma_cgs_cm3)
        chemistry = ionization_fraction_rate(trial, ngamma_cgs_cm3)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            energy_residual = (
                energy - energy_old - dt_value * thermal / rho_for_update
            ) / energy_residual_scale
        chemistry_residual = xhi - x_old - dt_value * chemistry
        return energy_residual, chemistry_residual, energy, xhi, trial

    log_energy = np.log(energy_old)
    logit_x = _logit(x_old)
    residual_energy, residual_x, _, _, initial_trial = _residual(
        log_energy, logit_x
    )
    def _floor_constraint(trial):
        if temperature_floor <= 0.0:
            return np.zeros_like(active_cells, dtype=bool)
        thermal = np.asarray(thermal_rate(trial, ngamma_cgs_cm3), dtype=float)
        return (
            active_cells
            & (
                np.asarray(trial['temperature_cgs_K'])
                <= temperature_floor
                * (1.0 + state.get('temperature_floor_tolerance', 1.0e-6))
            )
            & (thermal <= 0.0)
        )

    floor_constrained = _floor_constraint(initial_trial)
    converged = np.zeros_like(energy_old, dtype=bool)
    finite = np.isfinite(residual_energy) & np.isfinite(residual_x)
    converged[~active_cells] = True
    converged[finite] = (
        (np.abs(residual_energy[finite]) <= energy_residual_acceptance[finite])
        & (np.abs(residual_x[finite]) <= xhi_residual_tolerance[finite])
    )
    converged[floor_constrained & finite] = (
        np.abs(residual_x[floor_constrained & finite])
        <= xhi_residual_tolerance[floor_constrained & finite]
    )

    finite_difference_step = 1.0e-7
    for _ in range(int(max_iterations)):
        _, _, _, _, current_trial = _residual(log_energy, logit_x)
        floor_constrained = _floor_constraint(current_trial)
        current_energy_ok = np.abs(residual_energy) <= energy_residual_acceptance
        current_xhi_ok = np.abs(residual_x) <= xhi_residual_tolerance
        converged |= active_cells & current_energy_ok & current_xhi_ok
        converged |= active_cells & floor_constrained & current_xhi_ok
        active = ~converged & active_cells
        if not np.any(active):
            break

        forward_log_energy = log_energy + finite_difference_step
        floor_log_energy = np.full_like(log_energy, -np.inf)
        positive_floor = active_cells & (physical_energy_floor > 0.0)
        floor_log_energy[positive_floor] = (
            np.log(physical_energy_floor[positive_floor])
            + finite_difference_step
        )
        # At the physical floor, a symmetric/small perturbation can remain
        # clipped by _fast_update_temperature_from_energy. Use a strictly
        # one-sided forward probe above the floor and its actual log-distance.
        forward_log_energy = np.maximum(forward_log_energy, floor_log_energy)
        energy_log_step = forward_log_energy - log_energy
        energy_plus, x_plus = _residual(
            forward_log_energy,
            logit_x,
        )[:2]
        energy_x_plus, x_x_plus = _residual(
            log_energy,
            logit_x + finite_difference_step,
        )[:2]
        safe_energy_log_step = np.where(
            energy_log_step > 0.0,
            energy_log_step,
            np.nan,
        )
        jacobian_11 = (energy_plus - residual_energy) / safe_energy_log_step
        jacobian_21 = (x_plus - residual_x) / safe_energy_log_step
        jacobian_12 = (energy_x_plus - residual_energy) / finite_difference_step
        jacobian_22 = (x_x_plus - residual_x) / finite_difference_step
        determinant = jacobian_11 * jacobian_22 - jacobian_12 * jacobian_21
        good = (
            active
            & ~floor_constrained
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
            & (
                floor_constrained
                | (np.abs(residual_energy) <= energy_residual_acceptance)
            )
            & np.isfinite(jacobian_22)
            & (np.abs(jacobian_22) > 1.0e-30)
        )
        scalar_energy = (
            active
            & ~good
            & ~scalar_chemistry
            & (np.abs(residual_x) <= xhi_residual_tolerance)
            & np.isfinite(jacobian_11)
            & (np.abs(jacobian_11) > 1.0e-30)
        )
        solvable = good | scalar_chemistry | scalar_energy
        if not np.any(solvable):
            return _record_failure('no_solvable_jacobian', determinant)

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
        if trust_region:
            step_norm = np.maximum(np.abs(delta_energy), np.abs(delta_x))
            step_scale = np.minimum(
                1.0,
                trust_radius / np.maximum(step_norm, 1.0e-99),
            )
            delta_energy = np.where(finite_delta, delta_energy * step_scale, delta_energy)
            delta_x = np.where(finite_delta, delta_x * step_scale, delta_x)
        accepted = np.zeros_like(active, dtype=bool)
        current_norm = np.where(
            floor_constrained,
            np.abs(residual_x),
            np.maximum(np.abs(residual_energy), np.abs(residual_x)),
        )
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
            trial_log_energy = log_energy + damping * delta_energy
            trial_logit_x = logit_x + damping * delta_x
            trial_energy_residual, trial_x_residual = _residual(
                trial_log_energy,
                trial_logit_x,
            )[:2]
            trial_norm = np.where(
                floor_constrained,
                np.abs(trial_x_residual),
                np.maximum(np.abs(trial_energy_residual), np.abs(trial_x_residual)),
            )
            improve = (
                finite_delta
                & ~accepted
                & np.isfinite(trial_norm)
                & (
                    trial_norm
                    <= current_norm
                    * (1.0 - (1.0e-4 if trust_region else 0.0))
                )
            )
            if np.any(improve):
                log_energy[improve] = trial_log_energy[improve]
                logit_x[improve] = trial_logit_x[improve]
                residual_energy[improve] = trial_energy_residual[improve]
                residual_x[improve] = trial_x_residual[improve]
                accepted[improve] = True

        converged |= accepted & (
            (
                floor_constrained
                & (np.abs(residual_x) <= xhi_residual_tolerance)
            )
            | (
                ~floor_constrained
                & (np.abs(residual_energy) <= energy_residual_acceptance)
                & (np.abs(residual_x) <= xhi_residual_tolerance)
            )
        )
        # Reject the whole candidate interval as soon as any still-active
        # cell cannot reduce its residual. The adaptive driver then retries
        # from the original state with a smaller source timestep.
        if np.any(active & ~accepted):
            if trust_region:
                rejected = active & ~accepted
                trust_radius[rejected] *= 0.25
                if np.any(trust_radius[rejected] >= trust_radius_min):
                    continue
            return _record_failure('line_search_no_improvement', determinant)
        if trust_region:
            trust_radius[accepted] = np.minimum(
                4.0, trust_radius[accepted] * 1.5
            )

    final_residual_energy, final_residual_x, _, _, final_trial = _residual(
        log_energy, logit_x
    )
    floor_constrained = _floor_constraint(final_trial)
    final_finite = (
        np.isfinite(final_residual_energy)
        & np.isfinite(final_residual_x)
    )
    final_norm = np.where(
        floor_constrained,
        np.abs(final_residual_x),
        np.maximum(
            np.abs(final_residual_energy),
            np.abs(final_residual_x),
        ),
    )
    # A damped Newton step can fail to improve a nearly converged state. Do
    # not reject that state when its final finite residuals meet tolerance.
    final_acceptable = (
        ~active_cells
        | (
            final_finite
            & (np.abs(final_residual_energy) <= energy_residual_acceptance)
            & (np.abs(final_residual_x) <= xhi_residual_tolerance)
        )
    )
    if np.all(final_acceptable):
        converged[:] = True
    else:
        converged |= active_cells & final_finite & (
            (np.abs(final_residual_energy) <= energy_residual_acceptance)
            & (np.abs(final_residual_x) <= xhi_residual_tolerance)
        )
    if not np.all(converged):
        residual_energy = final_residual_energy
        residual_x = final_residual_x
        return _record_failure('maximum_newton_iterations')

    _, _, energy, xhi, trial = _residual(log_energy, logit_x)
    state['specific_energy_cgs_erg_g'] = np.where(
        active_cells, energy, state['specific_energy_cgs_erg_g']
    )
    state['specific_total_energy_cgs_erg_g'] = (
        state['specific_energy_cgs_erg_g'] + kinetic
    )
    state['xHI'] = np.where(
        active_cells,
        np.clip(xhi, x_floor, 1.0 - x_floor),
        state['xHI'],
    )
    if state.get('hydrogen_update_mu', False):
        state['mu'] = trial['mu']
    _fast_update_temperature_from_energy(state)
    return True


def _copy_fast_source_state(state):
    """Copy a numeric source state for a trial implicit update."""
    return copy.deepcopy(state)


def _explicit_source_state_update(state, remaining_s, par):
    """Advance a local source state with the existing explicit subcycler."""
    source_steps = 0
    while remaining_s > 0.0:
        if state['hydrogen_update_mu']:
            state['mu'] = rh.mean_molecular_weight_mu(
                state['xHI'],
                hydrogen_mass_fraction=state['hydrogen_mass_fraction'],
            )
        if state['thermal_coupling']:
            _fast_update_temperature_from_energy(state)
        temperature_before = np.asarray(
            state['temperature_cgs_K'], dtype=float
        ).copy()
        sub_dt_s, thermal_rate = get_timestep(
            state,
            state.get('ngamma_cgs_cm3'),
            remaining_s,
            remaining_s,
            verbose=getattr(par, 'verbose', 0) >= 2,
        )
        if not np.isfinite(sub_dt_s) or sub_dt_s <= 0.0:
            sub_dt_s = remaining_s
        sub_dt_s = min(sub_dt_s, remaining_s)

        if state['thermal_coupling']:
            _fast_apply_thermal_source(state, thermal_rate, sub_dt_s)
        if (
            state['recombination']
            or state['collisional_ionization']
            or state.get('ngamma_cgs_cm3') is not None
            or getattr(par, 'radiative_transfer', False)
            or getattr(par, 'hydrogen_radiation_field', False)
        ):
            ionization_fraction_implicit_update(
                state,
                state.get('ngamma_cgs_cm3'),
                sub_dt_s,
            )
        if state['hydrogen_update_mu']:
            state['mu'] = rh.mean_molecular_weight_mu(
                state['xHI'],
                hydrogen_mass_fraction=state['hydrogen_mass_fraction'],
            )
        if state['thermal_coupling']:
            _fast_update_temperature_from_energy(state)
        check_source_temperature(
            state, par, temperature_before,
            stage='hydrogen explicit source', source_step=source_steps + 1,
        )
        remaining_s -= sub_dt_s
        source_steps += 1
    return source_steps


def _split_implicit_source_state_update(state, dt_s, par):
    """Advance sources with explicit energy and implicit chemistry.

    The radiation field, when supplied by the outer source driver, is held
    fixed during this operator-split update. Radiative transfer is refreshed
    once per hydro step; this routine evolves only the thermal and chemical
    response to that field.
    """
    remaining_s = float(np.asarray(dt_s, dtype=float))
    if not np.isfinite(remaining_s) or remaining_s < 0.0:
        raise ValueError("split-implicit source timestep must be finite and non-negative")
    if remaining_s == 0.0:
        return 0

    source_steps = 0
    trial_dt_s = remaining_s
    max_subcycles = int(getattr(par, 'hydrogen_split_implicit_max_subcycles', 100000))
    dtmin_s = float(state.get('dtmin_s', 0.0) or 0.0)
    active = np.asarray(
        state.get('active', np.asarray(state['rho_cgs_g_cm3']) > 0.0),
        dtype=bool,
    )
    while remaining_s > 0.0:
        candidate_dt_s = min(trial_dt_s, remaining_s)
        while True:
            before = _copy_fast_source_state(state)
            trial = _copy_fast_source_state(state)
            if trial['hydrogen_update_mu']:
                trial['mu'] = rh.mean_molecular_weight_mu(
                    trial['xHI'],
                    hydrogen_mass_fraction=trial['hydrogen_mass_fraction'],
                )
            if trial['thermal_coupling']:
                _fast_update_temperature_from_energy(trial)
            thermal_rate_value = None
            if trial['thermal_coupling']:
                thermal_rate_value = thermal_rate(
                    trial, trial.get('ngamma_cgs_cm3')
                )
                _fast_apply_thermal_source(
                    trial, thermal_rate_value, candidate_dt_s
                )
            if (
                trial['recombination']
                or trial['collisional_ionization']
            ):
                ionization_fraction_implicit_update(
                    trial, trial.get('ngamma_cgs_cm3'), candidate_dt_s
                )
            if trial['hydrogen_update_mu']:
                trial['mu'] = rh.mean_molecular_weight_mu(
                    trial['xHI'],
                    hydrogen_mass_fraction=trial['hydrogen_mass_fraction'],
                )
            if trial['thermal_coupling']:
                _fast_update_temperature_from_energy(trial)

            old_energy = np.asarray(
                before.get('specific_energy_cgs_erg_g', before['specific_total_energy_cgs_erg_g']),
                dtype=float,
            )
            new_energy = np.asarray(
                trial.get('specific_energy_cgs_erg_g', trial['specific_total_energy_cgs_erg_g']),
                dtype=float,
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                relative_energy_change = np.abs(new_energy - old_energy) / np.maximum(
                    np.abs(old_energy), 1.0e-30
                )
            max_energy_change = float(
                np.max(relative_energy_change[active]) if np.any(active) else 0.0
            )
            if max_energy_change <= 0.1:
                _set_fast_source_state(state, trial)
                check_source_temperature(
                    state, par, before['temperature_cgs_K'],
                    stage='hydrogen split-implicit source',
                    source_step=source_steps + 1,
                )
                remaining_s -= candidate_dt_s
                source_steps += 1
                if source_steps > max_subcycles:
                    raise RuntimeError(
                        'split-implicit hydrogen source update exceeded '
                        f'{max_subcycles} subcycles'
                    )
                trial_dt_s = min(2.0 * candidate_dt_s, remaining_s)
                break

            candidate_dt_s *= 0.5
            if candidate_dt_s <= 0.0 or (
                dtmin_s > 0.0 and candidate_dt_s < dtmin_s
            ):
                raise RuntimeError(
                    'split-implicit hydrogen source update cannot satisfy '
                    'the 10% internal-energy change limit'
                )
        # The next interval starts from the accepted state and may grow again.
    return source_steps


def _source_relative_change(before, after):
    """Return the largest relative source-state change over active cells."""
    active = np.asarray(
        before.get('active', np.asarray(before['rho_cgs_g_cm3']) > 0.0),
        dtype=bool,
    )
    changes = []
    for key, floor in (
        ('temperature_cgs_K', 1.0),
        ('specific_energy_cgs_erg_g', 1.0e-30),
        ('xHI', 1.0e-12),
    ):
        if key not in before or key not in after:
            continue
        old = np.asarray(before[key], dtype=float)
        new = np.asarray(after[key], dtype=float)
        if np.any(active):
            denominator = np.maximum(np.abs(old[active]), floor)
            changes.append(np.max(np.abs(new[active] - old[active]) / denominator))
    return float(max(changes, default=0.0))


def _implicit_state_difference(coarse, fine):
    """Return the normalized difference between two implicit source states."""
    differences = []
    for key, floor in (
        ('specific_energy_cgs_erg_g', 1.0e-30),
        ('temperature_cgs_K', 1.0),
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


def _adaptive_coupled_implicit_source_update_group(
    state,
    dt_s,
    ngamma_cgs_cm3=None,
    tolerance=1.0e-6,
    convergence_tolerance=None,
    max_iterations=32,
    max_refinements=4,
    trust_region=False,
    absolute_temperature_tolerance=0.0,
    absolute_xhi_tolerance=0.0,
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
    last_failure = None
    while remaining_s > zero_time_s:
        accepted = False
        candidate_dt_s = min(trial_dt_s, remaining_s)
        for refinement in range(int(max_refinements) + 1):
            coarse = _copy_fast_source_state(state)
            fine = _copy_fast_source_state(state)
            coarse_ok = _coupled_implicit_source_update(
                coarse,
                candidate_dt_s,
                ngamma_cgs_cm3=ngamma_cgs_cm3,
                tolerance=tolerance,
                max_iterations=max_iterations,
                trust_region=trust_region,
                absolute_temperature_tolerance=absolute_temperature_tolerance,
                absolute_xhi_tolerance=absolute_xhi_tolerance,
            )
            if not coarse_ok:
                last_failure = dict(coarse.get('_implicit_failure', {}))
            half_dt_s = 0.5 * candidate_dt_s
            fine_ok = _coupled_implicit_source_update(
                fine,
                half_dt_s,
                ngamma_cgs_cm3=ngamma_cgs_cm3,
                tolerance=tolerance,
                max_iterations=max_iterations,
                trust_region=trust_region,
                absolute_temperature_tolerance=absolute_temperature_tolerance,
                absolute_xhi_tolerance=absolute_xhi_tolerance,
            )
            if not fine_ok:
                last_failure = dict(fine.get('_implicit_failure', {}))
            if fine_ok:
                fine_ok = _coupled_implicit_source_update(
                    fine,
                    half_dt_s,
                    ngamma_cgs_cm3=ngamma_cgs_cm3,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    trust_region=trust_region,
                    absolute_temperature_tolerance=absolute_temperature_tolerance,
                    absolute_xhi_tolerance=absolute_xhi_tolerance,
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
                # Recover from a short interval used to resolve a stiff
                # transient.  Keeping the first accepted size for the whole
                # hydro step can turn one refinement into millions of source
                # solves even after the state has relaxed.
                trial_dt_s = min(2.0 * candidate_dt_s, remaining_s)
                accepted = True
                break
            candidate_dt_s *= 0.5

        if not accepted:
            if last_failure is not None:
                state['_implicit_failure'] = last_failure
            return False, total_source_steps
    return True, total_source_steps


def _source_stiffness_groups(state, dt_s, ngamma_cgs_cm3=None):
    """Group active cells by their predicted local source change.

    Thermochemistry is cell-local.  Grouping cells with comparable source
    timescales prevents one cold/recombining outlier from forcing every cell
    through its adaptive substeps.
    """
    active = np.asarray(state['active'], dtype=bool)
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return []
    if not state.get('thermal_coupling', False):
        return [indices]
    rho = np.asarray(state['rho_cgs_g_cm3'], dtype=float)
    energy = np.maximum(
        np.abs(np.asarray(state['specific_energy_cgs_erg_g'], dtype=float)),
        1.0e-30,
    )
    xhi = np.asarray(state['xHI'], dtype=float)
    thermal = np.asarray(thermal_rate(state, ngamma_cgs_cm3), dtype=float)
    chemistry = np.asarray(ionization_fraction_rate(state, ngamma_cgs_cm3), dtype=float)
    dt_value = float(np.asarray(dt_s, dtype=float))
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        energy_change = dt_value * np.abs(thermal) / np.maximum(rho * energy, 1.0e-99)
        chemistry_scale = np.maximum(np.minimum(xhi, 1.0 - xhi), 1.0e-8)
        chemistry_change = dt_value * np.abs(chemistry) / chemistry_scale
    stiffness = np.maximum(energy_change, chemistry_change)
    stiffness = np.nan_to_num(stiffness, nan=np.inf, posinf=np.inf, neginf=np.inf)
    # Four powers of two per bin keeps the number of vector solves small while
    # separating genuinely stiff outliers by at least a factor of sixteen.
    finite_log = np.where(
        np.isfinite(stiffness),
        np.log2(np.maximum(stiffness, 1.0)),
        1024.0,
    )
    bins = np.floor(finite_log / 4.0).astype(int)
    return [indices[bins[indices] == value] for value in np.unique(bins[indices])]


def _slice_source_cells(state, indices):
    """Return a source state containing only ``indices`` cell arrays."""
    count = len(np.asarray(state['rho_cgs_g_cm3']))
    subset = copy.deepcopy(state)
    for key, value in list(subset.items()):
        if not isinstance(value, np.ndarray) or value.ndim == 0:
            continue
        if value.shape[0] == count:
            subset[key] = value[indices].copy()
        elif value.shape[-1] == count:
            subset[key] = value[..., indices].copy()
    return subset


def _slice_source_photons(ngamma_cgs_cm3, indices, count):
    if ngamma_cgs_cm3 is None:
        return None
    photons = np.asarray(ngamma_cgs_cm3)
    if photons.ndim > 0 and photons.shape[-1] == count:
        return photons[..., indices].copy()
    return copy.deepcopy(ngamma_cgs_cm3)


def _merge_source_cells(state, subset, indices):
    for key in (
        'specific_energy_cgs_erg_g',
        'specific_total_energy_cgs_erg_g',
        'temperature_cgs_K',
        'xHI',
        'mu',
    ):
        if key not in state or key not in subset:
            continue
        target = np.asarray(state[key]).copy()
        target[indices] = np.asarray(subset[key])
        state[key] = target


def _adaptive_coupled_implicit_source_update(
    state,
    dt_s,
    ngamma_cgs_cm3=None,
    tolerance=1.0e-6,
    convergence_tolerance=None,
    max_iterations=32,
    max_refinements=4,
    trust_region=False,
    absolute_temperature_tolerance=0.0,
    absolute_xhi_tolerance=0.0,
):
    """Advance independent stiffness groups without global subcycling."""
    cell_count = len(np.asarray(state['rho_cgs_g_cm3']))
    total_source_steps = 0
    for indices in _source_stiffness_groups(state, dt_s, ngamma_cgs_cm3):
        subset = _slice_source_cells(state, indices)
        subset_ngamma = _slice_source_photons(ngamma_cgs_cm3, indices, cell_count)
        solved, source_steps = _adaptive_coupled_implicit_source_update_group(
            subset,
            dt_s,
            ngamma_cgs_cm3=subset_ngamma,
            tolerance=tolerance,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
            max_refinements=max_refinements,
            trust_region=trust_region,
            absolute_temperature_tolerance=absolute_temperature_tolerance,
            absolute_xhi_tolerance=absolute_xhi_tolerance,
        )
        total_source_steps += source_steps
        if not solved:
            failure = dict(subset.get('_implicit_failure', {}))
            local_cell = failure.get('cell')
            if local_cell is not None:
                failure['cell'] = int(indices[int(local_cell)])
            for key in ('failed_cells', 'unconverged_cells'):
                for item in failure.get(key, []):
                    item['cell'] = int(indices[int(item['cell'])])
            state['_implicit_failure'] = failure
            return False, total_source_steps
        _merge_source_cells(state, subset, indices)
    _fast_update_temperature_from_energy(state)
    return True, total_source_steps


def _fast_sync_state_to_fluid(state, fluid, par):
    """Copy a float thermo-chemistry state back to the fluid container."""
    if state.get('thermal_coupling', False):
        _fast_update_temperature_from_energy(state)
    interior = state['interior']
    active = np.asarray(
        state.get('active', np.asarray(state['rho_cgs_g_cm3']) > 0.0),
        dtype=bool,
    )
    xhi = np.asarray(fluid.xHI[interior], dtype=float).copy()
    xhi[active] = state['xHI'][active]
    fluid.xHI[interior] = xhi
    if hasattr(fluid, 'ngamma_code') and state.get('ngamma_cgs_cm3') is not None:
        code = _code_units(par)
        target = from_unit_value(state['ngamma_cgs_cm3'], code.number_density_unit)
        if np.ndim(target) == 2:
            fluid.ngamma_code[:, interior] = target
        else:
            fluid.ngamma_code[interior] = target
    if hasattr(fluid, 'mu'):
        mu = np.asarray(fluid.mu[interior], dtype=float).copy()
        mu[active] = state['mu'][active]
        fluid.mu[interior] = mu
    code = _code_units(par)
    temperature = (
        state['temperature_cgs_K'] * state.get('source_temperature_factor', 1.0)
        / code.unit_conversion['temperature_cgs_K']
    )
    temp_code = np.asarray(fluid.temp_code[interior], dtype=float).copy()
    temp_code[active] = temperature[active]
    fluid.temp_code[interior] = temp_code
    if state.get('thermal_coupling', False):
        # The source state stores specific energies in physical cgs units
        # (erg/g), while Fluid pressure and Energy use the code velocity
        # unit.  Convert the specific energy terms before writing them back;
        # omitting this conversion injects velocity_unit_cgs**2 into the
        # conserved energy (1e10 for the standard 1 km/s code unit).
        specific_energy_code_factor = float(
            code.unit_conversion['velocity_cgs_cm_s']
        ) ** 2
        specific_internal_energy_physical = (
            state['specific_total_energy_cgs_erg_g']
            - state['specific_kinetic_energy_cgs_erg_g']
        )
        specific_internal_energy = (
            specific_internal_energy_physical
            * state.get('source_temperature_factor', 1.0)
        )
        specific_internal_energy_code = (
            specific_internal_energy / specific_energy_code_factor
        )
        specific_kinetic_energy_code = (
            state.get('specific_kinetic_energy_supercomoving_cgs_erg_g', 0.0)
            / specific_energy_code_factor
        )
        rotational_specific_code = np.asarray(
            state.get('specific_rotational_energy_code',
                      np.zeros_like(specific_internal_energy_code)),
            dtype=float,
        )
        specific_total_energy = (
            specific_internal_energy_code + specific_kinetic_energy_code
            + rotational_specific_code
        )
        pressure = (
            specific_internal_energy_code
            * np.asarray(fluid.rho_code[interior], dtype=float)
            * (fluid.eos.gamma - 1.0)
        )
        pre = np.asarray(fluid.pre_code[interior], dtype=float).copy()
        pre[active] = pressure[active]
        fluid.pre_code[interior] = pre
        energy = specific_total_energy * np.asarray(fluid.Mass_code[interior], dtype=float)
        conserved_energy = np.asarray(fluid.Energy_code[interior], dtype=float).copy()
        conserved_energy[active] = energy[active]
        fluid.Energy_code[interior] = conserved_energy
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
        raise ValueError("hydrogen thermo-chemistry requires configured code units")
    remaining_s = (
        to_unit_value(dt, code.time_unit)
        * state['source_scale_factor']**2
    )
    total_dt_s = remaining_s
    zero_time_s = 0.0
    source_steps = 0
    absorbed_integral = None
    source_solver = str(
        getattr(par, 'hydrogen_source_solver', 'hybrid')
    ).lower()
    if source_solver not in (
        'explicit', 'coupled_implicit', 'hybrid', 'trust_region',
        'split_implicit',
    ):
        raise ValueError(
            "hydrogen_source_solver must be 'explicit', 'coupled_implicit', "
            "'hybrid', 'trust_region', or 'split_implicit'"
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
        and state.get('ngamma_cgs_cm3') is None
    )
    if compton_only:
        _apply_compton_only_source(state, remaining_s)
        _fast_sync_state_to_fluid(state, fluid, par)
        return {
            'source_steps': 1,
            'absorbed_photon_rate': None,
            'photon_energy_cgs_erg': np.atleast_1d(
                _optional_numeric_value(
                    getattr(par, 'ionizing_photon_energy_cgs_erg',
                            getattr(par, 'hydrogen_photon_energy', 0.0)),
                    code.energy_unit,
                    default=0.0,
                )
            ),
            'direction': int(getattr(par, 'radiative_transfer_direction', 1)),
        }
    initial_state = _copy_fast_source_state(state)
    if source_solver == 'split_implicit' and remaining_s > zero_time_s:
        split_source_steps = _split_implicit_source_state_update(
            state, remaining_s, par
        )
        change = _source_relative_change(initial_state, state)
        _fast_sync_state_to_fluid(state, fluid, par)
        return {
            'source_steps': split_source_steps,
            'source_solver': 'split_implicit',
            'relative_change': change,
            'absorbed_photon_rate': None,
            'photon_energy_cgs_erg': np.atleast_1d(
                _optional_numeric_value(
                    getattr(par, 'ionizing_photon_energy_cgs_erg',
                            getattr(par, 'hydrogen_photon_energy', 0.0)),
                    code.energy_unit,
                    default=0.0,
                )
            ),
            'direction': int(getattr(par, 'radiative_transfer_direction', 1)),
        }
    if source_solver == 'hybrid' and getattr(
        par, 'hydrogen_hybrid_explicit_probe', False
    ) and remaining_s > zero_time_s:
        initial_state = _copy_fast_source_state(state)
        explicit_state = _copy_fast_source_state(state)
        explicit_steps = _explicit_source_state_update(
            explicit_state,
            remaining_s,
            par,
        )
        change = _source_relative_change(initial_state, explicit_state)
        threshold = float(
            getattr(par, 'hydrogen_hybrid_change_tolerance', 0.1)
        )
        if (
            change <= threshold
            or getattr(par, 'radiative_transfer', False)
            or getattr(par, 'hydrogen_radiation_field', False)
        ):
            _set_fast_source_state(state, explicit_state)
            _fast_sync_state_to_fluid(state, fluid, par)
            return {
                'source_steps': explicit_steps,
                'source_solver': 'explicit',
                'relative_change': change,
                'absorbed_photon_rate': None,
                'photon_energy_cgs_erg': np.atleast_1d(
                    _optional_numeric_value(
                        getattr(par, 'ionizing_photon_energy_cgs_erg',
                                getattr(par, 'hydrogen_photon_energy', 0.0)),
                        code.energy_unit,
                        default=0.0,
                    )
                ),
                'direction': int(getattr(par, 'radiative_transfer_direction', 1)),
            }
        state = initial_state
        can_solve_coupled = not getattr(par, 'radiative_transfer', False)
        solved = False
        if can_solve_coupled:
            solved, implicit_source_steps = _adaptive_coupled_implicit_source_update(
                state,
                remaining_s,
                ngamma_cgs_cm3=state.get('ngamma_cgs_cm3'),
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
                trust_region=(source_solver == 'trust_region'),
                absolute_temperature_tolerance=float(
                    getattr(par, 'hydrogen_implicit_absolute_temperature_tolerance', 0.0)
                ),
                absolute_xhi_tolerance=float(
                    getattr(par, 'hydrogen_implicit_absolute_xhi_tolerance', 0.0)
                ),
            )
        if solved:
            change = _source_relative_change(initial_state, state)
            _fast_sync_state_to_fluid(state, fluid, par)
            return {
                'source_steps': implicit_source_steps,
                'source_solver': (
                    'trust_region'
                    if source_solver == 'trust_region'
                    else 'coupled_implicit'
                ),
                'relative_change': change,
                'absorbed_photon_rate': None,
                'photon_energy_cgs_erg': np.atleast_1d(
                    _optional_numeric_value(
                        getattr(par, 'ionizing_photon_energy_cgs_erg',
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
        if fallback == 'error':
            raise RuntimeError(
                'hybrid hydrogen source implicit solve did not converge '
                f'(explicit relative change={change:.6g})'
            )
        _set_fast_source_state(state, explicit_state)
        _fast_sync_state_to_fluid(state, fluid, par)
        return {
            'source_steps': explicit_steps,
            'source_solver': 'explicit_fallback',
            'relative_change': change,
            'absorbed_photon_rate': None,
            'photon_energy_cgs_erg': np.atleast_1d(
                _optional_numeric_value(
                    getattr(par, 'ionizing_photon_energy_cgs_erg',
                            getattr(par, 'hydrogen_photon_energy', 0.0)),
                    code.energy_unit,
                    default=0.0,
                )
            ),
            'direction': int(getattr(par, 'radiative_transfer_direction', 1)),
        }
    if source_solver in ('hybrid', 'coupled_implicit', 'trust_region') and remaining_s > zero_time_s:
        # A ray-traced photon field can change during the source step.  Keep
        # that operator split on the established path; the coupled solver is
        # for a local, fixed photon field (including no photon field).
        can_solve_coupled = not getattr(par, 'radiative_transfer', False)
        solved = False
        if can_solve_coupled:
            solved, implicit_source_steps = _adaptive_coupled_implicit_source_update(
                state,
                remaining_s,
                ngamma_cgs_cm3=state.get('ngamma_cgs_cm3'),
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
                trust_region=(source_solver == 'trust_region'),
                absolute_temperature_tolerance=float(
                    getattr(par, 'hydrogen_implicit_absolute_temperature_tolerance', 0.0)
                ),
                absolute_xhi_tolerance=float(
                    getattr(par, 'hydrogen_implicit_absolute_xhi_tolerance', 0.0)
                ),
            )
        if solved:
            change = _source_relative_change(initial_state, state)
            _fast_sync_state_to_fluid(state, fluid, par)
            return {
                'source_steps': implicit_source_steps,
                'source_solver': (
                    'trust_region'
                    if source_solver == 'trust_region'
                    else 'coupled_implicit'
                ),
                'relative_change': change,
                'absorbed_photon_rate': None,
                'photon_energy_cgs_erg': np.atleast_1d(
                    _optional_numeric_value(
                        getattr(par, 'ionizing_photon_energy_cgs_erg',
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
        failure = state.get('_implicit_failure', {})
        if getattr(par, 'hydrogen_implicit_debug', False) or failure:
            print('[hydrogen implicit failure]', failure)
            for failed_cell in failure.get('failed_cells', []):
                print('[hydrogen implicit singular cell]', failed_cell)
            for unconverged_cell in failure.get('unconverged_cells', []):
                print('[hydrogen implicit unconverged cell]', unconverged_cell)
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
                        boundary=state['boundary_cgs_cm'],
                        vol=state['volume_cgs_cm3'],
                        coordsys=getattr(par, 'coordsys', 'cartesian'),
                    ),
                    rho=state['rho_cgs_g_cm3'],
                    xHI=state['xHI'],
                    hydrogen_mass_fraction=state['hydrogen_mass_fraction'],
                    # ``sigma_gamma_cgs_cm2`` is already expressed in cgs cm^2.
                    sigma_gamma=np.asarray(state['sigma_gamma_cgs_cm2'], dtype=float),
                    boundary_flux=boundary_flux,
                    # ``source_rate_s`` is already in cgs s^-1.  It was
                    # previously multiplied by the inverse code time unit a
                    # second time, suppressing the transported photon field.
                    source_photon_rate=np.asarray(state['source_rate_s'], dtype=float),
                    direction=getattr(par, 'radiative_transfer_direction', 1),
                    group_edges_eV=getattr(par, 'radiation_group_edges_eV', None),
                )
            state['ngamma_cgs_cm3'] = transport.cell_photon_density
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
        temperature_before = np.asarray(
            state['temperature_cgs_K'], dtype=float
        ).copy()
        sub_dt_s, thermal_rate = get_timestep(
            state,
            state.get('ngamma_cgs_cm3'),
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
                state.get('ngamma_cgs_cm3'),
                sub_dt_s,
            )
        if state['hydrogen_update_mu']:
            state['mu'] = rh.mean_molecular_weight_mu(
                state['xHI'],
                hydrogen_mass_fraction=state['hydrogen_mass_fraction'],
            )
        if state['thermal_coupling']:
            _fast_update_temperature_from_energy(state)
        check_source_temperature(
            state, par, temperature_before,
            stage='hydrogen source', source_step=source_steps + 1,
        )
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
    energy = getattr(par, 'ionizing_photon_energy_cgs_erg', None)
    if energy is None:
        energy = getattr(par, 'hydrogen_photon_energy', 0.0)
    if hasattr(energy, 'to_value'):
        energy = np.asarray(energy.to_value('erg'), dtype=float)
    return {
        'source_steps': source_steps,
        'absorbed_photon_rate': absorbed_rate,
        'photon_energy_cgs_erg': np.atleast_1d(energy),
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

    def ionization_fraction_rate(self, state, ngamma_cgs_cm3):
        return ionization_fraction_rate(state, ngamma_cgs_cm3)

    def thermal_rate(self, state, ngamma_cgs_cm3):
        return thermal_rate(state, ngamma_cgs_cm3)

    def get_timestep(self, state, ngamma_cgs_cm3, remaining_s, dtmax_s):
        return get_timestep(
            state,
            ngamma_cgs_cm3,
            remaining_s,
            dtmax_s,
        )

    def update_temperature_from_energy(self, state):
        return update_temperature_from_energy(state)

    def ionization_fraction_implicit_update(self, state, ngamma_cgs_cm3, dt_s):
        return ionization_fraction_implicit_update(state, ngamma_cgs_cm3, dt_s)

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
