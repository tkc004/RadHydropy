"""Coupled hydrogen/helium thermo-chemistry network.

The network evolves H I, He I, and He III; H II and He II are constrained by
element conservation.  It uses the shared multigroup radiation field and a
local implicit Euler/fixed-point substep with an explicit small-change path.
"""

import numpy as np

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS, SPEED_OF_LIGHT_CGS
import radhydropy.radiative_transfer as rrt
from radhydropy.units import _code_units, from_unit_value, to_unit_value
from radhydropy.thermo_networks.base import ThermochemistryNetwork
from radhydropy.thermo_networks.hydrogen import (
    _cgs_alpha_B, _cgs_beta, _cgs_gamma_B_eHII, _cgs_gamma_ff_eHII,
    _cgs_gamma_ion_eHI, _cgs_gamma_line_eHI,
    _fast_source_scaling,
)
from radhydropy.thermo_networks.compton import cmb_compton_rate
from radhydropy.diagnostics import thermochemistry_active_mask


def _alpha_heii(T):
    """He II radiative recombination, Hummer & Storey (1998)."""
    T = np.maximum(np.asarray(T, float), 1.0)
    return 1.26e-14 * (570670.0 / T) ** 0.750


def _alpha_heii_dielectronic(T):
    """He II dielectronic recombination, Aldrovandi & Pequignot (1973)."""
    T = np.maximum(np.asarray(T, float), 1.0)
    return (
        1.9e-3
        * T ** -1.5
        * np.exp(-4.7e5 / T)
        * (1.0 + 0.3 * np.exp(-9.4e4 / T))
    )


def _alpha_heiii(T):
    """He III case-B recombination, Hui & Gnedin (1997)."""
    T = np.maximum(np.asarray(T, float), 1.0)
    lam = 1263030.0 / T
    return 5.506e-14 * lam**1.5 * (1.0 + (460960.0 / T) ** 0.407) ** -2.242


def _beta_hei(T):
    """He I collisional ionization, Theuns et al. (1998)."""
    T = np.maximum(np.asarray(T, float), 1.0)
    return 4.76e-11 * np.sqrt(T) * np.exp(-285335.4 / T) / (1.0 + np.sqrt(T / 1.0e5))


def _beta_heii(T):
    """He II collisional ionization, Theuns et al. (1998)."""
    T = np.maximum(np.asarray(T, float), 1.0)
    return 1.14e-11 * np.sqrt(T) * np.exp(-631515.0 / T) / (1.0 + np.sqrt(T / 1.0e5))


def _gamma_ion_hei(T):
    T = np.maximum(np.asarray(T, float), 1.0)
    return 1.88e-21 * np.sqrt(T) * np.exp(-285335.4 / T) / (1.0 + np.sqrt(T / 1.0e5))


def _gamma_ion_heii(T):
    T = np.maximum(np.asarray(T, float), 1.0)
    return 9.90e-22 * np.sqrt(T) * np.exp(-631515.0 / T) / (1.0 + np.sqrt(T / 1.0e5))


def _gamma_line_hei(T):
    T = np.maximum(np.asarray(T, float), 1.0)
    return 9.10e-27 * T ** -0.1687 * np.exp(-13179.0 / T) / (1.0 + np.sqrt(T / 1.0e5))


def _gamma_line_heii(T):
    T = np.maximum(np.asarray(T, float), 1.0)
    return 5.54e-17 * T ** -0.397 * np.exp(-473638.0 / T) / (1.0 + np.sqrt(T / 1.0e5))


def _gamma_rec_heii(T, case="B"):
    """He II recombination cooling, using k_B T alpha (Hummer & Storey)."""
    alpha = _alpha_heii(T)
    if case.upper() == "A":
        alpha = 1.26e-14 * (570670.0 / np.maximum(np.asarray(T, float), 1.0)) ** 0.750
    return BOLTZMANN_CONSTANT_CGS * np.asarray(T, float) * alpha


def _gamma_rec_heiii(T, case="B"):
    """He III recombination cooling, Hui & Gnedin (1997)."""
    T = np.maximum(np.asarray(T, float), 1.0)
    lam = 1263030.0 / T
    if case.upper() == "A":
        return 1.4224e-28 * T * lam**1.965 * (1.0 + (lam / 0.522) ** 0.470) ** -1.923
    return 2.748e-29 * T * lam**1.970 * (1.0 + (lam / 2.250) ** 0.376) ** -3.720


def _gamma_dielectronic_heii(T):
    T = np.maximum(np.asarray(T, float), 1.0)
    return 1.24e-13 * T ** -1.5 * np.exp(-4.7e5 / T) * (1.0 + 0.3 * np.exp(-9.4e4 / T))


def _gamma_bremsstrahlung(T):
    T = np.maximum(np.asarray(T, float), 1.0)
    return 1.42e-27 * np.sqrt(T) * (
        1.1 + 0.34 * np.exp(-(5.5 - np.log10(T)) ** 2 / 3.0)
    )


def _state_density(state):
    rho = state['rho_g_cm3']
    nH = state['hydrogen_mass_fraction'] * rho / PROTON_MASS_CGS
    nHe = state['helium_mass_fraction'] * rho / (4.0 * PROTON_MASS_CGS)
    return nH, nHe


def _closure(state):
    nH, nHe = _state_density(state)
    xHI, xHeI, xHeIII = state['xHI'], state['xHeI'], state['xHeIII']
    xHII = 1.0 - xHI
    xHeII = np.clip(1.0 - xHeI - xHeIII, 0.0, 1.0)
    ne = nH * xHII + nHe * (xHeII + 2.0 * xHeIII)
    nt = nH + nHe + ne
    state['xHII'], state['xHeII'] = xHII, xHeII
    state['ne_cm3'] = ne
    state['mu'] = state['rho_g_cm3'] / np.maximum(PROTON_MASS_CGS * nt, 1.0e-99)


def _rates(state, ngamma):
    _closure(state)
    nH, nHe = _state_density(state)
    T = state['temperature_K']; ne = state['ne_cm3']
    sigma = state['sigma_gamma_cm2']; eps = state['epsilon_gamma_erg']
    photo = rrt.species_photoionization_rates(ngamma, sigma)
    photo_heat = rrt.species_photoionization_heating(ngamma, sigma, eps)
    xHI, xHII = state['xHI'], state['xHII']
    xHeI, xHeII, xHeIII = state['xHeI'], state['xHeII'], state['xHeIII']
    aH, bH = _cgs_alpha_B(T), _cgs_beta(T)
    a2 = _alpha_heii(T) + _alpha_heii_dielectronic(T)
    a3 = _alpha_heiii(T)
    b1, b2 = _beta_hei(T), _beta_heii(T)
    dHI = ne * aH * xHII - ne * bH * xHI - photo['HI'] * xHI
    dHeI = ne * a2 * xHeII - ne * b1 * xHeI - photo['HeI'] * xHeI
    dHeIII = ne * b2 * xHeII - ne * a3 * xHeIII + photo['HeII'] * xHeII
    heating = nH * xHI * photo_heat['HI'] + nHe * xHeI * photo_heat['HeI'] + nHe * xHeII * photo_heat['HeII']
    cooling = ne * nH * (
        xHI * (_cgs_gamma_line_eHI(T) + _cgs_gamma_ion_eHI(T))
        + xHII * (_cgs_gamma_ff_eHII(T) + _cgs_gamma_B_eHII(T))
    )
    cooling += ne * nHe * (
        xHeI * (_gamma_line_hei(T) + _gamma_ion_hei(T))
        + xHeII * (
            _gamma_line_heii(T)
            + _gamma_ion_heii(T)
            + _gamma_rec_heii(T)
            + _gamma_dielectronic_heii(T)
        )
        + xHeIII * _gamma_rec_heiii(T)
    )
    cooling += ne * _gamma_bremsstrahlung(T) * (
        nH * xHII + nHe * (xHeII + 4.0 * xHeIII)
    )
    metal_heating = np.zeros_like(T, dtype=float)
    metal_cooling = np.zeros_like(T, dtype=float)
    metal_table = state.get('metal_pie_table')
    if metal_table is not None:
        if getattr(metal_table, 'is_hm12_uv_background', False):
            metal_heating, metal_cooling = metal_table.rates(
                T,
                nH,
                metallicity=state.get('metallicity', 1.0),
                redshift=state.get('metal_pie_redshift', 0.0),
            )
        else:
            ngamma_total = np.sum(np.asarray(ngamma, dtype=float), axis=0)
            ionization_parameter = ngamma_total / np.maximum(nH, 1.0e-99)
            metal_heating, metal_cooling = metal_table.rates(
                T,
                nH,
                ionization_parameter,
                state.get('metallicity', 1.0),
            )
        max_heating_density = state.get(
            'metal_pie_photoheating_max_density_cm3', 50.0
        )
        if (
            getattr(metal_table, 'is_hm12_uv_background', False)
            and max_heating_density is not None
        ):
            metal_heating = np.where(
                nH > float(max_heating_density), 0.0, metal_heating
            )
    return dHI, dHeI, dHeIII, heating - cooling + metal_heating - metal_cooling + cmb_compton_rate(
        T,
        ne,
        enabled=state.get('compton_cmb_enabled', False),
        redshift=state.get('compton_cmb_redshift', 0.0),
        cmb_temperature_0_K=state.get('cmb_temperature_0_K', 2.7255),
    )


def source_state(mesh, fluid, par):
    code = _code_units(par); interior = slice(par.noghost, par.noghost + par.nogrid)
    gamma = getattr(par, 'gamma', 5.0 / 3.0)
    scaling = _fast_source_scaling(fluid, par, gamma)
    xHI = np.asarray(getattr(fluid, 'xHI', np.ones_like(fluid.rho[interior]))[interior], float).copy()
    xHeI = np.asarray(getattr(fluid, 'xHeI', np.ones_like(xHI))[interior] if hasattr(fluid, 'xHeI') else np.ones_like(xHI), float).copy()
    xHeII = np.asarray(getattr(fluid, 'xHeII', np.zeros_like(xHI))[interior] if hasattr(fluid, 'xHeII') else np.zeros_like(xHI), float).copy()
    xHeIII = np.clip(1.0 - xHeI - xHeII, 0.0, 1.0)
    sigma = {'HI': np.asarray(getattr(par, 'radiation_group_sigma_gamma'), float), 'HeI': np.asarray(getattr(par, 'radiation_group_sigma_gamma_HeI', getattr(par, 'radiation_group_sigma_gamma')), float), 'HeII': np.asarray(getattr(par, 'radiation_group_sigma_gamma_HeII', getattr(par, 'radiation_group_sigma_gamma')), float)}
    eps = {'HI': np.asarray(getattr(par, 'radiation_group_epsilon_gamma'), float), 'HeI': np.asarray(getattr(par, 'radiation_group_epsilon_gamma_HeI', getattr(par, 'radiation_group_epsilon_gamma')), float), 'HeII': np.asarray(getattr(par, 'radiation_group_epsilon_gamma_HeII', getattr(par, 'radiation_group_epsilon_gamma')), float)}
    rho_super = to_unit_value(fluid.rho[interior], code.density_unit)
    velocity_super = to_unit_value(fluid.vel[interior], code.velocity_unit)
    if hasattr(fluid, 'Mass'):
        mass = to_unit_value(fluid.Mass[interior], code.mass_unit)
    else:
        mass = rho_super * np.asarray(mesh.vol[interior], dtype=float) * code.mass_in_cgs
    total_super = to_unit_value(fluid.Energy[interior], code.energy_unit) / np.maximum(mass, 1.0e-99)
    specific_internal = np.maximum(total_super - 0.5 * velocity_super**2, 0.0) / scaling['temperature_factor']
    rho_physical = rho_super / scaling['density_factor']
    state = {'interior': interior, 'boundary_cm': to_unit_value(mesh.boundary[interior.start:interior.stop + 1], code.length_unit) * scaling['scale_factor'], 'volume_cm3': to_unit_value(mesh.vol[interior], code.volume_unit) * scaling['density_factor'], 'radius_kpc': to_unit_value(mesh.coordinate[interior], code.length_unit) * scaling['scale_factor'] / 3.08567758e21, 'rho_g_cm3': rho_physical, 'active': thermochemistry_active_mask(rho_physical, par, scaling['density_factor']), 'temperature_K': to_unit_value(fluid.temp[interior], code.temperature_unit) / scaling['temperature_factor'], 'specific_energy_erg_g': specific_internal, 'gamma': gamma, 'hydrogen_mass_fraction': getattr(par, 'hydrogen_mass_fraction', 0.7), 'helium_mass_fraction': getattr(par, 'helium_mass_fraction', 0.28), 'xHI': xHI, 'xHeI': xHeI, 'xHeIII': xHeIII, 'sigma_gamma_cm2': sigma, 'epsilon_gamma_erg': eps, 'thermal_coupling': getattr(par, 'hydrogen_thermal_coupling', True), 'compton_cmb_enabled': getattr(par, 'compton_cmb_enabled', False), 'compton_cmb_redshift': getattr(par, 'compton_cmb_redshift', 0.0), 'metal_pie_redshift': getattr(par, 'metal_pie_redshift', 0.0), 'cmb_temperature_0_K': float(to_unit_value(getattr(par, 'cmb_temperature_0', 2.7255), 'K')), 'explicit_tolerance': getattr(par, 'explicit_tolerance', 0.1), 'relative_tolerance': getattr(par, 'relative_tolerance', 1.0e-3), 'absolute_tolerance': getattr(par, 'absolute_tolerance', 1.0e-10), 'metal_pie_table': getattr(par, 'metal_pie_table', None), 'metallicity': getattr(par, 'metallicity', 1.0), 'metal_pie_photoheating_max_density_cm3': getattr(par, 'metal_pie_photoheating_max_density_cm3', 50.0), 'source_scale_factor': scaling['scale_factor'], 'source_temperature_factor': scaling['temperature_factor'], 'velocity_supercomoving_cm_s': velocity_super}
    state['coupled_implicit'] = getattr(par, 'hydrogen_helium_coupled_implicit', True)
    state['nH_cm3'] = state['rho_g_cm3'] * state['hydrogen_mass_fraction'] / PROTON_MASS_CGS
    _closure(state)
    state['specific_energy_erg_g'] = BOLTZMANN_CONSTANT_CGS * state['temperature_K'] / ((state['gamma'] - 1.0) * state['mu'] * PROTON_MASS_CGS)
    return state


def ionization_fraction_rate(state, ngamma):
    dHI, dHeI, dHeIII, _ = _rates(state, ngamma)
    return np.maximum(np.abs(dHI), np.maximum(np.abs(dHeI), np.abs(dHeIII)))


def thermal_rate(state, ngamma):
    return _rates(state, ngamma)[3]


def get_timestep(state, ngamma, remaining_s, dtmax_s):
    d_hi, d_hei, d_heiii, thermal = _rates(state, ngamma)
    if state.get('coupled_implicit', True):
        return min(float(remaining_s), float(dtmax_s)), thermal
    candidates = []
    for fraction, rate in (
        (state['xHI'], d_hi),
        (state['xHeI'], d_hei),
        (state['xHeIII'], d_heiii),
    ):
        scale = np.where(rate < 0.0, fraction, 1.0 - fraction)
        valid = (np.abs(rate) > 0.0) & (scale > 0.0)
        if np.any(valid):
            candidates.append(np.min(scale[valid] / np.abs(rate[valid])))
    if candidates:
        active = np.asarray(
            state.get('active', np.asarray(state['rho_g_cm3']) > 0.0),
            dtype=bool,
        )
        rho = np.where(active, state['rho_g_cm3'], 1.0)
        candidates.append(
            np.min(
                np.maximum(state['specific_energy_erg_g'], 1.0e-30)
                / np.maximum(
                    np.abs(thermal / rho),
                    1.0e-99,
                )
            )
        )
        chem = min(candidates)
    else:
        chem = float(remaining_s)
    return min(float(remaining_s), float(dtmax_s), 0.1 * chem), thermal


def update_temperature_from_energy(state):
    _closure(state)
    state['temperature_K'] = np.maximum((state['gamma'] - 1.0) * state['mu'] * PROTON_MASS_CGS * state['specific_energy_erg_g'] / 1.380649e-16, 1.0)


def ionization_fraction_implicit_update(state, ngamma, dt_s):
    old = np.array([state['xHI'], state['xHeI'], state['xHeIII']])
    trial = old.copy()
    for _ in range(12):
        state['xHI'], state['xHeI'], state['xHeIII'] = trial
        d1, d2, d3, _ = _rates(state, ngamma)
        new = old + dt_s * np.array([d1, d2, d3])
        new[0] = np.clip(new[0], 1.0e-12, 1.0 - 1.0e-12)
        new[1] = np.clip(new[1], 1.0e-12, 1.0 - 1.0e-12)
        new[2] = np.clip(new[2], 1.0e-12, 1.0 - new[1] - 1.0e-12)
        trial = 0.5 * trial + 0.5 * new
    state['xHI'], state['xHeI'], state['xHeIII'] = trial
    _closure(state)


def coupled_implicit_update(state, ngamma, dt_s):
    """Implicitly update ion fractions and thermal energy together."""
    active = np.asarray(
        state.get('active', np.asarray(state['rho_g_cm3']) > 0.0),
        dtype=bool,
    )
    rho = np.where(active, state['rho_g_cm3'], 1.0)
    old_x = np.array([state['xHI'], state['xHeI'], state['xHeIII']])
    old_energy = np.asarray(state['specific_energy_erg_g'], dtype=float).copy()
    trial_x = old_x.copy()
    trial_energy = old_energy.copy()
    for _ in range(32):
        state['xHI'], state['xHeI'], state['xHeIII'] = trial_x
        state['specific_energy_erg_g'] = trial_energy
        update_temperature_from_energy(state)
        d_hi, d_hei, d_heiii, thermal = _rates(state, ngamma)
        new_x = old_x + dt_s * np.array([d_hi, d_hei, d_heiii])
        new_x[0] = np.clip(new_x[0], 1.0e-12, 1.0 - 1.0e-12)
        new_x[1] = np.clip(new_x[1], 1.0e-12, 1.0 - 1.0e-12)
        new_x[2] = np.clip(new_x[2], 0.0, 1.0 - new_x[1] - 1.0e-12)
        new_energy = np.maximum(
            old_energy + dt_s * thermal / rho,
            1.0e6,
        )
        trial_x = 0.5 * trial_x + 0.5 * new_x
        trial_energy = 0.5 * trial_energy + 0.5 * new_energy
    state['xHI'] = np.where(active, trial_x[0], state['xHI'])
    state['xHeI'] = np.where(active, trial_x[1], state['xHeI'])
    state['xHeIII'] = np.where(active, trial_x[2], state['xHeIII'])
    state['specific_energy_erg_g'] = np.where(
        active, trial_energy, state['specific_energy_erg_g']
    )
    update_temperature_from_energy(state)


def apply_state(state, fluid, par):
    i = state['interior']; code = _code_units(par)
    fluid.xHI[i] = state['xHI']
    fluid.xHeI[i] = state['xHeI']; fluid.xHeII[i] = state['xHeII']; fluid.xHeIII[i] = state['xHeIII']
    temperature_factor = state.get('source_temperature_factor', 1.0)
    fluid.temp[i] = from_unit_value(
        state['temperature_K'] * temperature_factor,
        code.temperature_unit,
    )
    fluid.mu[i] = state['mu']
    if hasattr(fluid, 'ngamma') and state.get('ngamma_cm3') is not None:
        target = from_unit_value(state['ngamma_cm3'], code.number_density_unit)
        if np.ndim(target) == 2:
            fluid.ngamma[:, i] = target
        else:
            fluid.ngamma[i] = target
    if hasattr(fluid, 'Mass') and hasattr(fluid, 'Energy'):
        internal_super = state['specific_energy_erg_g'] * temperature_factor
        total_super = internal_super + 0.5 * state.get('velocity_supercomoving_cm_s', 0.0)**2
        specific_code = from_unit_value(total_super, code.specific_energy_unit)
        fluid.Energy[i] = fluid.Mass[i] * specific_code
        if hasattr(fluid, 'pre'):
            fluid.pre[i] = fluid.eos.pressure(fluid.rho[i], fluid.temp[i], fluid.mu[i])
        if hasattr(fluid, 'eth') and hasattr(fluid, 'pre'):
            fluid.eth[i] = fluid.eos.thermal_energy_density(fluid.pre[i])


class HydrogenHeliumNetwork(ThermochemistryNetwork):
    name = 'hydrogen_helium'
    scalar_fields = ('xHI', 'xHeI', 'xHeII', 'xHeIII')
    def enabled(self, fluid, par): return bool(getattr(par, 'hydrogen_chemistry', True))
    def radiation_enabled(self, fluid, par): return bool(getattr(par, 'radiative_transfer', False))
    def radiation_evolution_enabled(self, fluid, par): return False
    def advect_ionization_fraction(self, dt, mesh, fluid, par, old_mass, mass_flux): return 0
    def source_state(self, mesh, fluid, par): return source_state(mesh, fluid, par)
    def ionization_fraction_rate(self, state, ngamma): return ionization_fraction_rate(state, ngamma)
    def thermal_rate(self, state, ngamma): return thermal_rate(state, ngamma)
    def get_timestep(self, state, ngamma, remaining_s, dtmax_s): return get_timestep(state, ngamma, remaining_s, dtmax_s)
    def update_temperature_from_energy(self, state): return update_temperature_from_energy(state)
    def ionization_fraction_implicit_update(self, state, ngamma, dt_s): return ionization_fraction_implicit_update(state, ngamma, dt_s)
    def coupled_implicit_update(self, state, ngamma, dt_s): return coupled_implicit_update(state, ngamma, dt_s)
    def apply_state(self, state, fluid, par): return apply_state(state, fluid, par)
    def get_source_timestep_fast(self, mesh, fluid, par, remaining):
        state = source_state(mesh, fluid, par)
        code = _code_units(par)
        physical_remaining_s = (
            to_unit_value(remaining, code.time_unit)
            * state['source_scale_factor']**2
        )
        physical_dt_s = get_timestep(
            state, None, physical_remaining_s, physical_remaining_s
        )[0]
        return from_unit_value(
            physical_dt_s / state['source_scale_factor']**2,
            code.time_unit,
        )
    def apply_fast(self, dt, mesh, fluid, par): raise NotImplementedError('hydrogen_helium uses the static local subcycle path')
