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
)


def _alpha_heii(T):
    T = np.maximum(np.asarray(T, float), 1.0)
    return 1.5e-12 * (T / 1.0e4) ** -0.635


def _alpha_heiii(T):
    T = np.maximum(np.asarray(T, float), 1.0)
    return 3.36e-10 * T ** -0.5 * (T / 1.0e3) ** -0.2 / (1.0 + (T / 1.0e6) ** 0.7)


def _beta_he(T, threshold):
    T = np.maximum(np.asarray(T, float), 1.0)
    return 5.0e-11 * np.sqrt(T) * np.exp(-threshold / T) / (1.0 + np.sqrt(T / 1.0e5))


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
    a2, a3 = _alpha_heii(T), _alpha_heiii(T)
    b1, b2 = _beta_he(T, 285335.0), _beta_he(T, 631515.0)
    dHI = ne * aH * xHII - ne * bH * xHI - photo['HI'] * xHI
    dHeI = ne * a2 * xHeII - ne * b1 * xHeI - photo['HeI'] * xHeI
    dHeIII = ne * b2 * xHeII - ne * a3 * xHeIII + photo['HeII'] * xHeII
    heating = nH * xHI * photo_heat['HI'] + nHe * xHeI * photo_heat['HeI'] + nHe * xHeII * photo_heat['HeII']
    cooling = ne * (nH * xHI * (_cgs_gamma_line_eHI(T) + _cgs_gamma_ion_eHI(T)) + nH * xHII * (_cgs_gamma_ff_eHII(T) + _cgs_gamma_B_eHII(T)))
    cooling += ne * nHe * (xHeI * 1.0e-24 + xHeII * 3.0e-24 + xHeIII * 5.0e-27 * np.sqrt(T))
    return dHI, dHeI, dHeIII, heating - cooling


def source_state(mesh, fluid, par):
    code = _code_units(par); interior = slice(par.noghost, par.noghost + par.nogrid)
    xHI = np.asarray(getattr(fluid, 'xHI', np.ones_like(fluid.rho[interior]))[interior], float).copy()
    xHeI = np.asarray(getattr(fluid, 'xHeI', np.ones_like(xHI))[interior] if hasattr(fluid, 'xHeI') else np.ones_like(xHI), float).copy()
    xHeII = np.asarray(getattr(fluid, 'xHeII', np.zeros_like(xHI))[interior] if hasattr(fluid, 'xHeII') else np.zeros_like(xHI), float).copy()
    xHeIII = np.clip(1.0 - xHeI - xHeII, 0.0, 1.0)
    sigma = {'HI': np.asarray(getattr(par, 'radiation_group_sigma_gamma'), float), 'HeI': np.asarray(getattr(par, 'radiation_group_sigma_gamma_HeI', getattr(par, 'radiation_group_sigma_gamma')), float), 'HeII': np.asarray(getattr(par, 'radiation_group_sigma_gamma_HeII', getattr(par, 'radiation_group_sigma_gamma')), float)}
    eps = {'HI': np.asarray(getattr(par, 'radiation_group_epsilon_gamma'), float), 'HeI': np.asarray(getattr(par, 'radiation_group_epsilon_gamma_HeI', getattr(par, 'radiation_group_epsilon_gamma')), float), 'HeII': np.asarray(getattr(par, 'radiation_group_epsilon_gamma_HeII', getattr(par, 'radiation_group_epsilon_gamma')), float)}
    state = {'interior': interior, 'boundary_cm': to_unit_value(mesh.boundary[interior.start:interior.stop + 1], code.length_unit), 'volume_cm3': to_unit_value(mesh.vol[interior], code.volume_unit), 'radius_kpc': to_unit_value(mesh.coordinate[interior], code.length_unit) / 3.08567758e21, 'rho_g_cm3': to_unit_value(fluid.rho[interior], code.density_unit), 'temperature_K': to_unit_value(fluid.temp[interior], code.temperature_unit), 'specific_energy_erg_g': to_unit_value(fluid.Energy[interior], code.energy_unit) / to_unit_value(fluid.Mass[interior], code.mass_unit), 'gamma': getattr(par, 'gamma', 5.0 / 3.0), 'hydrogen_mass_fraction': getattr(par, 'hydrogen_mass_fraction', 0.7), 'helium_mass_fraction': getattr(par, 'helium_mass_fraction', 0.28), 'xHI': xHI, 'xHeI': xHeI, 'xHeIII': xHeIII, 'sigma_gamma_cm2': sigma, 'epsilon_gamma_erg': eps, 'thermal_coupling': getattr(par, 'hydrogen_thermal_coupling', True), 'explicit_tolerance': getattr(par, 'explicit_tolerance', 0.1), 'relative_tolerance': getattr(par, 'relative_tolerance', 1.0e-3), 'absolute_tolerance': getattr(par, 'absolute_tolerance', 1.0e-10)}
    state['nH_cm3'] = state['rho_g_cm3'] * state['hydrogen_mass_fraction'] / PROTON_MASS_CGS
    _closure(state)
    return state


def ionization_fraction_rate(state, ngamma):
    dHI, dHeI, dHeIII, _ = _rates(state, ngamma)
    return np.maximum(np.abs(dHI), np.maximum(np.abs(dHeI), np.abs(dHeIII)))


def thermal_rate(state, ngamma):
    return _rates(state, ngamma)[3]


def get_timestep(state, ngamma, remaining_s, dtmax_s):
    rate = ionization_fraction_rate(state, ngamma)
    scale = np.minimum.reduce([state['xHI'], 1.0 - state['xHI'], state['xHeI'], 1.0 - state['xHeI'], state['xHeIII'], 1.0 - state['xHeIII']])
    chem = np.min(scale / np.maximum(rate, 1.0e-99))
    return min(float(remaining_s), float(dtmax_s), 0.1 * chem), thermal_rate(state, ngamma)


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


def apply_state(state, fluid, par):
    i = state['interior']; code = _code_units(par)
    fluid.xHI[i] = state['xHI']
    fluid.xHeI[i] = state['xHeI']; fluid.xHeII[i] = state['xHeII']; fluid.xHeIII[i] = state['xHeIII']
    fluid.temp[i] = from_unit_value(state['temperature_K'], code.temperature_unit)
    fluid.mu[i] = state['mu']


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
    def apply_state(self, state, fluid, par): return apply_state(state, fluid, par)
    def get_source_timestep_fast(self, mesh, fluid, par, remaining): return get_timestep(source_state(mesh, fluid, par), None, remaining, remaining)[0]
    def apply_fast(self, dt, mesh, fluid, par): raise NotImplementedError('hydrogen_helium uses the static local subcycle path')
