"""Numerical tools for the constant-pressure HM12 PIE parcel benchmark."""

import numpy as np
from scipy.integrate import solve_ivp
import unyt


SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
PROTON_MASS_G = unyt.mp.to_value(unyt.g)
BOLTZMANN_ERG_cgs_K = unyt.kb.to_value(unyt.erg / unyt.K)


def hydrogen_number_density_isobaric_cgs_cm3(temperature_proper_cgs_K, density_initial_proper_cgs_cm3, temperature_initial_proper_cgs_K):
    """Return n_H for a parcel held at its initial ideal-gas pressure."""
    return density_initial_proper_cgs_cm3 * temperature_initial_proper_cgs_K / np.asarray(temperature_proper_cgs_K)


def pressure_proper_cgs_erg_cm3_from_nh(temperature_proper_cgs_K, density_nH_cgs_cm3, hydrogen_mass_fraction, mu):
    """Return ideal-gas pressure in erg cm^-3."""
    return (
        density_nH_cgs_cm3 * BOLTZMANN_ERG_cgs_K * temperature_proper_cgs_K
        / (hydrogen_mass_fraction * mu)
    )


def net_rate(table, temperature_proper_cgs_K, density_nH_cgs_cm3, metallicity, redshift):
    heating, cooling = table.rates(
        temperature_proper_cgs_K, density_nH_cgs_cm3, metallicity=metallicity, redshift=redshift
    )
    return np.asarray(heating) - np.asarray(cooling)


def integrate_isobaric_case(
    table,
    density_initial_proper_cgs_cm3,
    temperature_initial_proper_cgs_K,
    time_final_proper_Myr,
    output_count,
    hydrogen_mass_fraction,
    mu,
    gamma,
    metallicity,
    redshift,
    temperature_floor,
):
    """Integrate the one-zone isobaric enthalpy equation.

    At fixed pressure, n_H*T is constant and
    ``dT/dt = (gamma-1)/gamma * (mu m_p/k_B) * net_rate/rho``.
    """
    def rhs(time_proper_Myr, state):
        temperature_proper_cgs_K = max(float(state[0]), temperature_floor)
        density_nH_cgs_cm3 = hydrogen_number_density_isobaric_cgs_cm3(
            temperature_proper_cgs_K, density_initial_proper_cgs_cm3, temperature_initial_proper_cgs_K
        )
        rho_cgs_g_cm3 = density_nH_cgs_cm3 * PROTON_MASS_G / hydrogen_mass_fraction
        rate = float(net_rate(
            table, temperature_proper_cgs_K, density_nH_cgs_cm3, metallicity, redshift
        ))
        dtemperature_dt = (
            (gamma - 1.0) / gamma * mu * PROTON_MASS_G / BOLTZMANN_ERG_cgs_K
            * rate / rho_cgs_g_cm3 * SECONDS_PER_MYR
        )
        return [dtemperature_dt]

    time_proper_Myr = np.linspace(0.0, time_final_proper_Myr, output_count)
    solution = solve_ivp(
        rhs,
        (time_proper_Myr[0], time_proper_Myr[-1]),
        [temperature_initial_proper_cgs_K],
        t_eval=time_proper_Myr,
        method='BDF',
        rtol=2.0e-7,
        atol=max(temperature_floor * 1.0e-5, 1.0e-3),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    temperature_proper_cgs_K = np.maximum(solution.y[0], temperature_floor)
    density_nH_cgs_cm3 = hydrogen_number_density_isobaric_cgs_cm3(
        temperature_proper_cgs_K, density_initial_proper_cgs_cm3, temperature_initial_proper_cgs_K
    )
    pressure_proper_cgs_erg_cm3 = pressure_proper_cgs_erg_cm3_from_nh(
        temperature_proper_cgs_K, density_nH_cgs_cm3, hydrogen_mass_fraction, mu
    )
    return {
        'time_proper_Myr': solution.t,
        'temperature_proper_cgs_K': temperature_proper_cgs_K,
        'density_nH_cgs_cm3': density_nH_cgs_cm3,
        'pressure_proper_cgs_erg_cm3': pressure_proper_cgs_erg_cm3,
    }


def isobaric_growth_rate(
    table,
    temperature_proper_cgs_K,
    density_initial_proper_cgs_cm3,
    temperature_initial_proper_cgs_K,
    hydrogen_mass_fraction,
    mu,
    gamma,
    metallicity,
    redshift,
    temperature_floor,
):
    """Return d(dT/dt)/dT in Myr^-1 at fixed pressure.

    Positive values indicate locally growing isobaric temperature
    perturbations; negative values indicate local thermal stability.
    """
    grid = np.logspace(2, 8, 4096)
    density_nH_cgs_cm3 = hydrogen_number_density_isobaric_cgs_cm3(grid, density_initial_proper_cgs_cm3, temperature_initial_proper_cgs_K)
    rho_cgs_g_cm3 = density_nH_cgs_cm3 * PROTON_MASS_G / hydrogen_mass_fraction
    rate = net_rate(table, grid, density_nH_cgs_cm3, metallicity, redshift)
    dtemperature_dt = (
        (gamma - 1.0) / gamma * mu * PROTON_MASS_G / BOLTZMANN_ERG_cgs_K
        * rate / rho_cgs_g_cm3 * SECONDS_PER_MYR
    )
    growth_grid = np.gradient(dtemperature_dt, grid)
    return np.interp(
        np.log(np.maximum(temperature_proper_cgs_K, temperature_floor)),
        np.log(grid),
        growth_grid,
    )
