"""Numerical tools for the constant-pressure HM12 PIE parcel benchmark."""

import numpy as np
from scipy.integrate import solve_ivp
import unyt


SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
PROTON_MASS_G = unyt.mp.to_value(unyt.g)
BOLTZMANN_ERG_K = unyt.kb.to_value(unyt.erg / unyt.K)


def isobaric_density(temperature, density_initial, temperature_initial):
    """Return n_H for a parcel held at its initial ideal-gas pressure."""
    return density_initial * temperature_initial / np.asarray(temperature)


def pressure_from_nh(temperature, density, hydrogen_mass_fraction, mu):
    """Return ideal-gas pressure in erg cm^-3."""
    return (
        density * BOLTZMANN_ERG_K * temperature
        / (hydrogen_mass_fraction * mu)
    )


def net_rate(table, temperature, density, metallicity, redshift):
    heating, cooling = table.rates(
        temperature, density, metallicity=metallicity, redshift=redshift
    )
    return np.asarray(heating) - np.asarray(cooling)


def integrate_isobaric_case(
    table,
    density_initial,
    temperature_initial,
    time_final_Myr,
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
    def rhs(time_Myr, state):
        temperature = max(float(state[0]), temperature_floor)
        density = isobaric_density(
            temperature, density_initial, temperature_initial
        )
        rho = density * PROTON_MASS_G / hydrogen_mass_fraction
        rate = float(net_rate(
            table, temperature, density, metallicity, redshift
        ))
        dtemperature_dt = (
            (gamma - 1.0) / gamma * mu * PROTON_MASS_G / BOLTZMANN_ERG_K
            * rate / rho * SECONDS_PER_MYR
        )
        return [dtemperature_dt]

    times = np.linspace(0.0, time_final_Myr, output_count)
    solution = solve_ivp(
        rhs,
        (times[0], times[-1]),
        [temperature_initial],
        t_eval=times,
        method='BDF',
        rtol=2.0e-7,
        atol=max(temperature_floor * 1.0e-5, 1.0e-3),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    temperature = np.maximum(solution.y[0], temperature_floor)
    density = isobaric_density(
        temperature, density_initial, temperature_initial
    )
    pressure = pressure_from_nh(
        temperature, density, hydrogen_mass_fraction, mu
    )
    return {
        'time_Myr': solution.t,
        'temperature_K': temperature,
        'density_nH_cm3': density,
        'pressure_erg_cm3': pressure,
    }


def isobaric_growth_rate(
    table,
    temperature,
    density_initial,
    temperature_initial,
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
    density = isobaric_density(grid, density_initial, temperature_initial)
    rho = density * PROTON_MASS_G / hydrogen_mass_fraction
    rate = net_rate(table, grid, density, metallicity, redshift)
    dtemperature_dt = (
        (gamma - 1.0) / gamma * mu * PROTON_MASS_G / BOLTZMANN_ERG_K
        * rate / rho * SECONDS_PER_MYR
    )
    growth_grid = np.gradient(dtemperature_dt, grid)
    return np.interp(
        np.log(np.maximum(temperature, temperature_floor)),
        np.log(grid),
        growth_grid,
    )
