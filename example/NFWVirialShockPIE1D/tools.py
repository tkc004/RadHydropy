"""HM12 PIE diagnostics for the NFW virial-shock cooling experiment."""

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS
from radhydropy.thermo_networks.pie import MetalPIETable

BASE_PATH = Path(__file__).resolve().parents[1] / 'NFWVirialShockAdiabatic1D' / 'tools.py'
SPEC = importlib.util.spec_from_file_location('adiabatic_nfw_tools_for_pie', BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

nfw_halo_parameters = BASE.nfw_halo_parameters
virial_temperature = BASE.virial_temperature
_snapshot_profiles = BASE._snapshot_profiles
_locate_shock = BASE._locate_shock
CodeUnits = BASE.CodeUnits
cosmic_mean_baryon_density = BASE.cosmic_mean_baryon_density


def pie_equilibrium_temperature(
    density, table, hydrogen_mass_fraction, metallicity, redshift
):
    """Return the local HM12 PIE thermal-equilibrium temperature.

    The table is sampled over its temperature axis and the zero of
    ``heating-cooling`` is linearly interpolated in log temperature.  If a
    density has no sign change in the tabulated range, the closest tabulated
    state is used rather than extrapolating beyond the PIE table.
    """
    density = np.asarray(density, dtype=float)
    n_h = hydrogen_mass_fraction * density / PROTON_MASS_CGS
    log_temperature = np.asarray(table.log_temperature, dtype=float)
    temperature = 10.0 ** log_temperature
    heating, cooling = table.rates(
        temperature[:, None], n_h[None, :],
        metallicity=metallicity, redshift=redshift,
    )
    log_ratio = np.log10(np.maximum(heating, 1.0e-99)) - np.log10(
        np.maximum(cooling, 1.0e-99)
    )
    sign_change = log_ratio[:-1] * log_ratio[1:] <= 0.0
    score = np.abs(log_ratio)
    crossing_index = np.argmax(sign_change, axis=0)
    has_crossing = np.any(sign_change, axis=0)
    closest_index = np.argmin(score, axis=0)
    index = np.where(has_crossing, crossing_index, closest_index)
    index = np.clip(index, 0, len(log_temperature) - 2)
    lower = log_ratio[index, np.arange(density.size)]
    upper = log_ratio[index + 1, np.arange(density.size)]
    denominator = upper - lower
    weight = np.divide(-lower, denominator, out=np.zeros_like(lower), where=denominator != 0.0)
    equilibrium_log_temperature = log_temperature[index] + weight * (
        log_temperature[index + 1] - log_temperature[index]
    )
    closest_temperature = temperature[closest_index]
    result = np.where(has_crossing, 10.0 ** equilibrium_log_temperature, closest_temperature)
    return result * unyt.K


class Simwrap(BASE.Simwrap):
    """NFW PIE IC with a correlated, centrally enhanced baryon fluctuation."""

    def __init__(self, icparams, code_units=None, pie_table=None,
                 pie_redshift=0.0, metallicity=1.0,
                 hydrogen_mass_fraction=1.0):
        super().__init__(icparams, code_units=code_units)
        halo = nfw_halo_parameters(
            icparams['halo_mass'], icparams['concentration'],
            icparams['redshift'], icparams['overdensity'], icparams['h0'],
        )
        r200 = halo['virial_radius']
        radius_ratio = np.asarray(
            (self.mesh.coordinate / r200).to_value(unyt.dimensionless)
        )
        amplitude = float(icparams.get('density_fluctuation_amplitude', 0.0))
        slope = float(icparams.get('density_fluctuation_slope', 1.8))
        floor = float(icparams.get('density_fluctuation_floor', 0.0))
        fluctuation = floor + amplitude * np.maximum(radius_ratio, 1.0e-6) ** (-slope)
        mean_density = cosmic_mean_baryon_density(
            icparams['h0'], icparams['omega_b'], icparams['initial_redshift']
        )
        self.fluid.rho = mean_density * (1.0 + fluctuation)
        if pie_table is not None:
            self.fluid.temp = pie_equilibrium_temperature(
                self.fluid.rho.to_value(unyt.g / unyt.cm**3),
                pie_table, hydrogen_mass_fraction, metallicity, pie_redshift,
            )


def _pressure(rho, temperature, mu):
    return rho * BOLTZMANN_CONSTANT_CGS * temperature / (mu * PROTON_MASS_CGS)


def pie_stability_diagnostics(filenames, icparams, runparams, halo):
    profiles = [_snapshot_profiles(name, icparams, runparams) for name in filenames]
    if len(profiles) < 3:
        return []
    table = MetalPIETable(runparams['metal_pie_table_filename'])
    mu = float(icparams['mu'])
    hydrogen_mass_fraction = float(runparams['hydrogen_mass_fraction'])
    r200 = halo['virial_radius'].to_value(unyt.kpc)
    locations = []
    indices = []
    for _, radius, _, temperature, _ in profiles:
        index, location = _locate_shock(radius, temperature, r200)
        indices.append(index)
        locations.append(location)

    downstream = []
    for profile, index in zip(profiles, indices):
        time, radius, rho, temperature, velocity = profile
        if index < 8 or index + 5 > len(radius):
            downstream.append(None)
            continue
        band = slice(index - 8, index - 3)
        rho_post = float(np.median(rho[band]))
        temp_post = float(np.median(temperature[band]))
        pressure_post = float(_pressure(rho_post, temp_post, mu))
        downstream.append((rho_post, temp_post, pressure_post))

    rows = []
    for i in range(1, len(profiles) - 1):
        if downstream[i - 1] is None or downstream[i] is None or downstream[i + 1] is None:
            continue
        time, radius, rho, temperature, velocity = profiles[i]
        dt = profiles[i + 1][0] - profiles[i - 1][0]
        if dt <= 0.0 or indices[i] < 5 or indices[i] + 5 > len(radius):
            continue
        shock_speed = (locations[i + 1] - locations[i - 1]) / dt * 977.792221
        index = indices[i]
        upstream = slice(index + 2, index + 5)
        rho_upstream = float(np.median(rho[upstream]))
        temp_upstream = float(np.median(temperature[upstream]))
        velocity_upstream = float(np.median(velocity[upstream]))
        rho_post, temp_post, _ = downstream[i]
        nH_post = hydrogen_mass_fraction * rho_post / PROTON_MASS_CGS
        heating, cooling = table.rates(
            temp_post, nH_post,
            metallicity=float(runparams.get('metallicity', 1.0)),
            redshift=float(runparams.get('metal_pie_redshift', 0.0)),
        )
        heating = float(np.asarray(heating))
        cooling = float(np.asarray(cooling))
        net_cooling = cooling - heating
        u = abs(velocity_upstream - shock_speed) * 1.0e5
        shock_radius = locations[i] * 3.08567758e21
        lambda_rho = cooling / max(rho_post**2, 1.0e-99)
        lambda_net_rho = net_cooling / max(rho_post**2, 1.0e-99)
        s_cooling = rho_upstream * shock_radius * lambda_rho / max(u**3, 1.0e-99)
        s_net = rho_upstream * shock_radius * lambda_net_rho / max(u**3, 1.0e-99)
        p_before = downstream[i - 1][2]
        p_after = downstream[i + 1][2]
        rho_before = downstream[i - 1][0]
        rho_after = downstream[i + 1][0]
        dlnp_dt = np.log(max(p_after, 1.0e-99) / max(p_before, 1.0e-99)) / dt
        dlnrho_dt = np.log(max(rho_after, 1.0e-99) / max(rho_before, 1.0e-99)) / dt
        gamma_eff = dlnp_dt / dlnrho_dt if abs(dlnrho_dt) > 1.0e-12 else np.nan
        status = 'stable' if gamma_eff > 10.0 / 7.0 else 'unstable'
        rows.append({
            'time_Myr': time,
            'shock_radius_kpc': locations[i],
            'shock_radius_over_R200': locations[i] / r200,
            'shock_speed_km_s': shock_speed,
            'upstream_density_g_cm3': rho_upstream,
            'upstream_velocity_km_s': velocity_upstream,
            'postshock_temperature_K': temp_post,
            'cooling_rate_erg_cm3_s': cooling,
            'photoheating_rate_erg_cm3_s': heating,
            'net_cooling_rate_erg_cm3_s': net_cooling,
            'S_cooling': s_cooling,
            'S_net': s_net,
            'gamma_eff': gamma_eff,
            'shock_status': status,
        })
    return rows


def write_stability_report(rows, filename):
    header = (
        'time_Myr shock_radius_kpc shock_radius_over_R200 shock_speed_km_s '
        'upstream_density_g_cm3 upstream_velocity_km_s postshock_temperature_K '
        'cooling_rate_erg_cm3_s photoheating_rate_erg_cm3_s '
        'net_cooling_rate_erg_cm3_s S_cooling S_net gamma_eff shock_status\n'
    )
    with open(filename, 'w', encoding='utf-8') as report:
        report.write(header)
        for row in rows:
            report.write(
                '%(time_Myr).8g %(shock_radius_kpc).8g %(shock_radius_over_R200).8g '
                '%(shock_speed_km_s).8g %(upstream_density_g_cm3).8g '
                '%(upstream_velocity_km_s).8g %(postshock_temperature_K).8g '
                '%(cooling_rate_erg_cm3_s).8g %(photoheating_rate_erg_cm3_s).8g '
                '%(net_cooling_rate_erg_cm3_s).8g %(S_cooling).8g %(S_net).8g '
                '%(gamma_eff).8g %(shock_status)s\n' % row
            )


def plot_snapshots(filenames, icparams, runparams, halo, figure_filename):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(filenames)))
    r200 = halo['virial_radius'].to_value(unyt.kpc)
    tvir = virial_temperature(halo, icparams['mu']).to_value(unyt.K)
    for color, filename in zip(colors, filenames):
        time, radius, density, temperature, _ = _snapshot_profiles(filename, icparams, runparams)
        axes[0].plot(radius, density, color=color, label=f'{time:.0f} Myr')
        axes[1].plot(radius, temperature, color=color, label=f'{time:.0f} Myr')
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[0].set_xlabel('r [kpc]')
    axes[1].set_xlabel('r [kpc]')
    axes[0].set_ylabel(r'$\rho$ [g cm$^{-3}$]')
    axes[1].set_ylabel('T [K]')
    for axis in axes:
        axis.axvline(r200, color='black', ls=':', alpha=0.6)
        axis.axvline(2.0 * r200, color='black', ls='--', alpha=0.6)
        axis.grid(True, which='both', alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    axes[1].axhline(tvir, color='red', ls=':', label=r'$T_{\rm vir}$')
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('HM12 PIE virial shock around %.2g Msun NFW halo' % halo['mass'].to_value(unyt.Msun))
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)
