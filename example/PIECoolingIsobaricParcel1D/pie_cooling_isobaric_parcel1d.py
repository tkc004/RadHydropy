"""Constant-pressure HM12 PIE thermal-instability benchmark."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, EXAMPLE_ROOT, EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from example_utils import load_nested_example_parameters
from radhydropy.thermo_networks.pie import MetalPIETable
from tools import integrate_isobaric_case, isobaric_growth_rate, net_rate


DEFAULT_CONFIG = EXAMPLE_DIR / 'pie_cooling_isobaric_parcel1d.yaml'


def _write_case_csv(result, filename):
    fields = tuple(result)
    with open(filename, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(zip(*(result[field] for field in fields)))


def _plot(results, filename):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for result in results:
        label = result['label']
        color = result['color']
        axes[0, 0].plot(result['time_Myr'], result['temperature_cgs_K'], color=color, label=label)
        axes[0, 1].plot(result['time_Myr'], result['density_nH_cgs_cm3'], color=color, label=label)
        axes[1, 0].plot(
            result['time_Myr'], result['gamma_eff'], color=color, label=label
        )
        axes[1, 1].plot(
            result['time_Myr'], result['growth_rate_Myr_inv'], color=color, label=label
        )
    axes[0, 0].set_ylabel(r'$T$ [K]')
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_ylabel(r'$n_{\rm H}$ [cm$^{-3}$]')
    axes[0, 1].set_yscale('log')
    axes[1, 0].set_ylabel(r'$\gamma_{\rm eff}=d\ln P/d\ln\rho$')
    axes[1, 0].axhline(0.0, color='black', lw=0.8, ls=':')
    axes[1, 1].set_ylabel(r'$d(dT/dt)/dT$ [Myr$^{-1}$]')
    axes[1, 1].axhline(0.0, color='black', lw=0.8, ls=':')
    axes[1, 1].set_title('positive: isobaric thermal instability')
    for axis in axes.flat:
        axis.set_xlabel('time [Myr]')
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle('Isobaric HM12 PIE thermal-instability experiment')
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def _plot_rate(results, table, metallicity, redshift, filename):
    """Plot the PIE rate per n_H^2 along each constant-pressure path."""
    fig, axis = plt.subplots(figsize=(7.5, 5.2))
    temperatures = np.logspace(2, 8, 512)
    rate_results = (
        [result for result in results if result['label'].endswith('_cold')]
        + [result for result in results if not result['label'].endswith('_cold')]
    )
    for result in rate_results:
        density = result['density_nH_cgs_cm3'][0] * result['temperature_cgs_K'][0] / temperatures
        rate = net_rate(table, temperatures, density, metallicity, redshift)
        rate_per_nh2 = rate / density ** 2
        magnitude = np.maximum(np.abs(rate_per_nh2), 1.0e-99)
        heating = np.where(rate_per_nh2 >= 0.0, magnitude, np.nan)
        cooling = np.where(rate_per_nh2 < 0.0, magnitude, np.nan)
        marker = 'o' if result['label'].endswith('_cold') else None
        markevery = 32 if marker else None
        axis.plot(
            temperatures, heating, color=result['color'], marker=marker,
            markevery=markevery, markersize=3.0, label=result['label'],
        )
        axis.plot(
            temperatures, cooling, color=result['color'], linestyle='--',
            marker=marker, markevery=markevery, markersize=3.0,
            label='_nolegend_',
        )
    axis.set_xlabel('temperature [K]')
    axis.set_ylabel(r'$|\Gamma-\mathcal{C}|/n_{\rm H}^2$ [erg cm$^3$ s$^{-1}$]')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title('Isobaric HM12 PIE thermal rate')
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    runparams, _ = load_nested_example_parameters(config_filename)
    table_path = (config_filename.parent / runparams['metal_pie_table_filename']).resolve()
    table = MetalPIETable(table_path)
    if not table.is_hm12_uv_background:
        raise ValueError('the example requires an HM12 UV-background table')
    gamma = float(runparams['gamma'])
    mu = float(runparams['mu'])
    metallicity = float(runparams['metallicity'])
    redshift = float(runparams['metal_pie_redshift'])
    hydrogen_mass_fraction = float(runparams['hydrogen_mass_fraction'])
    time_final = runparams['timesim'].to_value('Myr')
    temperature_floor = runparams['temperature_floor'].to_value('K')
    output_count = int(runparams['output_count'])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    results = []
    report_rows = []
    output_dir = EXAMPLE_DIR / 'outputs'
    output_dir.mkdir(exist_ok=True)
    for stale_csv in output_dir.glob('*.csv'):
        stale_csv.unlink()
    for index, (label, density, temperature) in enumerate(runparams['cases']):
        density = float(density)
        temperature = float(temperature)
        result = integrate_isobaric_case(
            table, density, temperature, time_final, output_count,
            hydrogen_mass_fraction, mu, gamma, metallicity, redshift,
            temperature_floor,
        )
        result['growth_rate_Myr_inv'] = isobaric_growth_rate(
            table, result['temperature_cgs_K'], density, temperature,
            hydrogen_mass_fraction, mu, gamma, metallicity, redshift,
            temperature_floor,
        )
        # The integration enforces P=P_initial analytically, so the
        # effective EOS is gamma_eff=dlnP/dlnrho=0.  A finite-difference
        # estimate becomes undefined after a case reaches the temperature
        # floor and its density stops changing.
        result['gamma_eff'] = np.zeros_like(result['temperature_cgs_K'])
        result['label'] = str(label)
        result['color'] = colors[index % len(colors)]
        csv_result = {
            key: result[key]
            for key in (
                'time_Myr', 'temperature_cgs_K', 'density_nH_cgs_cm3',
                'pressure_cgs_erg_cm3', 'gamma_eff', 'growth_rate_Myr_inv',
            )
        }
        _write_case_csv(csv_result, output_dir / f'{label}.csv')
        results.append(result)
        report_rows.append((
            label, density, temperature,
            float(result['temperature_cgs_K'][-1]),
            float(result['density_nH_cgs_cm3'][-1]),
            float(np.max(result['growth_rate_Myr_inv'])),
            float(np.min(result['growth_rate_Myr_inv'])),
            float(np.max(np.abs(result['gamma_eff']))),
            float(np.max(np.abs(
                result['pressure_cgs_erg_cm3'] / result['pressure_cgs_erg_cm3'][0] - 1.0
            ))),
        ))
    report = EXAMPLE_DIR / 'PIECoolingIsobaricParcel1D_ThermalReport.txt'
    with open(report, 'w', encoding='utf-8') as handle:
        handle.write(
            'case nH_initial_cgs_cm3 T_initial_cgs_K T_final_cgs_K nH_final_cgs_cm3 '
            'max_growth_Myr^-1 min_growth_Myr^-1 max_abs_gamma_eff '
            'max_pressure_fractional_error\n'
        )
        for row in report_rows:
            handle.write('%s %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % row)
    _plot(results, EXAMPLE_DIR / 'PIECoolingIsobaricParcel1D.jpg')
    _plot_rate(
        results, table, metallicity, redshift,
        EXAMPLE_DIR / 'PIECoolingIsobaricParcel1D_Rate.jpg',
    )
    for row in report_rows:
        print(
            '%s: T_final=%.6g K, nH_final=%.6g cm^-3, '
            'growth=[%.3g, %.3g] Myr^-1, max|gamma_eff|=%.3g' %
            (row[0], row[3], row[4], row[5], row[6], row[7])
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
