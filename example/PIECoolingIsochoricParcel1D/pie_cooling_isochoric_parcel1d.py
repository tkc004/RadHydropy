"""Isochoric HM12 PIE heating/cooling parcel benchmark."""

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, EXAMPLE_ROOT, EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import radhydropy.io as rio
from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
from radhydropy.thermo_networks.pie import MetalPIETable
from radhydropy.units import CodeUnits
import example_utils as eu
from tools import Simwrap


DEFAULT_CONFIG = EXAMPLE_DIR / 'pie_cooling_isochoric_parcel1d.yaml'
CASES = (
    ('diffuse_cold', 1.0e-4, 1.0e4),
    ('diffuse_hot', 1.0e-4, 1.0e6),
    ('dense_cold', 1.0e-2, 1.0e4),
    ('dense_hot', 1.0e-2, 1.0e6),
)
SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
PROTON_MASS_G = unyt.mp.to_value(unyt.g)
BOLTZMANN_ERG_K = unyt.kb.to_value(unyt.erg / unyt.K)


def _net_rate(table, temperature, density, metallicity, redshift):
    heating, cooling = table.rates(
        temperature, density, metallicity=metallicity, redshift=redshift
    )
    return np.asarray(heating) - np.asarray(cooling)


def _equilibrium_temperature(table, density, metallicity, redshift, mu):
    temperatures = np.logspace(
        table.log_temperature[0], table.log_temperature[-1], 2048
    )
    net = _net_rate(table, temperatures, density, metallicity, redshift)
    crossings = np.flatnonzero((net[:-1] >= 0.0) & (net[1:] < 0.0))
    if not len(crossings):
        return np.nan
    index = crossings[0]
    x0, x1 = np.log(temperatures[index:index + 2])
    y0, y1 = net[index:index + 2]
    if y1 == y0:
        return float(temperatures[index])
    return float(np.exp(x0 - y0 * (x1 - x0) / (y1 - y0)))


def _snapshot(filename, time_Myr=None):
    with h5py.File(filename, 'r') as handle:
        data = handle['Data']
        header = handle['Header']
        noghost = int(header.attrs.get('noghost', 0))
        nogrid = int(header.attrs['nogrid'])
        first = noghost
        last = first + nogrid
        return {
            'time_Myr': (
                float(header.attrs['Time']) / SECONDS_PER_MYR
                if time_Myr is None else float(time_Myr)
            ),
            'density': np.asarray(data['Density'][()])[first:last],
            'temperature': np.asarray(data['Temperature'][()])[first:last],
        }


def _run_case(runparams, icparams, label, density, temperature, table):
    case = dict(runparams)
    output_dir = EXAMPLE_DIR / 'outputs' / label
    output_dir.mkdir(parents=True, exist_ok=True)
    case['outdir'] = str(output_dir)
    case['savedir'] = str(output_dir)
    case['outfileprefix'] = f'Output_{label}'
    case['ICfilename'] = str(output_dir / f'InitialCondition_{label}.hdf5')
    case_icparams = dict(icparams)
    case_icparams['tempini'] = temperature * unyt.K
    case_icparams['vini'] = 0.0 * unyt.cm / unyt.s
    case_icparams['hydrogen_mass_fraction'] = case['hydrogen_mass_fraction']
    case_icparams['proton_mass_g'] = PROTON_MASS_G
    eu.clean_previous_outputs(case)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_units = CodeUnits.from_mapping(case['CodeUnits'])
    initial = Simwrap(case_icparams, code_units, density)
    rio.writehdf5(initial, case['ICfilename'])
    runtime_only = {
        'box_size', 'coordinate_system', 'current_time', 'grid_cells',
        'number_of_cells', 'initial_temperature', 'mean_molecular_weight',
        'final_time', 'evolution_timestep', 'chemistry_timestep',
    }
    sim = Rsim({key: value for key, value in case.items()
                if key not in runtime_only})
    # This is a one-cell isochoric parcel.  Use the dedicated source-only
    # mode so no hydro flux gradient is evaluated on the single active cell.
    sim.RunAll(outputtime=0, mode='sources')
    snapshots = sorted(output_dir.glob(f'{case["outfileprefix"]}_*.hdf5'))
    if len(snapshots) < 2:
        raise RuntimeError(f'expected snapshots in {output_dir}')
    initial_rate = float(_net_rate(
        table, temperature, density, case['metallicity'], case['metal_pie_redshift']
    ))
    rho = density * PROTON_MASS_G / case['hydrogen_mass_fraction']
    thermal_energy = rho * BOLTZMANN_ERG_K * temperature / (
        (case['gamma'] - 1.0) * case_icparams['muini'] * PROTON_MASS_G
    )
    thermal_time = thermal_energy / max(abs(initial_rate), 1.0e-99) / SECONDS_PER_MYR
    equilibrium = _equilibrium_temperature(
        table, density, case['metallicity'], case['metal_pie_redshift'], case_icparams['muini']
    )
    return {
        'label': label,
        'density': density,
        'temperature_initial': temperature,
        'initial_rate': initial_rate,
        'thermal_time_Myr': thermal_time,
        'equilibrium_temperature': equilibrium,
        'snapshots': snapshots,
    }


def _write_report(results, filename):
    with open(filename, 'w', encoding='utf-8') as report:
        report.write(
            'case nH_cm3 T_initial_K T_final_K T_equilibrium_K '
            'initial_net_rate_erg_cm3_s initial_thermal_time_Myr '
            'density_relative_change\n'
        )
        for result in results:
            initial = _snapshot(result['snapshots'][0])
            final = _snapshot(result['snapshots'][-1])
            density_change = np.median(final['density']) / np.median(initial['density']) - 1.0
            report.write(
                '%s %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (
                    result['label'], result['density'], result['temperature_initial'],
                    np.median(final['temperature']), result['equilibrium_temperature'],
                    result['initial_rate'], result['thermal_time_Myr'], density_change,
                )
            )


def _plot(results, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    right_markers = []
    for result in results:
        times = []
        temperatures = []
        for output_index, snapshot in enumerate(result['snapshots']):
            data = _snapshot(snapshot, time_Myr=output_index * 50.0)
            times.append(data['time_Myr'])
            temperatures.append(np.median(data['temperature']))
        line, = axes[0].plot(times, temperatures, marker='o', label=result['label'])
        equilibrium = result['equilibrium_temperature']
        if np.isfinite(equilibrium):
            axes[0].axhline(equilibrium, color=line.get_color(), ls=':', alpha=0.6)
        temperatures_grid = np.logspace(2, 8, 512)
        net = _net_rate(
            TABLE, temperatures_grid, result['density'], METALLICITY, REDSHIFT
        )
        net_per_nh2 = net / result['density'] ** 2
        magnitude = np.maximum(np.abs(net_per_nh2), 1.0e-99)
        heating = np.where(net_per_nh2 >= 0.0, magnitude, np.nan)
        cooling = np.where(net_per_nh2 < 0.0, magnitude, np.nan)
        cold_markers = result['label'].endswith('_cold')
        axes[1].plot(
            temperatures_grid,
            heating,
            color=line.get_color(),
            marker='o' if cold_markers else None,
            markevery=32 if cold_markers else None,
            markersize=3.0,
            label=result['label'],
        )
        axes[1].plot(
            temperatures_grid,
            cooling,
            color=line.get_color(),
            linestyle='--',
            marker='o' if cold_markers else None,
            markevery=32 if cold_markers else None,
            markersize=3.0,
            label='_nolegend_',
        )
        initial_rate = float(_net_rate(
            TABLE,
            result['temperature_initial'],
            result['density'],
            METALLICITY,
            REDSHIFT,
        )) / result['density'] ** 2
        right_markers.append((
            result['temperature_initial'],
            max(abs(initial_rate), 1.0e-99),
            line.get_color(),
        ))
    for temperature, rate, color in right_markers:
        axes[1].plot(
            temperature,
            rate,
            marker='o',
            markersize=6,
            color=color,
            linestyle='none',
            markeredgecolor='black',
            markeredgewidth=0.5,
            label='_nolegend_',
            zorder=5,
        )
    axes[0].set_xlabel('time [Myr]')
    axes[0].set_ylabel('temperature [K]')
    axes[0].set_yscale('log')
    axes[1].set_xlabel('temperature [K]')
    axes[1].set_ylabel(r'$|\Gamma-\mathcal{C}|/n_{\rm H}^2$ [erg cm$^3$ s$^{-1}$]')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle('Isochoric HM12 PIE thermal evolution')
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


TABLE = None
METALLICITY = 1.0
REDSHIFT = 0.0


def main(config_filename=DEFAULT_CONFIG):
    global TABLE, METALLICITY, REDSHIFT
    config_filename = Path(config_filename).resolve()
    runparams, icparams = load_example_parameters(config_filename)
    table_path = (config_filename.parent / runparams['metal_pie_table_filename']).resolve()
    runparams['metal_pie_table_filename'] = str(table_path)
    TABLE = MetalPIETable(table_path)
    if not TABLE.is_hm12_uv_background:
        raise ValueError('the example requires an HM12 UV-background table')
    METALLICITY = float(runparams['metallicity'])
    REDSHIFT = float(runparams['metal_pie_redshift'])
    results = []
    for label, density, temperature in CASES:
        results.append(_run_case(runparams, icparams, label, density, temperature, TABLE))
    figure = EXAMPLE_DIR / 'PIECoolingIsochoricParcel1D.jpg'
    report = EXAMPLE_DIR / 'PIECoolingIsochoricParcel1D_ThermalReport.txt'
    _plot(results, figure)
    _write_report(results, report)
    for result in results:
        final = _snapshot(result['snapshots'][-1])
        print(
            '%s: T_initial=%.6g K, T_final=%.6g K, T_eq=%.6g K' % (
                result['label'], result['temperature_initial'],
                np.median(final['temperature']), result['equilibrium_temperature'],
            )
        )
    print('figure = %s' % figure)
    print('report = %s' % report)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
