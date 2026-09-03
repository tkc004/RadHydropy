"""Run and compare stable, marginal, and low-mass virial-shock cases."""

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
for path in (PROJECT_ROOT, EXAMPLE_DIR.parent, EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'radhydropy-cache'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'radhydropy-matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

import example_utils as eu
from radhydropy.io import load_output_time_list
from radhydropy.thermo_networks.pie import MetalPIETable
from tools import (
    GAMMA_CRITICAL, locate_shock, load_snapshot, nfw_halo_parameters,
    pie_stability_diagnostics,
)

RUNNER_SPEC = importlib.util.spec_from_file_location(
    'boundary_virial_shock_runner',
    EXAMPLE_DIR / 'nfw_boundary_driven_virial_shock1d.py',
)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(RUNNER)

CONFIGS = (
    EXAMPLE_DIR / 'nfw_boundary_driven_virial_shock1d.yaml',
    EXAMPLE_DIR / 'nfw_boundary_driven_virial_shock_3e11.yaml',
    EXAMPLE_DIR / 'nfw_boundary_driven_virial_shock_1e11.yaml',
)


def _case_diagnostics(config_filename):
    config = eu.load_nested_example_config(config_filename)
    par_config = config['par']
    icparams = config['initial_condition']
    exampleparams = config['example']
    pie_table_filename = (
        config_filename.parent
        / par_config['thermochemistry']['metal_pie_table_filename']
    ).resolve()
    pie_outdir = (config_filename.parent / exampleparams['pie_outdir']).resolve()
    pie_schedule = (
        config_filename.parent / exampleparams['pie_outputtimefilename']
    )
    files = sorted(pie_outdir.glob('Output_*.hdf5'))
    relative_times = load_output_time_list(pie_schedule).to_value(unyt.Myr)
    offset = exampleparams['adiabatic_final_time'].to_value(unyt.Myr)
    times = relative_times + offset
    if len(files) != len(times):
        raise RuntimeError(f'{config_filename.name}: output count does not match schedule')
    halo = nfw_halo_parameters(
        icparams['halo_mass'], icparams['concentration'], icparams['redshift'],
        icparams['overdensity'], icparams['h0'],
    )
    table = MetalPIETable(pie_table_filename)
    stability = pie_stability_diagnostics(
        files, times, halo, table, par_config, icparams['mu']
    )
    stability_by_time = {row['time_Myr']: row for row in stability}
    shock_radius = []
    gamma_eff = []
    for filename, time in zip(files, times):
        snapshot = load_snapshot(filename)
        index = locate_shock(
            snapshot, halo['virial_radius'].to_value(unyt.kpc)
        )
        shock_radius.append(
            np.nan if index is None else snapshot['radius_kpc'][index]
            / halo['virial_radius'].to_value(unyt.kpc)
        )
        row = stability_by_time.get(float(time))
        gamma_eff.append(np.nan if row is None else row['gamma_eff'])
    return {
        'mass_Msun': halo['mass'].to_value(unyt.Msun),
        'label': r'$10^{12}\,M_\odot$' if halo['mass'].to_value(unyt.Msun) > 5e11
        else (r'$3\times10^{11}\,M_\odot$' if halo['mass'].to_value(unyt.Msun) > 2e11
              else r'$10^{11}\,M_\odot$'),
        'times_Myr': np.asarray(times),
        'shock_radius': np.asarray(shock_radius),
        'gamma_eff': np.asarray(gamma_eff),
    }


def _write_summary(cases, filename):
    with Path(filename).open('w', encoding='utf-8') as stream:
        stream.write('halo_mass_Msun time_Myr shock_radius_over_R200 gamma_eff status\n')
        for case in cases:
            for time, radius, gamma_eff in zip(
                case['times_Myr'], case['shock_radius'], case['gamma_eff']
            ):
                if not np.isfinite(radius):
                    status = 'no_resolved_virial_shock'
                elif np.isfinite(gamma_eff) and gamma_eff < GAMMA_CRITICAL:
                    status = 'unstable'
                else:
                    status = 'supported'
                stream.write(
                    f"{case['mass_Msun']:.8g} {time:.8g} {radius:.8g} "
                    f"{gamma_eff:.8g} {status}\n"
                )


def _plot(cases, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    for case in cases:
        axes[0].plot(case['times_Myr'], case['shock_radius'], 'o-', label=case['label'])
        axes[1].plot(case['times_Myr'], case['gamma_eff'], 'o-', label=case['label'])
    axes[0].axhspan(0.5, 1.2, color='tab:green', alpha=0.08)
    axes[0].set_ylabel(r'$r_{\rm shock}/R_{200}$')
    axes[1].axhline(5.0 / 3.0, color='black', ls=':', label=r'$5/3$')
    axes[1].axhline(GAMMA_CRITICAL, color='red', ls='--', label=r'$10/7$')
    axes[1].set_ylabel(r'$\gamma_{\rm eff}$')
    for axis in axes:
        axis.set_xlabel('total time [Myr]')
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_title('Resolved virial-shock radius')
    axes[1].set_title('Birnboim--Dekel stability index')
    fig.suptitle(r'Boundary-driven halo-mass sequence, $\dot M=30\,M_\odot\,{\rm yr}^{-1}$')
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def main(run_cases=True):
    if run_cases:
        for config in CONFIGS:
            RUNNER.main(config)
    cases = [_case_diagnostics(config) for config in CONFIGS]
    output = EXAMPLE_DIR / 'outputs'
    figure = output / 'NFWBoundaryDrivenVirialShock1D_MassSequence.jpg'
    report = output / 'NFWBoundaryDrivenVirialShock1D_MassSequence.txt'
    _plot(cases, figure)
    _write_summary(cases, report)
    print(f'mass-sequence figure = {figure}')
    print(f'mass-sequence report = {report}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--skip-runs', action='store_true',
        help='regenerate the comparison from existing snapshots',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(run_cases=not args.skip_runs)
