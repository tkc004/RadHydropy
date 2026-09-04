"""HM12 PIE radiative colliding-flow shock-tube benchmark."""

import argparse
import copy
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
from example_utils import load_nested_example_config
from radhydropy.rsim import Rsim
from radhydropy.thermo_networks.pie import MetalPIETable
from radhydropy.units import CodeUnits
import example_utils as eu
from tools import (
    Simwrap,
    cooling_length_estimate,
    load_snapshot,
    strong_shock_expectation,
    PROTON_MASS_G,
)


DEFAULT_CONFIG = EXAMPLE_DIR / 'pie_radiative_shock_tube_1d.yaml'
SECONDS_PER_MYR = (1.0 * unyt.Myr).to_value(unyt.s)
KPC_CM = (1.0 * unyt.kpc).to_value(unyt.cm)


def _run_case(
    config, label, metallicity, hydrogen_density, table,
    adiabatic=False,
):
    par_config = config['par']
    thermo = par_config['thermochemistry']
    initial = config['initial_condition']
    case = copy.deepcopy(par_config)
    output_dir = EXAMPLE_DIR / 'outputs' / label
    output_dir.mkdir(parents=True, exist_ok=True)
    case['simulation']['name'] = label
    case['simulation']['initial_condition_filename'] = str(
        output_dir / f'InitialCondition_{label}.hdf5'
    )
    case['output'].update({
        'directory': str(output_dir),
        'savedir': str(output_dir),
        'filename_prefix': f'Output_{label}',
    })
    output_prefix = case['output']['filename_prefix']
    case['thermochemistry']['metallicity'] = metallicity
    case_initial = dict(initial)
    case_initial['hydrogen_density'] = hydrogen_density * unyt.cm**-3
    eu.clean_previous_outputs(case)
    code_units = CodeUnits.from_mapping(case['units']['CodeUnits'])
    case_config = {'par': case, 'initial_condition': case_initial}
    initial = Simwrap(
        case_config, code_units,
        case['thermochemistry']['hydrogen_mass_fraction'],
    )
    rio.writehdf5(initial, case['simulation']['initial_condition_filename'])
    sim = Rsim(case)
    sim.RunAll(outputtime=0, mode='hydro' if adiabatic else 'hydro_sources')
    snapshots = sorted(output_dir.glob(f'{output_prefix}_*.hdf5'))
    if len(snapshots) < 2:
        raise RuntimeError(f'expected snapshots in {output_dir}')
    return {
        'label': label,
        'metallicity': metallicity,
        'adiabatic': adiabatic,
        'snapshots': snapshots,
        'initial_density_cgs_g_cm3': (
            case_initial['hydrogen_density'].to_value('cm**-3')
            * PROTON_MASS_G / case['thermochemistry']['hydrogen_mass_fraction']
        ),
        'upstream_velocity_cgs_cm_s': case_initial['collision_velocity'],
        'mu': case_initial['mean_molecular_weight'],
    }


def _shock_diagnostics(result, table, config):
    thermo = config['par']['thermochemistry']
    gamma = float(config['par']['hydrodynamics']['gamma'])
    mu = float(config['initial_condition']['mean_molecular_weight'])
    snapshot = load_snapshot(result['snapshots'][-1])
    shock_snapshot = load_snapshot(result['snapshots'][1])
    density = shock_snapshot['density_cgs_g_cm3']
    temperature = shock_snapshot['temperature_cgs_K']
    velocity = shock_snapshot['velocity_cgs_cm_s']
    boundary = shock_snapshot['boundary_cgs_cm']
    centers = 0.5 * (boundary[1:] + boundary[:-1])
    center = 0.5 * np.max(boundary)
    right = (centers > center) & (centers < center + 0.45 * np.max(boundary))
    gradient = np.abs(np.gradient(np.log(np.maximum(density, 1.0e-99)), centers))
    shock_index = np.flatnonzero(right)[np.argmax(gradient[right])]
    buffer_cm = 3.0 * (boundary[1] - boundary[0])
    upstream_slice = right & (centers > centers[shock_index] + buffer_cm)
    post_slice = (
        centers > center
    ) & (centers < centers[shock_index] - buffer_cm)
    if not np.any(post_slice):
        post_slice = (centers > center) & (centers < centers[shock_index])
    upstream_density = float(np.median(density[upstream_slice]))
    upstream_velocity = float(np.median(np.abs(velocity[upstream_slice])))
    immediate_start = max(0, shock_index - 3)
    immediate_stop = shock_index
    immediate_slice = np.zeros_like(right, dtype=bool)
    immediate_slice[immediate_start:immediate_stop] = True
    post_density = float(np.median(density[immediate_slice]))
    post_temperature = float(np.median(temperature[immediate_slice]))
    compression = post_density / upstream_density
    expected_compression, post_velocity, expected_temperature = strong_shock_expectation(
        gamma, upstream_velocity, mu
    )
    if result['adiabatic']:
        cooling_length = np.nan
        measured_length = np.nan
    else:
        cooling_length = cooling_length_estimate(
            table, post_temperature, post_density,
            float(thermo['hydrogen_mass_fraction']), mu,
            gamma, result['metallicity'], float(thermo['metal_pie_redshift']),
            post_velocity,
        )
    final_boundary = snapshot['boundary_cgs_cm']
    final_centers = 0.5 * (final_boundary[1:] + final_boundary[:-1])
    final_centers -= center
    hot_layer = (
        (final_centers > 0.0)
        & (final_centers < centers[shock_index] - center)
        & (snapshot['temperature_cgs_K'] > 0.9 * post_temperature)
    )
    measured_length = (
        centers[shock_index] - np.min(final_centers[hot_layer])
        if np.any(hot_layer) else np.nan
    )
    result.update({
        'snapshot': snapshot,
        'shock_radius_cgs_cm': centers[shock_index] - 0.5 * np.max(boundary),
        'upstream_density_cgs_g_cm3': upstream_density,
        'upstream_velocity_cgs_cm_s': upstream_velocity,
        'post_density_cgs_g_cm3': post_density,
        'post_temperature_cgs_K': post_temperature,
        'compression': compression,
        'expected_compression': expected_compression,
        'expected_post_temperature_cgs_K': expected_temperature,
        'cooling_length_expected_cm': cooling_length,
        'cooling_length_measured_cm': measured_length,
    })
    return result


def _plot(results, filename):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex='col')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for index, result in enumerate(results):
        data = result['snapshot']
        x_kpc = (
            0.5 * (data['boundary_cgs_cm'][1:] + data['boundary_cgs_cm'][:-1])
            / KPC_CM
        )
        x_kpc -= 0.5 * np.max(x_kpc)
        style = '--' if result['adiabatic'] else '-'
        label = result['label']
        color = colors[index % len(colors)]
        result['_plot_color'] = color
        axes[0, 0].plot(x_kpc, data['density_cgs_g_cm3'], style, color=color, label=label)
        axes[0, 1].plot(x_kpc, data['temperature_cgs_K'], style, color=color, label=label)
        axes[1, 0].plot(x_kpc, data['velocity_cgs_cm_s'] / 1.0e5, style, color=color, label=label)
        axes[1, 1].plot(
            x_kpc, data['temperature_cgs_K'] / result['post_temperature_cgs_K'],
            style, color=color, label=label,
        )
    for result in results:
        if result['adiabatic']:
            continue
        shock_kpc = result['shock_radius_cgs_cm'] / KPC_CM
        expected_kpc = result['cooling_length_expected_cm'] / KPC_CM
        measured_kpc = result['cooling_length_measured_cm'] / KPC_CM
        color = result['_plot_color']
        if np.isfinite(expected_kpc):
            axes[1, 1].axvspan(
                shock_kpc - expected_kpc, shock_kpc,
                color=color, alpha=0.10, lw=0,
            )
        if np.isfinite(measured_kpc):
            axes[1, 1].plot(
                [shock_kpc - measured_kpc, shock_kpc], [0.035, 0.035],
                color=color, lw=3.0, solid_capstyle='butt',
            )
    axes[0, 0].set_ylabel(r'$\rho$ [g cm$^{-3}$]')
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_ylabel('$T$ [K]')
    axes[0, 1].set_yscale('log')
    axes[1, 0].set_ylabel('$v$ [km s$^{-1}$]')
    axes[1, 1].set_ylabel(r'$T/T_{\rm post}$')
    axes[1, 1].set_title('shaded: predicted cooling length; bar: measured hot layer')
    axes[1, 1].set_yscale('log')
    for axis in axes.flat:
        axis.set_xlabel('position [kpc]')
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=7)
    fig.suptitle('HM12 PIE colliding-flow radiative shock tube')
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    nested = eu.load_nested_example_config(config_filename)
    par_config = nested['par']
    config = nested
    thermo = config['par']['thermochemistry']
    table_path = (config_filename.parent / thermo['metal_pie_table_filename']).resolve()
    thermo['metal_pie_table_filename'] = str(table_path)
    table = MetalPIETable(table_path)
    if not table.is_hm12_uv_background:
        raise ValueError('the example requires an HM12 UV-background table')
    cases = [
        ('PIE_Z1_nH1e-3', 1.0, 1.0e-3, False),
        ('PIE_Z0p1_nH1e-3', 0.1, 1.0e-3, False),
        ('PIE_Z1_nH1e-2', 1.0, 1.0e-2, False),
        ('adiabatic_control', 1.0, 1.0e-3, True),
    ]
    results = []
    for label, metallicity, hydrogen_density, adiabatic in cases:
        result = _run_case(
            config, label, metallicity, hydrogen_density,
            table, adiabatic
        )
        if adiabatic:
            result['metallicity'] = 0.0
        results.append(_shock_diagnostics(result, table, config))
    report = EXAMPLE_DIR / 'PIERadiativeShockTube1D_ShockReport.txt'
    with open(report, 'w', encoding='utf-8') as handle:
        handle.write(
            'case metallicity shock_position_kpc upstream_velocity_km_s '
            'compression expected_compression post_temperature_cgs_K '
            'expected_post_temperature_cgs_K cooling_length_kpc '
            'measured_hot_layer_kpc\n'
        )
        for result in results:
            handle.write(
                '%s %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (
                    result['label'], result['metallicity'],
                    result['shock_radius_cgs_cm'] / KPC_CM,
                    result['upstream_velocity_cgs_cm_s'] / 1.0e5,
                    result['compression'], result['expected_compression'],
                    result['post_temperature_cgs_K'],
                    result['expected_post_temperature_cgs_K'],
                    result['cooling_length_expected_cm'] / KPC_CM,
                    result['cooling_length_measured_cm'] / KPC_CM,
                )
            )
    _plot(results, EXAMPLE_DIR / 'PIERadiativeShockTube1D.jpg')
    for result in results:
        print(
            '%s: compression=%.4g (strong-shock %.4g), '
            'Lcool=%.4g kpc' % (
                result['label'], result['compression'],
                result['expected_compression'],
                result['cooling_length_expected_cm'] / KPC_CM,
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
