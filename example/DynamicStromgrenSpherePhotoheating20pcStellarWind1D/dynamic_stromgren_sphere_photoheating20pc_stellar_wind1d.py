"""Photoheated 20 pc Stromgren sphere with a central stellar wind."""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / 'example'
TEMPLATE_DIR = EXAMPLE_ROOT / 'DynamicStromgrenSpherePhotoheating20pc1D'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from radhydropy.units import CodeUnits
import tools as et


def _load_template_runner():
    module_name = '_radhydropy_dynamic_stromgren_wind_template'
    spec = importlib.util.spec_from_file_location(
        module_name,
        TEMPLATE_DIR / 'dynamic_stromgren_sphere_photoheating20pc1d.py',
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _pressure_diagnostic(snapshot, config):
    """Return shell wind pressure, photoheated gas pressure, and shell radius."""
    par, mesh, fluid = et.load_output_state(snapshot, config)
    interior = et.interior_slice(par)
    radius_pc = et._to_kpc(mesh.coordinate[interior], par) * 1000.0
    density = et._to_number_density(fluid.rho_code[interior], par)
    pressure = et._to_pressure(fluid.pre_code[interior], par)
    xhi = np.asarray(fluid.xHI[interior], dtype=float)

    # The wind shell is the strongest density peak outside the injection cell.
    shell_index = int(np.argmax(density))
    shell_radius_pc = float(radius_pc[shell_index])
    if shell_radius_pc <= 0.0:
        wind_pressure = 0.0
    else:
        mdot = config['wind_mass_loss_rate'].to_value(unyt.g / unyt.s)
        wind_velocity = config['wind_velocity'].to_value(unyt.cm / unyt.s)
        shell_radius_cm = shell_radius_pc * (1.0 * unyt.pc).to_value(unyt.cm)
        wind_pressure = mdot * wind_velocity / (
            4.0 * np.pi * shell_radius_cm**2
        )

    code = CodeUnits.from_mapping(par.CodeUnits)
    volume_cm3 = np.asarray(mesh.vol[interior], dtype=float) * float(
        (1.0 * code.volume_unit).to_value(unyt.cm**3)
    )
    # The photoheated ambient gas lies between the wind cavity and the shell.
    # Exclude the shocked wind interior and the dense shell itself.
    ambient_ionized = (xhi < 0.5) & (radius_pc < shell_radius_pc)
    if not np.any(ambient_ionized):
        ambient_ionized = xhi < 0.5
    weighted_volume = float(np.sum(volume_cm3[ambient_ionized]))
    gas_pressure = (
        float(np.sum(pressure[ambient_ionized] * volume_cm3[ambient_ionized]) / weighted_volume)
        if weighted_volume > 0.0
        else 0.0
    )
    time_myr = float(np.asarray(et._to_myr(fluid.time, par)))
    return time_myr, wind_pressure, gas_pressure, shell_radius_pc


def pressure_diagnostic_from_profile(profile, config):
    """Estimate the same pressures from a saved radial-profile CSV."""
    fields = np.genfromtxt(profile, delimiter=',', names=True)
    radius_pc = np.asarray(fields['RADIUS_PC'], dtype=float)
    density = np.asarray(fields['DENSITY_CM3'], dtype=float)
    temperature = np.asarray(fields['TEMP_K'], dtype=float)
    shell_index = 2 + int(np.argmax(density[2:]))
    shell_radius_pc = float(radius_pc[shell_index])
    mdot = config['wind_mass_loss_rate'].to_value(unyt.g / unyt.s)
    wind_velocity = config['wind_velocity'].to_value(unyt.cm / unyt.s)
    shell_radius_cm = shell_radius_pc * (1.0 * unyt.pc).to_value(unyt.cm)
    wind_pressure = mdot * wind_velocity / (4.0 * np.pi * shell_radius_cm**2)
    photoheated = (np.arange(radius_pc.size) >= 2) & (
        np.arange(radius_pc.size) < shell_index
    ) & (temperature > 500.0)
    gas_pressure = float(
        np.mean(density[photoheated] * unyt.kb.to_value(unyt.erg / unyt.K) * temperature[photoheated])
    ) if np.any(photoheated) else 0.0
    time_myr = float(Path(profile).stem.rsplit('_', 1)[-1].replace('Myr', ''))
    return time_myr, wind_pressure, gas_pressure, shell_radius_pc


def save_pressure_ratio_plot(diagnostics, output_dir):
    """Save pressure curves and their ratio from diagnostic rows."""
    diagnostics = np.asarray(diagnostics, dtype=float)
    diagnostics = diagnostics[np.argsort(diagnostics[:, 0])]
    times, wind_pressure, gas_pressure, shell_radius = diagnostics.T
    pressure_ratio = np.zeros_like(wind_pressure)
    nonzero = gas_pressure > 0.0
    pressure_ratio[nonzero] = wind_pressure[nonzero] / gas_pressure[nonzero]
    figure_stem = 'DynamicStromgrenSpherePhotoheating20pcStellarWind1D'
    pressure_figure = Path(output_dir) / f'{figure_stem}_PressureRatio.jpg'
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.5), sharex=True)
    axes[0].plot(times, wind_pressure, label='wind ram pressure at shell')
    axes[0].plot(times, gas_pressure, label='photoheated-gas thermal pressure')
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'pressure [dyn cm$^{-2}$]')
    axes[0].legend(frameon=False)
    axes[1].plot(times, pressure_ratio, color='tab:purple', marker='o')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('time [Myr]')
    axes[1].set_ylabel(r'$P_{\rm wind}/P_{\rm gas}$')
    for axis in axes:
        axis.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(pressure_figure, dpi=180)
    plt.close(fig)
    pressure_csv = Path(output_dir) / f'{figure_stem}_PressureRatio.csv'
    np.savetxt(
        pressure_csv,
        np.column_stack((times, shell_radius, wind_pressure, gas_pressure, pressure_ratio)),
        delimiter=',',
        header='time_Myr,shell_radius_pc,wind_pressure_dyn_cm2,gas_pressure_dyn_cm2,pressure_ratio',
        comments='',
    )
    return pressure_figure, pressure_csv, pressure_ratio


def main(config_filename=None):
    template = _load_template_runner()
    if config_filename is None:
        config_filename = Path(__file__).resolve().with_name(
            'dynamic_stromgren_sphere_photoheating20pc_stellar_wind1d.yaml'
        )
    template.main(config_filename)

    nested_config = et.eu.load_nested_example_config(config_filename)
    runparams = et.eu.runtime_parameters(nested_config)
    output_dir = Path(runparams['output']['directory'])
    old_csv = output_dir / 'radial_profile_rhd.csv'
    wind_csv = output_dir / 'radial_profile_rhd_wind.csv'
    if old_csv.exists():
        old_csv.replace(wind_csv)
    print('RHD wind profile CSV = %s' % wind_csv)

    output_files = sorted(output_dir.glob(f"{runparams['output'].get('filename_prefix', 'Output')}_*.hdf5"))
    # Do not use et.load_parameters here: it intentionally removes stale
    # snapshots for a new run. This is postprocessing, so preserve the files
    # just written by the simulation.
    run_parameters, ic_parameters = et.load_parameters(config_filename)
    config = {**run_parameters, **ic_parameters}
    snapshots = [_pressure_diagnostic(filename, config) for filename in output_files]
    diagnostics = np.asarray(snapshots, dtype=float)
    times = diagnostics[:, 0]
    wind_pressure = diagnostics[:, 1]
    gas_pressure = diagnostics[:, 2]
    shell_radius = diagnostics[:, 3]
    pressure_ratio = np.zeros_like(wind_pressure)
    nonzero = gas_pressure > 0.0
    pressure_ratio[nonzero] = wind_pressure[nonzero] / gas_pressure[nonzero]

    figure_stem = 'DynamicStromgrenSpherePhotoheating20pcStellarWind1D'
    pressure_figure = output_dir / f'{figure_stem}_PressureRatio.jpg'
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.5), sharex=True)
    axes[0].plot(times, wind_pressure, label='wind ram pressure at shell')
    axes[0].plot(times, gas_pressure, label='photoheated-gas thermal pressure')
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'pressure [dyn cm$^{-2}$]')
    axes[0].legend(frameon=False)
    axes[1].plot(times, pressure_ratio, color='tab:purple')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('time [Myr]')
    axes[1].set_ylabel(r'$P_{\rm wind}/P_{\rm gas}$')
    for axis in axes:
        axis.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(pressure_figure, dpi=180)
    plt.close(fig)

    pressure_csv = output_dir / f'{figure_stem}_PressureRatio.csv'
    np.savetxt(
        pressure_csv,
        np.column_stack((times, shell_radius, wind_pressure, gas_pressure, pressure_ratio)),
        delimiter=',',
        header='time_Myr,shell_radius_pc,wind_pressure_dyn_cm2,gas_pressure_dyn_cm2,pressure_ratio',
        comments='',
    )
    print('final wind/gas pressure ratio = %.6e' % pressure_ratio[-1])
    print('pressure ratio figure = %s' % pressure_figure)
    print('pressure ratio data = %s' % pressure_csv)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a photoheated Stromgren sphere with a stellar wind.',
    )
    parser.add_argument(
        '--config',
        default=Path(__file__).resolve().with_name(
            'dynamic_stromgren_sphere_photoheating20pc_stellar_wind1d.yaml'
        ),
        help='YAML file containing runparams and ICparams.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
