"""Compare total gas energy with and without direct radiation pressure."""

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

HERE = Path(__file__).resolve().parent
EXAMPLE_DIR = HERE.parent
PACKAGE_DIR = HERE.parent.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from example_utils import load_nested_example_config
from radhydropy.units import CodeUnits


NO_PRESSURE_DIR = HERE.parent / 'DynamicStromgrenSpherePhotoheating20pc1D'
NO_PRESSURE_CONFIG = NO_PRESSURE_DIR / 'dynamic_stromgren_sphere_photoheating20pc1d.yaml'
PRESSURE_CONFIG = HERE / 'dynamic_stromgren_sphere_photoheating20pc_radiation_pressure1d.yaml'


def _load_tools(example_dir):
    module_name = f'_radhydropy_energy_tools_{example_dir.name}'
    spec = importlib.util.spec_from_file_location(module_name, example_dir / 'tools.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_config(config_filename):
    return load_nested_example_config(config_filename)


def _snapshot_energy(snapshot, config, tools):
    par, mesh, fluid = tools.load_output_state(snapshot, config)
    interior = tools.interior_slice(par)
    code = CodeUnits.from_mapping(par.CodeUnits)
    volume_cgs_cm3 = np.asarray(mesh.vol[interior], dtype=float) * float(
        (1.0 * code.volume_unit).to_value(unyt.cm**3)
    )
    pressure_cgs_erg_cm3 = np.asarray(fluid.pre_code[interior], dtype=float) * float(
        (1.0 * code.pressure_unit).to_value(unyt.erg / unyt.cm**3)
    )
    density_cgs_g_cm3 = np.asarray(fluid.rho_code[interior], dtype=float) * float(
        (1.0 * code.density_unit).to_value(unyt.g / unyt.cm**3)
    )
    velocity_cgs_cm_s = np.asarray(fluid.vel_code[interior], dtype=float) * float(
        (1.0 * code.velocity_unit).to_value(unyt.cm / unyt.s)
    )
    thermal = float(np.sum(pressure_cgs_erg_cm3 / (par.gamma - 1.0) * volume_cgs_cm3))
    kinetic = float(np.sum(0.5 * density_cgs_g_cm3 * velocity_cgs_cm_s**2 * volume_cgs_cm3))
    time_myr = float(
        np.asarray(fluid.time_code) * (1.0 * code.time_unit).to_value(unyt.Myr)
    )
    return time_myr, thermal, kinetic, thermal + kinetic


def _history(example_dir, config_filename):
    config = _load_config(config_filename)
    tools = _load_tools(example_dir)
    prefix = config['par']['output'].get('filename_prefix', 'Output')
    snapshots = sorted(example_dir.glob(f'{prefix}_*.hdf5'))
    if not snapshots:
        raise FileNotFoundError(f'No {prefix}_*.hdf5 snapshots found in {example_dir}')
    return np.asarray(
        [_snapshot_energy(snapshot, config, tools) for snapshot in snapshots],
        dtype=float,
    )


def main(no_pressure_dir=NO_PRESSURE_DIR, pressure_dir=HERE):
    no_pressure = _history(no_pressure_dir, no_pressure_dir / NO_PRESSURE_CONFIG.name)
    pressure = _history(pressure_dir, pressure_dir / PRESSURE_CONFIG.name)
    no_pressure = no_pressure[np.argsort(no_pressure[:, 0])]
    pressure = pressure[np.argsort(pressure[:, 0])]

    common_times = np.intersect1d(pressure[:, 0], no_pressure[:, 0])
    if common_times.size == 0:
        raise ValueError('The two examples have no snapshots at matching times.')
    pressure = pressure[np.isin(pressure[:, 0], common_times)]
    no_pressure = no_pressure[np.isin(no_pressure[:, 0], common_times)]
    pressure = pressure[np.argsort(pressure[:, 0])]
    no_pressure = no_pressure[np.argsort(no_pressure[:, 0])]

    energy_difference = pressure[:, 3] - no_pressure[:, 3]
    relative_difference = np.zeros_like(energy_difference)
    nonzero = no_pressure[:, 3] != 0.0
    relative_difference[nonzero] = energy_difference[nonzero] / no_pressure[nonzero, 3]

    figure = pressure_dir / 'DynamicStromgrenSpherePhotoheating20pcRadiationPressure1D_TotalGasEnergy.jpg'
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    axes[0].plot(pressure[:, 0], pressure[:, 3], 'o-', label='with radiation pressure')
    axes[0].plot(no_pressure[:, 0], no_pressure[:, 3], 'o-', label='without radiation pressure')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('total gas energy [erg]')
    axes[0].legend(frameon=False)
    axes[1].plot(pressure[:, 0], relative_difference, 'o-', color='tab:purple')
    axes[1].set_xlabel('time [Myr]')
    axes[1].set_ylabel(
        r'$(E_{\rm rad}-E_{\rm no\ rad})/E_{\rm no\ rad}$'
    )
    for axis in axes:
        axis.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    data = pressure_dir / 'DynamicStromgrenSpherePhotoheating20pcRadiationPressure1D_TotalGasEnergy.csv'
    np.savetxt(
        data,
        np.column_stack((
            common_times,
            no_pressure[:, 1], no_pressure[:, 2], no_pressure[:, 3],
            pressure[:, 1], pressure[:, 2], pressure[:, 3],
            energy_difference, relative_difference,
        )),
        delimiter=',',
        header=(
            'time_Myr,no_pressure_thermal_cgs_erg,no_pressure_kinetic_cgs_erg,no_pressure_total_cgs_erg,'
            'radiation_pressure_thermal_cgs_erg,radiation_pressure_kinetic_cgs_erg,'
            'radiation_pressure_total_cgs_erg,energy_difference_cgs_erg,relative_difference'
        ),
        comments='',
    )
    print('final radiation/no-radiation relative difference = %.6e' % relative_difference[-1])
    print('energy figure = %s' % figure)
    print('energy data = %s' % data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-pressure-dir', type=Path, default=NO_PRESSURE_DIR)
    parser.add_argument('--pressure-dir', type=Path, default=HERE)
    args = parser.parse_args()
    main(args.no_pressure_dir.resolve(), args.pressure_dir.resolve())
