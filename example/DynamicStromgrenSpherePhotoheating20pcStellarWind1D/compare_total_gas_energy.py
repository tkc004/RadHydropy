"""Compare total gas energy with and without the stellar wind.

The comparison uses matched snapshot times when available and integrates over
physical cells only.  Gas energy is the sum of thermal and radial kinetic
energy.
"""

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from radhydropy.example_config import load_example_parameters
from radhydropy.units import CodeUnits


HERE = Path(__file__).resolve().parent
NO_WIND_DIR = HERE.parent / 'DynamicStromgrenSpherePhotoheating20pc1D'
NO_WIND_CONFIG = NO_WIND_DIR / 'dynamic_stromgren_sphere_photoheating20pc1d.yaml'
WIND_CONFIG = HERE / 'dynamic_stromgren_sphere_photoheating20pc_stellar_wind1d.yaml'
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))


def _load_tools(example_dir):
    module_name = f'_radhydropy_energy_tools_{example_dir.name}'
    spec = importlib.util.spec_from_file_location(
        module_name, example_dir / 'tools.py'
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_config(config_filename):
    runparams, icparams = load_example_parameters(config_filename)
    return {**runparams, **icparams}


def _snapshot_energy(snapshot, config, tools):
    par, mesh, fluid = tools.load_output_state(snapshot, config)
    interior = tools.interior_slice(par)
    code = CodeUnits.from_mapping(par.CodeUnits)
    volume_cm3 = np.asarray(mesh.vol[interior], dtype=float) * float(
        (1.0 * code.volume_unit).to_value(unyt.cm**3)
    )
    pressure_erg_cm3 = np.asarray(fluid.pre_code[interior], dtype=float) * float(
        (1.0 * code.pressure_unit).to_value(unyt.erg / unyt.cm**3)
    )
    density_g_cm3 = np.asarray(fluid.rho_code[interior], dtype=float) * float(
        (1.0 * code.density_unit).to_value(unyt.g / unyt.cm**3)
    )
    velocity_cm_s = np.asarray(fluid.vel_code[interior], dtype=float) * float(
        (1.0 * code.velocity_unit).to_value(unyt.cm / unyt.s)
    )
    thermal = float(np.sum(pressure_erg_cm3 / (par.gamma - 1.0) * volume_cm3))
    kinetic = float(np.sum(0.5 * density_g_cm3 * velocity_cm_s**2 * volume_cm3))
    time_myr = float(
        np.asarray(fluid.time) * (1.0 * code.time_unit).to_value(unyt.Myr)
    )
    return time_myr, thermal, kinetic, thermal + kinetic


def _history(example_dir, config_filename):
    config = _load_config(config_filename)
    tools = _load_tools(example_dir)
    prefix = config.get('outfileprefix', 'Output')
    snapshots = sorted(example_dir.glob(f'{prefix}_*.hdf5'))
    if not snapshots:
        raise FileNotFoundError(f'No {prefix}_*.hdf5 snapshots found in {example_dir}')
    return np.asarray(
        [_snapshot_energy(snapshot, config, tools) for snapshot in snapshots],
        dtype=float,
    )


def main(no_wind_dir=NO_WIND_DIR, wind_dir=HERE):
    no_wind = _history(no_wind_dir, no_wind_dir / NO_WIND_CONFIG.name)
    wind = _history(wind_dir, wind_dir / WIND_CONFIG.name)
    no_wind = no_wind[np.argsort(no_wind[:, 0])]
    wind = wind[np.argsort(wind[:, 0])]

    common_times = np.intersect1d(wind[:, 0], no_wind[:, 0])
    if common_times.size == 0:
        raise ValueError('The two examples have no snapshots at matching times.')
    wind = wind[np.isin(wind[:, 0], common_times)]
    no_wind = no_wind[np.isin(no_wind[:, 0], common_times)]
    wind = wind[np.argsort(wind[:, 0])]
    no_wind = no_wind[np.argsort(no_wind[:, 0])]
    energy_difference = wind[:, 3] - no_wind[:, 3]
    relative_difference = np.zeros_like(energy_difference)
    nonzero = no_wind[:, 3] != 0.0
    relative_difference[nonzero] = energy_difference[nonzero] / no_wind[nonzero, 3]

    figure = wind_dir / 'DynamicStromgrenSpherePhotoheating20pcStellarWind1D_TotalGasEnergy.jpg'
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    axes[0].plot(wind[:, 0], wind[:, 3], 'o-', label='with stellar wind')
    axes[0].plot(no_wind[:, 0], no_wind[:, 3], 'o-', label='without stellar wind')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('total gas energy [erg]')
    axes[0].legend(frameon=False)
    axes[1].plot(wind[:, 0], relative_difference, 'o-', color='tab:purple')
    axes[1].set_ylabel(
        r'$(E_{\rm wind}-E_{\rm no\ wind})/E_{\rm no\ wind}$'
    )
    axes[1].set_xlabel('time [Myr]')
    for axis in axes:
        axis.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    data = wind_dir / 'DynamicStromgrenSpherePhotoheating20pcStellarWind1D_TotalGasEnergy.csv'
    np.savetxt(
        data,
        np.column_stack((
            common_times,
            no_wind[:, 1], no_wind[:, 2], no_wind[:, 3],
            wind[:, 1], wind[:, 2], wind[:, 3],
            energy_difference, relative_difference,
        )),
        delimiter=',',
        header=(
            'time_Myr,no_wind_thermal_erg,no_wind_kinetic_erg,no_wind_total_erg,'
            'wind_thermal_erg,wind_kinetic_erg,wind_total_erg,'
            'wind_minus_no_wind_erg,relative_difference'
        ),
        comments='',
    )
    print('final wind/no-wind total-energy difference = %.6e erg' % energy_difference[-1])
    print('final relative difference = %.6e' % relative_difference[-1])
    print('energy figure = %s' % figure)
    print('energy data = %s' % data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-wind-dir', type=Path, default=NO_WIND_DIR)
    parser.add_argument('--wind-dir', type=Path, default=HERE)
    args = parser.parse_args()
    main(args.no_wind_dir.resolve(), args.wind_dir.resolve())
