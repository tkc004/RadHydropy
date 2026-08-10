"""Fixed-density CMB Compton heating and cooling benchmark.

The example runs a hot cooling parcel and a cold heating parcel at fixed
redshift, then compares both source integrations with the analytic exponential
solution for Compton coupling to an isotropic CMB background.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
import yaml

import radhydropy.io as rio
from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
from radhydropy.thermo_networks.compton import cmb_compton_rate
from radhydropy.units import CodeUnits
import example_utils as eu
from tools import Simwrap


DEFAULT_CONFIG = Path(__file__).resolve().with_name('compton_cmb_heating1d.yaml')


def _analytic_temperature(
    time_s,
    initial_temperature,
    redshift,
    nH_cm3,
    neutral_fraction,
):
    """Return the fixed-density analytic CMB-coupling temperature."""
    cmb_temperature = 2.7255 * (1.0 + redshift)
    proton_mass_g = float(unyt.mp.to_value(unyt.g))
    rho_g_cm3 = nH_cm3 * proton_mass_g
    electron_fraction = 1.0 - neutral_fraction
    electron_density_cm3 = nH_cm3 * electron_fraction
    mu = 1.0 / (1.0 + electron_fraction)
    gamma = 5.0 / 3.0
    slope = float(
        cmb_compton_rate(
            np.array([0.0]),
            np.array([electron_density_cm3]),
            enabled=True,
            redshift=redshift,
        )[0]
    ) / cmb_temperature
    temperature_rate_coefficient = slope * (
        (gamma - 1.0) * mu * proton_mass_g
        / float(unyt.kb.to_value(unyt.erg / unyt.K))
        / rho_g_cm3
    )
    return cmb_temperature + (
        initial_temperature - cmb_temperature
    ) * np.exp(-temperature_rate_coefficient * time_s)


def _run_case(runparams, icparams, label, initial_temperature):
    case_params = dict(runparams)
    case_params['ICfilename'] = str(
        Path(runparams['outdir']) / f'ComptonCMBHeating1D_{label}_InitialCondition.hdf5'
    )
    case_icparams = dict(icparams)
    case_icparams['tempini'] = initial_temperature * unyt.K
    code_units = CodeUnits.from_mapping(case_params.get('CodeUnits'))

    ric = Simwrap(case_icparams, code_units)
    rio.writehdf5(ric, case_params['ICfilename'])

    sim = Rsim(case_params)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    history = sim.EvolveStaticThermochemistry(
        case_params['final_time'],
        case_params['evolution_timestep'],
        include_thermal_history=True,
    )
    myr_seconds = float((1.0 * unyt.Myr).to_value(unyt.s))
    time_s = np.asarray(history['time_Myr']) * myr_seconds
    temperature = np.asarray(history['mean_ionized_temp_K'])
    analytic = _analytic_temperature(
        time_s,
        initial_temperature,
        float(case_params['compton_cmb_redshift']),
        float(case_icparams['nHini'].to_value(1.0 / unyt.cm**3)),
        float(case_icparams['xHIini']),
    )
    return time_s, temperature, analytic


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename)
    runparams, icparams = load_example_parameters(config_filename)
    with config_filename.open() as config_file:
        raw_config = yaml.safe_load(config_file)
    cases = raw_config['cases']
    eu.clean_previous_outputs(runparams)

    histories = {}
    for label, initial_temperature in cases.items():
        histories[label] = _run_case(
            runparams,
            icparams,
            label,
            float(initial_temperature),
        )

    cmb_temperature = 2.7255 * (1.0 + runparams['compton_cmb_redshift'])
    figure_filename = Path(runparams['savedir']) / 'ComptonCMBHeating1D.jpg'
    figure_filename.parent.mkdir(parents=True, exist_ok=True)
    fig, (temperature_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.0),
        sharex=True,
        gridspec_kw={'height_ratios': (2.0, 1.0)},
    )
    for label, (time_s, temperature, analytic) in histories.items():
        time_myr = time_s / float((1.0 * unyt.Myr).to_value(unyt.s))
        temperature_axis.plot(time_myr, temperature, marker='o', ms=3, lw=0,
                               label=f'RadHydropy: {label}')
        temperature_axis.plot(time_myr, analytic, lw=1.8,
                              label=f'analytic: {label}')
        relative_error = np.abs((temperature - analytic) / analytic)
        error_axis.plot(time_myr, relative_error, marker='o', ms=3, lw=0,
                        label=label)

    temperature_axis.axhline(cmb_temperature, color='black', ls='--',
                             label=fr'$T_{{\rm CMB}}={cmb_temperature:.2f}$ K')
    temperature_axis.set_yscale('log')
    temperature_axis.set_ylabel('Temperature [K]')
    temperature_axis.grid(True, which='both', alpha=0.25)
    temperature_axis.legend(frameon=False, fontsize=8, ncol=2)
    error_axis.set_yscale('log')
    error_axis.set_xlabel('Time [Myr]')
    error_axis.set_ylabel('relative error')
    error_axis.grid(True, which='both', alpha=0.25)
    error_axis.legend(frameon=False)
    fig.suptitle(
        'CMB Compton heating and cooling '
        f'($z={runparams["compton_cmb_redshift"]:.1f}$)'
    )
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f'CMB temperature = {cmb_temperature:.6g} K')
    print(f'figure = {figure_filename}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the fixed-density CMB Compton benchmark.',
    )
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
