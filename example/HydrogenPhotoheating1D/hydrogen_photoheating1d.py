"""Optically thin hydrogen photoheating and recombination parcel.

An initially neutral pure-hydrogen parcel with fixed total density is exposed
to a spatially uniform ionizing radiation field. The radiation is treated as
optically thin, so the photon density is fixed while the source is on and set
to zero when the source switches off. The run writes HDF5 snapshots, reloads
them, and plots the thermal and ionization history from those outputs.
"""

import argparse
import os
import sys
from pathlib import Path
import tempfile

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unyt

from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
import radhydropy.io as rio
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name('hydrogen_photoheating1d.yaml')


def load_parameters(config_filename=DEFAULT_CONFIG, rundir=None):
    config_filename = Path(config_filename)
    runparams, ICparams = load_example_parameters(config_filename, rundir)
    return runparams, ICparams


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    runparams, ICparams = load_parameters(config_filename, rundir)

    reference = et.reference_values(
        runparams['photon_flux'],
        ICparams['nHini'],
        runparams['excess_photoionization_energy'],
        runparams['sigma_gamma'],
        runparams['thermal_equilibrium_timescale'],
    )

    ric = et.Simwrap(ICparams)
    rio.writehdf5(ric, runparams['ICfilename'])

    sim = Rsim(runparams)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.RunHydrogenPhotoheating(
        runparams['source_switch_time'],
        reference['photon_density_on'],
        outputtime=0,
    )

    outputfiles = et.output_files(
        runparams['outdir'],
        runparams['outfileprefix'],
    )
    history = et.load_history_from_outputs(
        outputfiles,
        ICparams,
        runparams['noghost'],
    )

    figure_filename = Path(runparams['savedir']) / 'HydrogenPhotoheating1D.jpg'
    xHI_reference = et.save_history_plot(history, str(figure_filename), reference)

    print('Hydrogen photoheating example finished')
    print('time = %.3e yr' % et.time_value(sim, unyt.yr))
    print('mean temperature = %.3e K' % et.mean_temperature(sim).to_value(unyt.K))
    print('mean neutral fraction = %.3e' % et.mean_neutral_fraction(sim))
    print(
        'mean photon number density = %.3e cm^-3'
        % et.mean_photon_number_density(sim).to_value(1.0 / unyt.cm**3)
    )
    print(
        'sigma_gamma = %.3e cm^2'
        % runparams['sigma_gamma'].to_value(unyt.cm**2)
    )
    print(
        'epsilon_gamma = %.3e eV'
        % runparams['excess_photoionization_energy'].to_value(unyt.eV)
    )
    print(
        'photoionization equilibrium temperature = %.3e K'
        % reference['photoionization_temperature'].to_value(unyt.K)
    )
    print(
        'thermal equilibrium reference temperature = %.3e K'
        % reference['thermal_temperature'].to_value(unyt.K)
    )
    print(
        'ionization time = %.3e yr'
        % xHI_reference['ionization_timescale'].to_value(unyt.yr)
    )
    print(
        'recombination time at T_ion = %.3e yr'
        % xHI_reference['recombination_timescale'].to_value(unyt.yr)
    )
    print(
        'ionization equilibrium neutral fraction = %.3e'
        % xHI_reference['xHI']
    )
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the optically thin hydrogen photoheating example.',
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG,
        help='YAML file containing runparams and ICparams.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
