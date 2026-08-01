import argparse
import os
import sys
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import unyt

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'),
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import radhydropy.io as rio
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name('stellar_wind_bubble1d.yaml')


def load_snapshots(runparams, ICparams, max_outputs=10, start_index=1):
    """Load example outputs into lightweight snapshot wrappers."""

    snapshots = []
    for outindex in range(start_index, max_outputs):
        outfilename = os.path.join(
            runparams['outdir'],
            runparams['outfileprefix'] + '_%03d' % outindex + '.hdf5',
        )
        snapshots.append(et.load_snapshot(outfilename, ICparams, runparams))
    return snapshots


def main(config_filename=DEFAULT_CONFIG, plot_only=False):
    et.set_plot_style()
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    runparams, ICparams = load_example_parameters(config_filename, rundir)
    eu.clean_previous_outputs(runparams)

    if not plot_only:
        code_units_obj = CodeUnits.from_mapping(runparams.get('CodeUnits'))
        ric = et.Simwrap(ICparams, code_units=code_units_obj)
        rio.writehdf5(ric, runparams['ICfilename'])
        mainrun = Rsim(runparams)
        mainrun.RunAll(outputtime=0)

    snapshots = load_snapshots(runparams, ICparams)

    profile_figure = et.make_profile_figure(snapshots, ICparams, runparams)
    profile_figure_filename = os.path.join(
        runparams['savedir'],
        'StellarWindBubble1D_profiles.jpg',
    )
    profile_figure.savefig(profile_figure_filename, dpi=200)
    plt.close(profile_figure)
    print('figure = %s' % profile_figure_filename)

    radius_figure = et.make_radius_figure(snapshots, ICparams, runparams)
    radius_figure_filename = os.path.join(
        runparams['savedir'],
        'StellarWindBubble1D_radius.jpg',
    )
    radius_figure.savefig(radius_figure_filename, dpi=200)
    plt.close(radius_figure)
    print('figure = %s' % radius_figure_filename)

    velocity_figure = et.make_velocity_figure(snapshots, ICparams, runparams)
    if velocity_figure is not None:
        velocity_figure_filename = os.path.join(
            runparams['savedir'],
            'StellarWindBubble1D_velocity.jpg',
        )
        velocity_figure.savefig(velocity_figure_filename, dpi=200)
        plt.close(velocity_figure)
        print('figure = %s' % velocity_figure_filename)

    pressure_figure = et.make_pressure_figure(snapshots, ICparams, runparams)
    if pressure_figure is not None:
        pressure_figure_filename = os.path.join(
            runparams['savedir'],
            'StellarWindBubble1D_pressure.jpg',
        )
        pressure_figure.savefig(pressure_figure_filename, dpi=200)
        plt.close(pressure_figure)
        print('figure = %s' % pressure_figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the spherical stellar-wind bubble example.')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='YAML file with runparams and ICparams.')
    parser.add_argument('--plot-only', action='store_true', help='Skip the hydro run and rebuild the figure from existing outputs.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, plot_only=args.plot_only)
