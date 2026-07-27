from radhydropy.rsim import Rsim
import argparse
import unyt
import os
from pathlib import Path
import tempfile
import yaml

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'),
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import radhydropy.io as rio
import tools as et

DEFAULT_CONFIG = Path(__file__).resolve().with_name('advection1d.yaml')
PATH_PARAMS = {'ICfilename', 'outdir', 'savedir'}


def _load_yaml_value(value):
    if isinstance(value, dict) and {'value', 'unit'} <= value.keys():
        return value['value'] * unyt.Unit(value['unit'])
    if isinstance(value, dict):
        return {key: _load_yaml_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_load_yaml_value(item) for item in value]
    return value


def _resolve_path(value, rundir):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(rundir / path)


def load_parameters(config_filename=DEFAULT_CONFIG, rundir=None):
    rundir = Path.cwd().resolve() if rundir is None else Path(rundir).resolve()
    config_filename = Path(config_filename)
    with config_filename.open() as config_file:
        config = yaml.safe_load(config_file)

    runparams = _load_yaml_value(config['runparams'])
    ICparams = _load_yaml_value(config['ICparams'])
    for key in PATH_PARAMS:
        if key in runparams:
            runparams[key] = _resolve_path(runparams[key], rundir)
    return runparams, ICparams


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    runparams, ICparams = load_parameters(config_filename, rundir)

    ric = et.Simwrap(ICparams)
    rio.writehdf5(ric,runparams['ICfilename'])
    mainrun = Rsim(runparams)
    mainrun.RunAll(outputtime=0)
    ax = plt.gca()
    for outindex in range(0,10,5):
        outfilename = os.path.join(
            runparams['outdir'],
            runparams['outfileprefix']+'_%03d'%outindex+'.hdf5',
        )
        et.ReadandPlot(
            outfilename,
            ICparams,
            runparams,
            ls='none',
            marker='o',
            mfc='none',
            markevery=10,
            color=next(ax._get_lines.prop_cycler)['color'],
        )
    figure_filename = os.path.join(runparams['savedir'], 'Advection1D.jpg')
    plt.tight_layout()
    plt.savefig(figure_filename, dpi=200)
    plt.close()
    print('figure = %s' % figure_filename)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the 1D advection example.')
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG,
        help='YAML file containing runparams and ICparams.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
