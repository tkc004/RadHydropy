"""Photoheated 20 pc Stromgren sphere with a central stellar wind."""

import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / 'example'
TEMPLATE_DIR = EXAMPLE_ROOT / 'DynamicStromgrenSpherePhotoheating20pc1D'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


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


def main(config_filename=None):
    template = _load_template_runner()
    if config_filename is None:
        config_filename = Path(__file__).resolve().with_name(
            'dynamic_stromgren_sphere_photoheating20pc_stellar_wind1d.yaml'
        )
    template.main(config_filename)

    with Path(config_filename).open(encoding='utf-8') as handle:
        runparams = yaml.safe_load(handle)['runparams']
    output_dir = (Path(config_filename).parent / runparams.get('outdir', '.')).resolve()
    old_csv = output_dir / 'radial_profile_rhd.csv'
    wind_csv = output_dir / 'radial_profile_rhd_wind.csv'
    if old_csv.exists():
        old_csv.replace(wind_csv)
    print('RHD wind profile CSV = %s' % wind_csv)


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
