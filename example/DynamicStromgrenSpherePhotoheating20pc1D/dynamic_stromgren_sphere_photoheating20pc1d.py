"""Photoheated Stromgren sphere in a 20 pc, 100 cm^-3 cloud.

This is a compact variant of the DynamicStromgrenSpherePhotoheating1D
example.  It uses the same tested workflow and helper implementation while
keeping its configuration and generated outputs separate.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / 'example'
TEMPLATE_DIR = EXAMPLE_ROOT / 'DynamicStromgrenSpherePhotoheating1D'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import example_utils as eu


def _load_template_runner():
    module_name = '_radhydropy_dynamic_stromgren_template'
    spec = importlib.util.spec_from_file_location(
        module_name,
        TEMPLATE_DIR / 'dynamic_stromgren_sphere_photoheating1d.py',
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
            'dynamic_stromgren_sphere_photoheating20pc1d.yaml'
        )
    template.main(config_filename)

    config_filename = Path(config_filename).resolve()
    config = eu.load_nested_example_config(config_filename)
    runparams = eu.runtime_parameters(config)
    output_dir = Path(runparams['output']['directory'])
    output_files = sorted(output_dir.glob(f"{runparams['output'].get('filename_prefix', 'Output')}_*.hdf5"))
    if not output_files:
        raise FileNotFoundError(f'No output HDF5 files found in {output_dir}')
    rhd_csv_filename = output_dir / 'radial_profile_rhd.csv'
    eu.write_radial_profile_csv(output_files[-1], rhd_csv_filename)
    print('RHD profile CSV = %s' % rhd_csv_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the 20 pc dynamic photoheated Stromgren sphere example.',
    )
    parser.add_argument(
        '--config',
        default=Path(__file__).resolve().with_name(
            'dynamic_stromgren_sphere_photoheating20pc1d.yaml'
        ),
        help='YAML file containing runparams and ICparams.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
