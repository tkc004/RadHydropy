"""Helpers for example scripts that read parameters from YAML files."""

from pathlib import Path

import unyt
import yaml


def _load_yaml_value(value):
    if isinstance(value, dict) and {'value', 'unit'} <= value.keys():
        return float(value['value']) * unyt.Unit(value['unit'])
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


def load_example_parameters(config_filename, rundir=None):
    """Load ``runparams`` and ``ICparams`` from an example YAML file."""
    config_filename = Path(config_filename)
    rundir = config_filename.parent.resolve()
    with config_filename.open() as config_file:
        config = yaml.safe_load(config_file)

    runparams = _load_yaml_value(config['runparams'])
    icparams = _load_yaml_value(config['ICparams'])
    for key in {'ICfilename', 'outdir', 'outputtimefilename', 'savedir'}:
        if key in runparams:
            runparams[key] = _resolve_path(runparams[key], rundir)
    return runparams, icparams
