"""Load the canonical nested configuration used by RadHydropy examples."""

from pathlib import Path

import unyt
import yaml

from radhydropy.radiation_spectrum import (
    load_radiation_spectrum,
    resolve_spectrum_filename,
)


def _load_yaml_value(value):
    """Convert YAML ``value``/``unit`` mappings into unyt quantities."""
    if isinstance(value, dict) and {'value', 'unit'} <= value.keys():
        return float(value['value']) * unyt.Unit(value['unit'])
    if isinstance(value, dict):
        return {key: _load_yaml_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_load_yaml_value(item) for item in value]
    return value


def _resolve_path(value, base_directory):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(Path(base_directory) / path)


def load_example_config(config_filename):
    """Load a complete nested example configuration.

    The result always contains the independent ``par``, ``initial_condition``,
    and ``example`` sections.  Paths are resolved relative to the YAML file,
    and configured radiation spectra are loaded into ``par``.
    """
    config_filename = Path(config_filename).resolve()
    with config_filename.open(encoding='utf-8') as config_file:
        raw = yaml.safe_load(config_file)
    config = _load_yaml_value(raw)
    required_sections = {'par', 'initial_condition', 'example'}
    if not isinstance(config, dict) or not required_sections.issubset(config):
        raise ValueError(
            "nested example configuration requires 'par', "
            "'initial_condition', and 'example' sections"
        )

    par = config['par']
    initial_condition = config['initial_condition']
    if 'mesh' in par and 'grid_cells' not in par['mesh']:
        if 'grid_cells' in initial_condition:
            par['mesh']['grid_cells'] = initial_condition['grid_cells']

    simulation = par.get('simulation', {})
    output = par.get('output', {})
    if 'initial_condition_filename' in simulation:
        simulation['initial_condition_filename'] = _resolve_path(
            simulation['initial_condition_filename'], config_filename.parent
        )
    for key in ('directory', 'savedir', 'time_list_filename'):
        if key in output:
            output[key] = _resolve_path(output[key], config_filename.parent)

    spectrum_filename = par.get('radiation', {}).get('radiation_spectrum_filename')
    if spectrum_filename:
        par['radiation'].update(load_radiation_spectrum(
            resolve_spectrum_filename(spectrum_filename, config_filename.parent)
        ))
    metal_table = par.get('thermochemistry', {}).get('metal_pie_table_filename')
    if metal_table:
        par['thermochemistry']['metal_pie_table_filename'] = _resolve_path(
            metal_table, config_filename.parent
        )
    return {
        'par': par,
        'initial_condition': initial_condition,
        'example': config['example'],
    }
