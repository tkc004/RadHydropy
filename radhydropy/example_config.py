"""Helpers for example scripts that read parameters from YAML files."""

from pathlib import Path

import unyt
import yaml
from radhydropy.radiation_spectrum import load_radiation_spectrum, resolve_spectrum_filename


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


def _default_code_units():
    return {
        'CodeUnits': {
            'InternalUnitSystem': {
                'UnitMass_in_cgs': 1.0,
                'UnitLength_in_cgs': 1.0,
                'UnitVelocity_in_cgs': 1.0,
                'UnitCurrent_in_cgs': 1.0,
                'UnitTemp_in_cgs': 1.0,
            }
        }
    }


def load_example_parameters(config_filename, rundir=None):
    """Load ``runparams`` and ``ICparams`` from an example YAML file."""
    config_filename = Path(config_filename)
    rundir = config_filename.parent.resolve()
    with config_filename.open() as config_file:
        config = yaml.safe_load(config_file)

    if 'runparams' in config:
        runparams = _load_yaml_value(config['runparams'])
        icparams = _load_yaml_value(config['ICparams'])
    else:
        # Keep the legacy helper usable for migrated examples and older tests.
        # New runners should use load_nested_example_config instead.
        nested_par = _load_yaml_value(config['par'])
        icparams = _load_yaml_value(config['initial_condition'])
        runparams = dict(icparams)
        simulation = nested_par.get('simulation', {})
        mesh = nested_par.get('mesh', {})
        hydro = nested_par.get('hydrodynamics', {})
        boundary = nested_par.get('boundary', {})
        timestep = nested_par.get('timestep', {})
        output = nested_par.get('output', {})
        runparams.update({
            'simname': simulation.get('name'),
            'ICfilename': simulation.get('initial_condition_filename'),
            'coordsys': simulation.get('coordinate_system'),
            'final_time': simulation.get('final_time'),
            'timesim': simulation.get('final_time'),
            'number_of_cells': mesh.get('grid_cells', icparams.get('grid_cells')),
            'nogrid': mesh.get('grid_cells', icparams.get('grid_cells')),
            'noghost': mesh.get('ghost_cells', 2),
            'EOStype': hydro.get('eos_type', 'polytropic'),
            'gamma': hydro.get('gamma', 5.0 / 3.0),
            'CFL': hydro.get('CFL', 0.1),
            'order': hydro.get('order', 0),
            'boundcond': boundary.get('condition', 'OpenSph'),
            'dtmin': timestep.get('dtmin'),
            'dtmax': timestep.get('dtmax'),
            'chemistry_timestep': timestep.get('chemistry_timestep'),
            'evolution_timestep': timestep.get('evolution_timestep'),
            'outdir': output.get('directory', '.'),
            'savedir': output.get('savedir', output.get('directory', '.')),
            'outfileprefix': output.get('filename_prefix', 'Output'),
            'outputtimefilename': output.get('time_list_filename'),
            'CodeUnits': nested_par.get('units', {}).get('CodeUnits'),
        })
        icparams.setdefault(
            'number_of_cells', mesh.get('grid_cells', icparams.get('grid_cells'))
        )
        for group in ('chemistry', 'thermochemistry', 'radiation'):
            runparams.update(nested_par.get(group, {}))
        runparams.setdefault(
            'radiative_transfer_temporal_scheme',
            nested_par.get('radiation', {}).get(
                'radiative_transfer_temporal_scheme', 'instantaneous'
            ),
        )
        for alias, source in (
            ('source_photon_rate', 'radiative_transfer_source_photon_rate'),
            ('alpha_B_coefficient', 'hydrogen_alpha_B'),
            ('sigma_gamma', 'hydrogen_sigma_gamma'),
            ('epsilon_gamma', 'hydrogen_epsilon_gamma'),
        ):
            if alias not in runparams and source in runparams:
                runparams[alias] = runparams[source]
        example_values = _load_yaml_value(config.get('example', {}))
        runparams.update(example_values)
        icparams.update({
            key: value for key, value in example_values.items()
            if key in {'analytic_inner_radius'}
        })
        output = runparams.get('output', {})
        simulation = runparams.get('simulation', {})
        units = runparams.get('units', {})
        if 'CodeUnits' in units:
            runparams['CodeUnits'] = units['CodeUnits']
        if 'initial_condition_filename' in simulation:
            runparams['ICfilename'] = simulation['initial_condition_filename']
        if 'directory' in output:
            runparams['outdir'] = output['directory']
        if 'time_list_filename' in output:
            runparams['outputtimefilename'] = output['time_list_filename']
    if 'CodeUnits' not in runparams and 'InternalUnitSystem' not in runparams:
        raise ValueError("CodeUnits or InternalUnitSystem is required")
    for key in {'ICfilename', 'outdir', 'outputtimefilename', 'savedir'}:
        if key in runparams and runparams[key] is not None:
            runparams[key] = _resolve_path(runparams[key], rundir)
    if runparams.get('radiation_spectrum_filename') is not None:
        runparams.update(
            load_radiation_spectrum(
                resolve_spectrum_filename(
                    runparams['radiation_spectrum_filename'], rundir
                )
            )
        )
    return runparams, icparams
