"""Shared helpers for example scripts in this directory."""

import csv
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np
import unyt
import yaml

from radhydropy.example_config import _load_yaml_value, _resolve_path

from radhydropy.cosmological_variables import (
    physical_density,
    physical_pressure,
    physical_radius,
    physical_temperature,
    physical_velocity,
)
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.io import _restore_header_attr_value


def load_nested_example_config(config_filename):
    """Load a nested example YAML configuration with unit-aware values.

    The returned mapping has separate ``par``, ``initial_condition``, and
    ``example`` sections. Values written as ``{value: ..., unit: ...}`` are
    converted to unyt quantities before being passed to the runtime.
    """
    config_filename = Path(config_filename).resolve()
    with config_filename.open(encoding='utf-8') as config_file:
        raw = yaml.safe_load(config_file)
    config = _load_yaml_value(raw)
    if not isinstance(config, dict) or 'par' not in config:
        raise ValueError("nested example configuration requires a 'par' section")
    par = config['par']
    initial_condition = config.get('initial_condition', {})
    if 'mesh' in par and 'grid_cells' not in par['mesh']:
        if 'grid_cells' in initial_condition:
            par['mesh']['grid_cells'] = initial_condition['grid_cells']
    if 'simulation' in par and 'initial_condition_filename' in par['simulation']:
        par['simulation']['initial_condition_filename'] = _resolve_path(
            par['simulation']['initial_condition_filename'], config_filename.parent
        )
    for section in ('par', 'example'):
        values = config.get(section, {})
        for key in ('output_directory', 'savedir', 'outputtimefilename'):
            if key in values:
                values[key] = _resolve_path(values[key], config_filename.parent)
        if section == 'par' and 'output' in values:
            output = values['output']
            for key in ('directory', 'time_list_filename'):
                if key in output:
                    output[key] = _resolve_path(output[key], config_filename.parent)
    return {
        'par': par,
        'initial_condition': initial_condition,
        'example': config.get('example', {}),
    }


def legacy_example_parameters(config):
    """Project a nested example config for legacy IC/plot helper APIs.

    Runtime solvers must receive ``config['par']`` directly.  This narrow
    adapter is retained for older example helper functions while they are
    being migrated to consume nested groups themselves.
    """
    par = config['par']
    initial = config.get('initial_condition', {})
    flat = dict(initial)
    simulation = par.get('simulation', {})
    mesh = par.get('mesh', {})
    hydro = par.get('hydrodynamics', {})
    boundary = par.get('boundary', {})
    timestep = par.get('timestep', {})
    output = par.get('output', {})
    flat.update({
        'simname': simulation.get('name'),
        'ICfilename': simulation.get('initial_condition_filename'),
        'coordsys': simulation.get('coordinate_system'),
        'final_time': simulation.get('final_time'),
        'timesim': simulation.get('final_time'),
        'nogrid': mesh.get('grid_cells', initial.get('grid_cells')),
        'number_of_cells': mesh.get('grid_cells', initial.get('grid_cells')),
        'noghost': mesh.get('ghost_cells', 2),
        'EOStype': hydro.get('eos_type', 'polytropic'),
        'gamma': hydro.get('gamma', 5.0 / 3.0),
        'CFL': hydro.get('CFL', 0.1),
        'order': hydro.get('order', 0),
        'boundcond': boundary.get('condition', 'OpenSph'),
        'dtmin': timestep.get('dtmin'),
        'dtmax': timestep.get('dtmax'),
        'outdir': output.get('directory', '.'),
        'savedir': output.get('savedir', output.get('directory', '.')),
        'outfileprefix': output.get('filename_prefix', 'Output'),
        'CodeUnits': par.get('units', {}).get('CodeUnits'),
    })
    for group in ('chemistry', 'thermochemistry', 'radiation'):
        flat.update(par.get(group, {}))
    flat.update(config.get('example', {}))
    return flat


def snapshot_physical_fields(hdf5_filename):
    """Return radial snapshot fields converted to physical quantities.

    The snapshot metadata determines whether conversion is needed. Ordinary
    physical snapshots are returned unchanged; supercomoving snapshots use
    the canonical cosmology header contract.
    """
    with h5py.File(hdf5_filename, 'r') as hdf5:
        header = hdf5['Header']
        data = hdf5['Data']
        boundary = np.asarray(data['Boundary'][()], dtype=float)
        density = np.asarray(data['Density'][()], dtype=float)
        velocity = np.asarray(data['Velocity'][()], dtype=float)
        temperature = np.asarray(data['Temperature'][()], dtype=float)
        representation = header.attrs.get('VelocityRepresentation', 'physical')
        if isinstance(representation, bytes):
            representation = representation.decode()
        if representation == 'supercomoving_peculiar':
            code_units = _restore_header_attr_value(header.attrs['CodeUnits'])
            from radhydropy.units import CodeUnits
            code_units = CodeUnits.from_mapping(code_units)
            cosmology = EinsteinDeSitter.from_code_units(
                code_units,
                t_ref=float(header.attrs['CosmologyTRef']),
                a_ref=float(header.attrs['CosmologyARef']),
            )
            tau = float(np.asarray(data.file['Header']['Time'][()]))
            gamma = float(header.attrs.get('gamma', 5.0 / 3.0))
            scale_factor = float(cosmology.scale_factor_from_supercomoving(tau))
            hubble = float(cosmology.hubble_from_supercomoving(tau))
            radius = 0.5 * (boundary[:-1] + boundary[1:])
            return {
                'boundary': physical_radius(boundary, scale_factor),
                'radius': physical_radius(radius, scale_factor),
                'density': physical_density(density, scale_factor),
                'velocity': physical_velocity(velocity, radius, scale_factor, hubble),
                'temperature': physical_temperature(
                    temperature, scale_factor, gamma
                ),
            }
        return {
            'boundary': boundary,
            'radius': 0.5 * (boundary[:-1] + boundary[1:]),
            'density': density,
            'velocity': velocity,
            'temperature': temperature,
        }


def clean_previous_outputs(runparams):
    """Delete stale ``Output_*.hdf5`` files before running an example."""
    if 'output' in runparams:
        runparams = runparams['output']
        outdir = Path(runparams.get('directory', '.'))
        prefix = runparams.get('filename_prefix', 'Output')
    else:
        outdir = Path(runparams.get('outdir', '.'))
        prefix = runparams.get('outfileprefix', 'Output')
    if not outdir.exists():
        return
    for path in outdir.glob(f'{prefix}_*.hdf5'):
        path.unlink(missing_ok=True)


def write_radial_profile_csv(hdf5_filename, csv_filename=None):
    """Write physical radial velocity, hydrogen density, and temperature.

    The HDF5 datasets are expected to be ``Data/Boundary``, ``Data/Velocity``,
    ``Data/Density``, and ``Data/Temperature`` as written by
    :func:`radhydropy.io.writehdf5`.  The boundary dataset is used to calculate
    cell-center radii.  Ghost cells, when identified by
    ``Header.attrs['noghost']``, are omitted from the CSV.

    Parameters
    ----------
    hdf5_filename : str or pathlib.Path
        Snapshot HDF5 file to read.
    csv_filename : str or pathlib.Path, optional
        Destination CSV file.  Defaults to the HDF5 filename with a ``.csv``
        suffix.

    Returns
    -------
    pathlib.Path
        The written CSV path.
    """
    hdf5_filename = Path(hdf5_filename)
    csv_filename = (
        hdf5_filename.with_suffix('.csv')
        if csv_filename is None
        else Path(csv_filename)
    )

    with h5py.File(hdf5_filename, 'r') as hdf5:
        header = hdf5['Header']
        data = hdf5['Data']
        boundary_dataset = data['Boundary']
        velocity_dataset = data['Velocity']
        density_dataset = data['Density']
        temperature_dataset = data['Temperature']

        fields = snapshot_physical_fields(hdf5_filename)
        boundaries = np.asarray(fields['boundary'], dtype=float)
        velocity = np.asarray(fields['velocity'], dtype=float)
        density = np.asarray(fields['density'], dtype=float)
        temperature = np.asarray(fields['temperature'], dtype=float)
        if len(boundaries) != len(velocity) + 1:
            raise ValueError(
                'Data/Boundary must contain exactly one more value than '
                'Data/Velocity, Data/Density, and Data/Temperature.'
            )
        if not (len(velocity) == len(density) == len(temperature)):
            raise ValueError(
                'Data/Velocity, Data/Density, and Data/Temperature must '
                'have the same length.'
            )

        noghost = header.attrs.get('noghost', 0)
        if not isinstance(noghost, Integral):
            noghost = int(np.asarray(noghost).item())
        noghost = int(noghost)
        if noghost < 0 or 2 * noghost >= len(velocity):
            raise ValueError(f'Invalid number of ghost cells: {noghost}')
        start = noghost
        stop = len(velocity) - noghost

        radius = 0.5 * (boundaries[start:stop] + boundaries[start + 1:stop + 1])
        velocity = velocity[start:stop]
        density = density[start:stop]
        temperature = temperature[start:stop]

        radius = unyt.unyt_array(
            radius,
            boundary_dataset.attrs.get('units', 'cm'),
        ).to_value(unyt.pc)
        velocity = unyt.unyt_array(
            velocity,
            velocity_dataset.attrs.get('units', 'cm/s'),
        ).to_value(unyt.km / unyt.s)
        density = unyt.unyt_array(
            density,
            density_dataset.attrs.get('units', 'g/cm**3'),
        ).to_value(unyt.g / unyt.cm**3)
        density /= (1.0 * unyt.mp).to_value(unyt.g)
        temperature = unyt.unyt_array(
            temperature,
            temperature_dataset.attrs.get('units', 'K'),
        ).to_value(unyt.K)

    csv_filename.parent.mkdir(parents=True, exist_ok=True)
    with csv_filename.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(('RADIUS_PC', 'VELOCITY_KMS', 'DENSITY_CM3', 'TEMP_K'))
        writer.writerows(
            zip(
                (f'{value:.8g}' for value in radius),
                (f'{value:.8g}' for value in velocity),
                (f'{value:.8g}' for value in density),
                (f'{value:.8g}' for value in temperature),
            )
        )
    return csv_filename
