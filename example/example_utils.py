"""Shared helpers for example scripts in this directory."""

import csv
import copy
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
            for key in ('directory', 'savedir', 'time_list_filename'):
                if key in output:
                    output[key] = _resolve_path(output[key], config_filename.parent)
    return {
        'par': par,
        'initial_condition': initial_condition,
        'example': config.get('example', {}),
    }


def snapshot_physical_fields(hdf5_filename):
    """Return radial snapshot fields converted to physical quantities.

    The snapshot metadata determines whether conversion is needed. Ordinary
    physical snapshots are returned unchanged; supercomoving snapshots use
    the canonical cosmology header contract.
    """
    with h5py.File(hdf5_filename, 'r') as hdf5:
        header = hdf5['Header']
        data = hdf5['Data']
        if 'boundary_comoving_code' in data:
            boundary_comoving_code = np.asarray(data['boundary_comoving_code'][()], dtype=float)
            density_comoving_code = np.asarray(data['rho_comoving_code'][()], dtype=float)
            velocity_supercomoving_code = np.asarray(data['vel_supercomoving_code'][()], dtype=float)
            temperature_supercomoving_code = np.asarray(data['temp_supercomoving_code'][()], dtype=float)
            cosmological_fields = True
        else:
            boundary_proper_code = np.asarray(data['boundary_proper_code'][()], dtype=float)
            density_proper_code = np.asarray(data['rho_proper_code'][()], dtype=float)
            velocity_proper_code = np.asarray(data['vel_proper_code'][()], dtype=float)
            temperature_proper_code = np.asarray(data['temp_proper_code'][()], dtype=float)
            cosmological_fields = False
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
            tau = float(np.asarray(data.file['Header']['tau_supercomoving_code'][()]))
            gamma = float(header.attrs.get('gamma', 5.0 / 3.0))
            scale_factor = float(cosmology.scale_factor_from_supercomoving(tau))
            hubble = float(cosmology.hubble_from_supercomoving(tau))
            radius = 0.5 * (boundary_comoving_code[:-1] + boundary_comoving_code[1:])
            return {
                'boundary_proper_cgs_cm': physical_radius(boundary_comoving_code, scale_factor),
                'radius_proper_cgs_cm': physical_radius(radius, scale_factor),
                'rho_proper_cgs_g_cm3': physical_density(density_comoving_code, scale_factor),
                'vel_peculiar_proper_cgs_cm_s': physical_velocity(
                    velocity_supercomoving_code, radius, scale_factor, hubble
                ),
                'temperature_proper_cgs_K': physical_temperature(
                    temperature_supercomoving_code, scale_factor, gamma
                ),
            }
        if cosmological_fields:
            raise ValueError('Cosmological HDF5 fields require supercomoving metadata')
        return {
            'boundary_proper_code': boundary_proper_code,
            'radius_proper_code': 0.5 * (boundary_proper_code[:-1] + boundary_proper_code[1:]),
            'rho_proper_code': density_proper_code,
            'vel_proper_code': velocity_proper_code,
            'temp_proper_code': temperature_proper_code,
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
        if 'boundary_proper_code' in data:
            boundary_dataset = data['boundary_proper_code']
            velocity_dataset = data['vel_proper_code']
            density_dataset = data['rho_proper_code']
            temperature_dataset = data['temp_proper_code']
            physical_values = False
        else:
            boundary_dataset = data['boundary_comoving_code']
            velocity_dataset = data['vel_supercomoving_code']
            density_dataset = data['rho_comoving_code']
            temperature_dataset = data['temp_supercomoving_code']
            physical_values = True

        fields = snapshot_physical_fields(hdf5_filename)
        if physical_values:
            boundaries = np.asarray(fields['boundary_proper_cgs_cm'], dtype=float)
            velocity = np.asarray(fields['vel_peculiar_proper_cgs_cm_s'], dtype=float)
            density = np.asarray(fields['rho_proper_cgs_g_cm3'], dtype=float)
            temperature = np.asarray(fields['temperature_proper_cgs_K'], dtype=float)
        else:
            boundaries = np.asarray(fields['boundary_proper_code'], dtype=float)
            velocity = np.asarray(fields['vel_proper_code'], dtype=float)
            density = np.asarray(fields['rho_proper_code'], dtype=float)
            temperature = np.asarray(fields['temp_proper_code'], dtype=float)
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

        if physical_values:
            radius = radius / (1.0 * unyt.pc).to_value(unyt.cm)
            velocity = velocity / (1.0 * unyt.km).to_value(unyt.cm)
            density = density / (1.0 * unyt.mp).to_value(unyt.g)
        else:
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
        writer.writerow(('RADIUS_PC', 'VELOCITY_cgs_KMS', 'DENSITY_CM3', 'TEMP_cgs_K'))
        writer.writerows(
            zip(
                (f'{value:.8g}' for value in radius),
                (f'{value:.8g}' for value in velocity),
                (f'{value:.8g}' for value in density),
                (f'{value:.8g}' for value in temperature),
            )
        )
    return csv_filename
