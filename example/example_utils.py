"""Shared helpers for example scripts in this directory."""

import csv
from collections.abc import Mapping
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
    required_sections = {'par', 'initial_condition', 'example'}
    if not isinstance(config, dict) or not required_sections.issubset(config):
        raise ValueError(
            "nested example configuration requires 'par', "
            "'initial_condition', and 'example' sections"
        )
    par = config['par']
    initial_condition = config['initial_condition']
    example = config['example']
    if 'mesh' in par and 'grid_cells' not in par['mesh']:
        if 'grid_cells' in initial_condition:
            par['mesh']['grid_cells'] = initial_condition['grid_cells']
    if 'simulation' in par and 'initial_condition_filename' in par['simulation']:
        par['simulation']['initial_condition_filename'] = _resolve_path(
            par['simulation']['initial_condition_filename'], config_filename.parent
        )
    for values in (par, example):
        for key in ('output_directory', 'savedir', 'outputtimefilename'):
            if key in values:
                values[key] = _resolve_path(values[key], config_filename.parent)
    if 'thermochemistry' in par:
        filename = par['thermochemistry'].get('metal_pie_table_filename')
        if filename:
            par['thermochemistry']['metal_pie_table_filename'] = _resolve_path(
                filename, config_filename.parent
            )
        output = par.get('output', {})
        for key in ('directory', 'savedir', 'time_list_filename'):
            if key in output:
                output[key] = _resolve_path(output[key], config_filename.parent)
    return {
        'par': par,
        'initial_condition': initial_condition,
        'example': example,
    }


def _require_complete_example_config(config, helper_name):
    if not isinstance(config, Mapping) or not {
        'par', 'initial_condition', 'example',
    }.issubset(config):
        raise TypeError(
            f'{helper_name} requires a complete nested example config'
        )


def snapshot_physical_fields(hdf5_filename, config):
    """Return radial snapshot fields converted to physical quantities.

    The snapshot metadata determines whether conversion is needed. Ordinary
    physical snapshots are returned unchanged; supercomoving snapshots use
    the canonical cosmology header contract.
    """
    _require_complete_example_config(config, 'snapshot_physical_fields')
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
            radius_comoving_code = 0.5 * (
                boundary_comoving_code[:-1] + boundary_comoving_code[1:]
            )
            return {
                'boundary_proper_cgs_cm': physical_radius(boundary_comoving_code, scale_factor),
                'radius_proper_cgs_cm': physical_radius(
                    radius_comoving_code, scale_factor
                ),
                'rho_proper_cgs_g_cm3': physical_density(
                    density_comoving_code, scale_factor
                ),
                'vel_peculiar_proper_cgs_cm_s': physical_velocity(
                    velocity_supercomoving_code,
                    radius_comoving_code,
                    scale_factor,
                    hubble,
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


def clean_previous_outputs(config):
    """Delete stale output files using a complete nested configuration."""
    if not isinstance(config, dict) or 'par' not in config:
        raise TypeError('clean_previous_outputs requires a complete example config')
    output_config = config['par'].get('output', {})
    outdir = Path(output_config.get('directory', '.'))
    prefix = output_config.get('filename_prefix', 'Output')
    if not outdir.exists():
        return
    for path in outdir.glob(f'{prefix}_*.hdf5'):
        path.unlink(missing_ok=True)


def write_radial_profile_csv(hdf5_filename, config, csv_filename=None):
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
    _require_complete_example_config(config, 'write_radial_profile_csv')
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
            boundary_proper_code_dataset = data['boundary_proper_code']
            vel_proper_code_dataset = data['vel_proper_code']
            rho_proper_code_dataset = data['rho_proper_code']
            temp_proper_code_dataset = data['temp_proper_code']
            physical_values = False
        else:
            boundary_comoving_code_dataset = data['boundary_comoving_code']
            vel_supercomoving_code_dataset = data['vel_supercomoving_code']
            rho_comoving_code_dataset = data['rho_comoving_code']
            temp_supercomoving_code_dataset = data['temp_supercomoving_code']
            physical_values = True

        fields = snapshot_physical_fields(hdf5_filename, config)
        if physical_values:
            boundary_proper_cgs_cm = np.asarray(
                fields['boundary_proper_cgs_cm'], dtype=float
            )
            vel_peculiar_proper_cgs_cm_s = np.asarray(
                fields['vel_peculiar_proper_cgs_cm_s'], dtype=float
            )
            rho_proper_cgs_g_cm3 = np.asarray(
                fields['rho_proper_cgs_g_cm3'], dtype=float
            )
            temperature_proper_cgs_K = np.asarray(
                fields['temperature_proper_cgs_K'], dtype=float
            )
        else:
            boundary_proper_code = np.asarray(
                fields['boundary_proper_code'], dtype=float
            )
            vel_proper_code = np.asarray(fields['vel_proper_code'], dtype=float)
            rho_proper_code = np.asarray(fields['rho_proper_code'], dtype=float)
            temp_proper_code = np.asarray(fields['temp_proper_code'], dtype=float)
        if physical_values:
            boundary_count = len(boundary_proper_cgs_cm)
            cell_count = len(vel_peculiar_proper_cgs_cm_s)
        else:
            boundary_count = len(boundary_proper_code)
            cell_count = len(vel_proper_code)
        if boundary_count != cell_count + 1:
            raise ValueError(
                'Data/Boundary must contain exactly one more value than '
                'Data/Velocity, Data/Density, and Data/Temperature.'
            )
        if physical_values:
            quantity_count = (
                len(vel_peculiar_proper_cgs_cm_s),
                len(rho_proper_cgs_g_cm3),
                len(temperature_proper_cgs_K),
            )
        else:
            quantity_count = (
                len(vel_proper_code),
                len(rho_proper_code),
                len(temp_proper_code),
            )
        if not (quantity_count[0] == quantity_count[1] == quantity_count[2]):
            raise ValueError(
                'Data/Velocity, Data/Density, and Data/Temperature must '
                'have the same length.'
            )

        noghost = header.attrs.get('noghost', 0)
        if not isinstance(noghost, Integral):
            noghost = int(np.asarray(noghost).item())
        noghost = int(noghost)
        if noghost < 0 or 2 * noghost >= cell_count:
            raise ValueError(f'Invalid number of ghost cells: {noghost}')
        start = noghost
        stop = cell_count - noghost

        if physical_values:
            radius_proper_cgs_cm = 0.5 * (
                boundary_proper_cgs_cm[start:stop]
                + boundary_proper_cgs_cm[start + 1:stop + 1]
            )
            vel_peculiar_proper_cgs_cm_s = vel_peculiar_proper_cgs_cm_s[start:stop]
            rho_proper_cgs_g_cm3 = rho_proper_cgs_g_cm3[start:stop]
            temperature_proper_cgs_K = temperature_proper_cgs_K[start:stop]
        else:
            radius_proper_code = 0.5 * (
                boundary_proper_code[start:stop]
                + boundary_proper_code[start + 1:stop + 1]
            )
            vel_proper_code = vel_proper_code[start:stop]
            rho_proper_code = rho_proper_code[start:stop]
            temp_proper_code = temp_proper_code[start:stop]

        if physical_values:
            radius_proper_cgs_pc = radius_proper_cgs_cm / (
                1.0 * unyt.pc
            ).to_value(unyt.cm)
            vel_peculiar_proper_cgs_km_s = vel_peculiar_proper_cgs_cm_s / (
                1.0 * unyt.km
            ).to_value(unyt.cm)
            number_density_cgs_cm3 = rho_proper_cgs_g_cm3 / (
                1.0 * unyt.mp
            ).to_value(unyt.g)
        else:
            radius_proper_cgs_pc = unyt.unyt_array(
                radius_proper_code,
                boundary_proper_code_dataset.attrs.get('units', 'cm'),
            ).to_value(unyt.pc)
            vel_peculiar_proper_cgs_km_s = unyt.unyt_array(
                vel_proper_code,
                vel_proper_code_dataset.attrs.get('units', 'cm/s'),
            ).to_value(unyt.km / unyt.s)
            number_density_cgs_cm3 = unyt.unyt_array(
                rho_proper_code,
                rho_proper_code_dataset.attrs.get('units', 'g/cm**3'),
            ).to_value(unyt.g / unyt.cm**3)
            number_density_cgs_cm3 /= (1.0 * unyt.mp).to_value(unyt.g)
            temperature_proper_cgs_K = unyt.unyt_array(
                temp_proper_code,
                temp_proper_code_dataset.attrs.get('units', 'K'),
            ).to_value(unyt.K)

    csv_filename.parent.mkdir(parents=True, exist_ok=True)
    with csv_filename.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(('RADIUS_PC', 'VELOCITY_cgs_KMS', 'DENSITY_CM3', 'TEMP_cgs_K'))
        writer.writerows(
            zip(
                (f'{value:.8g}' for value in radius_proper_cgs_pc),
                (f'{value:.8g}' for value in vel_peculiar_proper_cgs_km_s),
                (f'{value:.8g}' for value in number_density_cgs_cm3),
                (f'{value:.8g}' for value in temperature_proper_cgs_K),
            )
        )
    return csv_filename
