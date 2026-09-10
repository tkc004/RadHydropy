"""Shared helpers for example scripts in this directory."""

import csv
from collections.abc import Mapping
from pathlib import Path

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
from radhydropy.rsim import Rsim
import radhydropy.io as rio
from radhydropy.units import CodeUnits


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
        for key in ('output_directory', 'directory'):
            if key in values:
                values[key] = _resolve_path(values[key], config_filename.parent)
    if 'thermochemistry' in par:
        filename = par['thermochemistry'].get('metal_pie_table_filename')
        if filename:
            par['thermochemistry']['metal_pie_table_filename'] = _resolve_path(
                filename, config_filename.parent
            )
        output = par.get('output', {})
        for key in ('directory', 'time_list_filename'):
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
    snapshot = Rsim(config['par'])
    rio.readhdf5(snapshot.par, snapshot.mesh, snapshot.fluid, str(hdf5_filename))
    par, mesh, fluid = snapshot.par, snapshot.mesh, snapshot.fluid
    first = int(par.mesh.ghost_cells)
    last = first + int(par.mesh.grid_cells)
    code_units = par.units.CodeUnits
    if par.supercomoving_coordinates:
        boundary_comoving_code = np.asarray(
            mesh.boundary_comoving_code[first:last + 1], dtype=float
        )
        radius_comoving_code = 0.5 * (
            boundary_comoving_code[:-1] + boundary_comoving_code[1:]
        )
        density_comoving_code = np.asarray(
            fluid.rho_comoving_code[first:last], dtype=float
        )
        velocity_supercomoving_code = np.asarray(
            fluid.vel_supercomoving_code[first:last], dtype=float
        )
        temperature_supercomoving_code = np.asarray(
            fluid.temp_supercomoving_code[first:last], dtype=float
        )
        tau_supercomoving_code = float(
            np.asarray(fluid.tau_supercomoving_code, dtype=float).flat[0]
        )
        _, scale_factor, hubble = par.cosmology.background_state_from_supercomoving(
            tau_supercomoving_code
        )
        length_cgs_cm = float(code_units.length_unit.to_value(unyt.cm))
        density_cgs_g_cm3 = float(code_units.density_unit.to_value(unyt.g / unyt.cm**3))
        velocity_cgs_cm_s = float(code_units.velocity_unit.to_value(unyt.cm / unyt.s))
        temperature_cgs_K = float(code_units.temperature_unit.to_value(unyt.K))
        gamma = float(par.hydrodynamics.gamma)
        return {
            'boundary_proper_cgs_cm': physical_radius(
                boundary_comoving_code, scale_factor
            ) * length_cgs_cm,
            'radius_proper_cgs_cm': physical_radius(
                radius_comoving_code, scale_factor
            ) * length_cgs_cm,
            'rho_proper_cgs_g_cm3': physical_density(
                density_comoving_code, scale_factor
            ) * density_cgs_g_cm3,
            'vel_peculiar_proper_cgs_cm_s': physical_velocity(
                velocity_supercomoving_code,
                radius_comoving_code,
                scale_factor,
                hubble,
            ) * velocity_cgs_cm_s,
            'temperature_proper_cgs_K': physical_temperature(
                temperature_supercomoving_code, scale_factor, gamma
            ) * temperature_cgs_K,
        }
    boundary_proper_code = np.asarray(
        mesh.boundary_proper_code[first:last + 1], dtype=float
    )
    return {
        'boundary_proper_code': boundary_proper_code,
        'radius_proper_code': 0.5 * (
            boundary_proper_code[:-1] + boundary_proper_code[1:]
        ),
        'rho_proper_code': np.asarray(fluid.rho_proper_code[first:last], dtype=float),
        'vel_proper_code': np.asarray(fluid.vel_proper_code[first:last], dtype=float),
        'temp_proper_code': np.asarray(fluid.temp_proper_code[first:last], dtype=float),
        'code_units': code_units,
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

    fields = snapshot_physical_fields(hdf5_filename, config)
    physical_values = 'boundary_proper_cgs_cm' in fields
    if physical_values:
        boundary_proper_cgs_cm = np.asarray(fields['boundary_proper_cgs_cm'], dtype=float)
        vel_peculiar_proper_cgs_cm_s = np.asarray(fields['vel_peculiar_proper_cgs_cm_s'], dtype=float)
        rho_proper_cgs_g_cm3 = np.asarray(fields['rho_proper_cgs_g_cm3'], dtype=float)
        temperature_proper_cgs_K = np.asarray(fields['temperature_proper_cgs_K'], dtype=float)
        boundary_count = len(boundary_proper_cgs_cm)
        cell_count = len(vel_peculiar_proper_cgs_cm_s)
    else:
        boundary_proper_code = np.asarray(fields['boundary_proper_code'], dtype=float)
        vel_proper_code = np.asarray(fields['vel_proper_code'], dtype=float)
        rho_proper_code = np.asarray(fields['rho_proper_code'], dtype=float)
        temp_proper_code = np.asarray(fields['temp_proper_code'], dtype=float)
        code_units = fields['code_units']
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

    if physical_values:
        radius_proper_cgs_cm = 0.5 * (
            boundary_proper_cgs_cm[:-1] + boundary_proper_cgs_cm[1:]
        )
        radius_proper_cgs_pc = radius_proper_cgs_cm / (1.0 * unyt.pc).to_value(unyt.cm)
        vel_peculiar_proper_cgs_km_s = vel_peculiar_proper_cgs_cm_s / (1.0 * unyt.km).to_value(unyt.cm)
        number_density_cgs_cm3 = rho_proper_cgs_g_cm3 / (1.0 * unyt.mp).to_value(unyt.g)
    else:
        radius_proper_code = fields['radius_proper_code']
        radius_proper_cgs_pc = radius_proper_code * float(code_units.length_unit.to_value(unyt.pc))
        vel_peculiar_proper_cgs_km_s = vel_proper_code * float(code_units.velocity_unit.to_value(unyt.km / unyt.s))
        number_density_cgs_cm3 = rho_proper_code * float(code_units.density_unit.to_value(unyt.g / unyt.cm**3)) / (1.0 * unyt.mp).to_value(unyt.g)
        temperature_proper_cgs_K = temp_proper_code * float(code_units.temperature_unit.to_value(unyt.K))

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
