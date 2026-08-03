"""Shared helpers for example scripts in this directory."""

import csv
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np
import unyt


def clean_previous_outputs(runparams):
    """Delete stale ``Output_*.hdf5`` files before running an example."""
    outdir = Path(runparams.get('outdir', '.'))
    prefix = runparams.get('outfileprefix', 'Output')
    if not outdir.exists():
        return
    for path in outdir.glob(f'{prefix}_*.hdf5'):
        path.unlink(missing_ok=True)


def write_radial_profile_csv(hdf5_filename, csv_filename=None):
    """Write physical radial velocity and hydrogen density from an HDF5 file.

    The HDF5 datasets are expected to be ``Data/Boundary``, ``Data/Velocity``,
    and ``Data/Density`` as written by :func:`radhydropy.io.writehdf5`.  The
    boundary dataset is used to calculate cell-center radii.  Ghost cells,
    when identified by ``Header.attrs['noghost']``, are omitted from the CSV.

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

        boundaries = np.asarray(boundary_dataset[()], dtype=float)
        velocity = np.asarray(velocity_dataset[()], dtype=float)
        density = np.asarray(density_dataset[()], dtype=float)
        if len(boundaries) != len(velocity) + 1:
            raise ValueError(
                'Data/Boundary must contain exactly one more value than '
                'Data/Velocity and Data/Density.'
            )
        if len(velocity) != len(density):
            raise ValueError(
                'Data/Velocity and Data/Density must have the same length.'
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

    csv_filename.parent.mkdir(parents=True, exist_ok=True)
    with csv_filename.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('RADIUS_PC', 'VELOCITY_KMS', 'DENSITY_CM3'))
        writer.writerows(
            zip(
                (f'{value:.8g}' for value in radius),
                (f'{value:.8g}' for value in velocity),
                (f'{value:.8g}' for value in density),
            )
        )
    return csv_filename
