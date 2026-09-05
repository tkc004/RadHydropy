"""Write radial-profile CSV files from stellar-wind HDF5 snapshots.

Run without arguments to process every ``Output_*.hdf5`` in this directory.
CSV files are written to the ``radial_profiles`` subdirectory. Pass one
snapshot path to write the generic ``radial_profile.csv`` file there.
"""

import argparse
import sys
from pathlib import Path

import h5py
import unyt

EXAMPLE_DIR = Path(__file__).resolve().parent
EXAMPLE_ROOT = EXAMPLE_DIR.parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import example_utils as eu


def snapshot_time_myr(snapshot_filename):
    """Read the snapshot time from the HDF5 header and return Myr."""
    with h5py.File(snapshot_filename, 'r') as hdf5:
        time_dataset = hdf5['Header']['time_code']
        time = unyt.unyt_quantity(
            time_dataset[()],
            time_dataset.attrs.get('units', 's'),
        )
    return time.to_value(unyt.Myr)


def write_snapshot_profile(snapshot_filename, csv_filename):
    """Convert one HDF5 snapshot to a radial-profile CSV file."""
    return eu.write_radial_profile_csv(snapshot_filename, csv_filename)


def process_snapshots(snapshot_directory=EXAMPLE_DIR):
    """Write one time-stamped radial-profile CSV for every snapshot."""
    snapshot_directory = Path(snapshot_directory).resolve()
    csv_directory = snapshot_directory / 'radial_profiles'
    csv_directory.mkdir(parents=True, exist_ok=True)
    snapshots = sorted(snapshot_directory.glob('Output_*.hdf5'))
    if not snapshots:
        raise FileNotFoundError(f'No Output_*.hdf5 files found in {snapshot_directory}')

    csv_files = []
    for snapshot in snapshots:
        time_myr = snapshot_time_myr(snapshot)
        time_label = f'{time_myr:.6g}'
        csv_filename = csv_directory / f'radial_profile_{time_label}Myr.csv'
        csv_files.append(write_snapshot_profile(snapshot, csv_filename))
        print(f'{snapshot.name} -> {csv_filename.name}')
    return csv_files


def parse_args():
    parser = argparse.ArgumentParser(
        description='Write radial-profile CSV files from HDF5 snapshots.',
    )
    parser.add_argument(
        'snapshot',
        nargs='?',
        type=Path,
        help='One snapshot to write as radial_profile.csv.',
    )
    parser.add_argument(
        '--directory',
        type=Path,
        default=EXAMPLE_DIR,
        help='Directory containing Output_*.hdf5 files.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.snapshot is not None:
        snapshot = args.snapshot.resolve()
        output_directory = snapshot.parent / 'radial_profiles'
        output_directory.mkdir(parents=True, exist_ok=True)
        output = output_directory / 'radial_profile.csv'
        write_snapshot_profile(snapshot, output)
        print(f'{snapshot.name} -> {output.name}')
    else:
        process_snapshots(args.directory)


if __name__ == '__main__':
    main()
