"""HDF5 input/output and output-scheduling interfaces.

The implementation is split into submodules by responsibility, while this
module preserves the established ``radhydropy.io`` public import surface.
The first migration keeps the existing serializer together in ``hdf5``;
metadata, cosmology, field, and scheduling helpers can be extracted without
changing callers in subsequent steps.
"""

from radhydropy.io.cosmology_state import (
    read_supercomoving_state_hdf5,
    write_supercomoving_state_hdf5,
)
from radhydropy.io.hdf5 import (
    loadhdf5,
    readhdf5,
    write_snapshot_hdf5,
    writehdf5,
)
from radhydropy.io.metadata import (
    parameter_tree,
    update_used_parameters_yaml,
    write_used_parameters,
)
from radhydropy.io.scheduling import (
    hdf5_output_callback,
    load_output_time_list,
    run_with_output_times,
    write_numbered_hdf5,
)
from radhydropy.io.validation import SnapshotConfigurationError

__all__ = [
    "SnapshotConfigurationError",
    "hdf5_output_callback",
    "load_output_time_list",
    "loadhdf5",
    "parameter_tree",
    "read_supercomoving_state_hdf5",
    "readhdf5",
    "run_with_output_times",
    "update_used_parameters_yaml",
    "write_numbered_hdf5",
    "write_snapshot_hdf5",
    "write_supercomoving_state_hdf5",
    "write_used_parameters",
    "writehdf5",
]
