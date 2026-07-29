"""Shared helpers for example scripts in this directory."""

from pathlib import Path


def clean_previous_outputs(runparams):
    """Delete stale ``Output_*.hdf5`` files before running an example."""
    outdir = Path(runparams.get('outdir', '.'))
    prefix = runparams.get('outfileprefix', 'Output')
    if not outdir.exists():
        return
    for path in outdir.glob(f'{prefix}_*.hdf5'):
        path.unlink(missing_ok=True)
