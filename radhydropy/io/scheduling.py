# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Output-time parsing and simulation output scheduling interfaces."""

from pathlib import Path
from typing import Any

import numpy as np
import unyt


def load_output_time_list(filename: str | None) -> Any:
    """Load explicit output times from a text file."""
    if not filename:
        return None

    outputtimepath = Path(filename)
    if not outputtimepath.exists():
        raise FileNotFoundError(f"Output-time file not found: {outputtimepath}")

    unit = None
    output_times = []
    with outputtimepath.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if unit is None:
                unit = tokens[0]
                for token in tokens[1:]:
                    output_times.append(float(token))
                continue
            for token in tokens:
                output_times.append(float(token))

    if unit is None:
        raise ValueError(f"Output-time file is empty: {outputtimepath}")

    return np.asarray(output_times, dtype=float) * unyt.Unit(unit)


def write_numbered_hdf5(sim: Any, outindex: int) -> Any:
    from radhydropy.output import write_numbered_hdf5 as implementation

    return implementation(sim, outindex)


def hdf5_output_callback(
    sim: Any,
    outputtime: Any = 0,
    output_state: Any = None,
    snapshot_callback: Any = None,
) -> Any:
    from radhydropy import io as public_io
    from radhydropy.output import hdf5_output_callback as implementation

    return implementation(
        sim,
        outputtime,
        output_state,
        output_writer=public_io.write_numbered_hdf5,
        snapshot_callback=snapshot_callback,
    )


def run_with_output_times(
    sim: Any,
    outputtime: Any = 0,
    mode: str = "hydro_sources",
    advect_chemistry: bool = True,
    stop_condition: Any = None,
    step_backend: Any = None,
    step_backend_kwargs: Any = None,
    before_step_callback: Any = None,
    history_callback: Any = None,
    snapshot_callback: Any = None,
) -> Any:
    from radhydropy import io as public_io
    from radhydropy.output import run_with_output_times as implementation

    return implementation(
        sim,
        outputtime=outputtime,
        mode=mode,
        advect_chemistry=advect_chemistry,
        stop_condition=stop_condition,
        step_backend=step_backend,
        step_backend_kwargs=step_backend_kwargs,
        output_writer=public_io.write_numbered_hdf5,
        before_step_callback=before_step_callback,
        history_callback=history_callback,
        snapshot_callback=snapshot_callback,
    )
