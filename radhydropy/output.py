# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Simulation output scheduling helpers."""  # noqa: CPY001

import logging

import numpy as np
import unyt

from radhydropy.diagnostic_logging import log_diagnostic
from radhydropy.runtime_fields import runtime_fields
from radhydropy.units import code_unit_scales


def write_numbered_hdf5(sim, outindex):
    """Write ``Output_###.hdf5`` for the supplied simulation."""
    filename = (
        sim.par.output.directory
        + "/"
        + sim.par.output.filename_prefix
        + "_%03d" % outindex
        + ".hdf5"
    )
    # The live runtime already owns ghost-filled, solver-ready state.  Route
    # snapshots directly to the serializer so the IC preparation boundary
    # does not append a second set of ghost cells.
    from radhydropy.io import write_snapshot_hdf5

    write_snapshot_hdf5(sim, filename)
    return filename


def hdf5_output_callback(
    sim,
    outputtime=0,
    output_state=None,
    output_writer=None,
    snapshot_callback=None,
):
    """Return a callback that writes HDF5 snapshots at fixed cadence."""
    if output_writer is None:
        output_writer = write_numbered_hdf5
    if output_state is None:
        output_state = {
            "outtime": 0.0 * sim.par.simulation.final_time,
            "outindex": 1,
            "last_output_time_s": float(
                np.asarray(
                    getattr(sim.fluid, runtime_fields(sim.par).time),
                    dtype=float,
                ),
            ),
        }
    else:
        output_state.setdefault(
            "outtime",
            0.0 * sim.par.simulation.final_time,
        )
        output_state.setdefault("outindex", 1)
        output_state.setdefault(
            "last_output_time_s",
            float(
                np.asarray(
                    getattr(sim.fluid, runtime_fields(sim.par).time),
                    dtype=float,
                ),
            ),
        )

    def callback(sim, step):
        dt = step["dt"]
        if getattr(dt, "shape", None) == (1,):
            dt = dt[0]
        if getattr(sim.par, "verbose", 0) >= 1:
            fields = runtime_fields(sim.par)
            log_diagnostic(
                logging.INFO,
                "hydro_step",
                runtime_time_field=fields.time,
                runtime_time_value=getattr(sim.fluid, fields.time),
                dt_runtime_code=dt,
            )
        if output_state["outtime"] >= sim.par.output.cadence:
            snapshot_filename = output_writer(
                sim,
                output_state["outindex"],
            )
            if snapshot_callback is not None:
                snapshot_callback(
                    sim,
                    snapshot_filename,
                    output_state["outindex"],
                )
            output_state["last_output_time_s"] = float(
                np.asarray(
                    getattr(sim.fluid, runtime_fields(sim.par).time),
                    dtype=float,
                ),
            )
            output_state["outtime"] = 0.0 * sim.par.simulation.final_time
            output_state["outindex"] += 1
        else:
            output_state["outtime"] += dt

    return callback


def run_with_output_times(
    sim,
    outputtime=0,
    mode="hydro_sources",
    advect_chemistry=True,
    stop_condition=None,
    step_backend=None,
    step_backend_kwargs=None,
    output_writer=None,
    before_step_callback=None,
    history_callback=None,
    snapshot_callback=None,
):
    """Run a simulation using an explicit output-time list."""
    if output_writer is None:
        output_writer = write_numbered_hdf5
    if step_backend is None:
        step_backend = sim.Step
    if step_backend_kwargs is None:
        step_backend_kwargs = {}
    if before_step_callback is not None:
        before_step_callback(sim)
    initial_filename = output_writer(sim, 0)
    if snapshot_callback is not None:
        snapshot_callback(sim, initial_filename, 0)
    if history_callback is not None:
        history_callback(sim)
    last_output_time_s = float(
        np.asarray(
            getattr(sim.fluid, runtime_fields(sim.par).time),
            dtype=float,
        ),
    )
    current_time = getattr(sim.fluid, runtime_fields(sim.par).time)
    final_time = sim.par.simulation.final_time
    time_tol = max(abs(float(np.asarray(final_time, dtype=float))) * 1.0e-12, 1.0e-30)
    from radhydropy.io import load_output_time_list

    output_times = load_output_time_list(getattr(sim.par, "outputtimefilename", None))
    if output_times is None:
        output_times = []
    else:
        if hasattr(final_time, "units"):
            target_unit = final_time.units
            sorted_values = np.unique(
                np.asarray(output_times.to_value(target_unit), dtype=float),
            )
        else:
            code_units = getattr(sim.par, "CodeUnits", None)
            if code_units is None:
                code_units = getattr(
                    getattr(sim.par, "units", None),
                    "CodeUnits",
                    None,
                )
            sorted_values = np.unique(
                np.asarray(output_times.to_value(unyt.s), dtype=float)
                / code_unit_scales(code_units)["time_s"],
            )
        output_times = [
            value * final_time.units if hasattr(final_time, "units") else value
            for value in sorted_values
            if (
                (value * final_time.units if hasattr(final_time, "units") else value) > current_time
                and (value * final_time.units if hasattr(final_time, "units") else value)
                <= final_time
            )
        ]

    from radhydropy.rsim.evolution import _advance_until

    outindex = 1
    for target_time in output_times:
        if stop_condition is not None and stop_condition(sim):
            break
        target_time_value = float(np.asarray(target_time, dtype=float))
        _advance_until(
            sim,
            final_time=target_time,
            mode=mode,
            advect_chemistry=advect_chemistry,
            stop_condition=stop_condition,
            step_backend=step_backend,
            step_backend_kwargs=step_backend_kwargs,
            before_step_callback=before_step_callback,
            history_callback=history_callback,
            emit_initial_history=False,
        )
        if stop_condition is not None and stop_condition(sim):
            break
        # Euler/source steps can cross a target by a roundoff- or CFL-sized
        # amount.  Treat the first state at or beyond the target as the
        # requested snapshot instead of silently dropping the output.
        if (
            float(
                np.asarray(
                    getattr(sim.fluid, runtime_fields(sim.par).time),
                    dtype=float,
                ),
            )
            >= target_time_value - time_tol
        ):
            snapshot_filename = output_writer(sim, outindex)
            if snapshot_callback is not None:
                snapshot_callback(sim, snapshot_filename, outindex)
            last_output_time_s = float(
                np.asarray(
                    getattr(sim.fluid, runtime_fields(sim.par).time),
                    dtype=float,
                ),
            )
            outindex += 1

    _advance_until(
        sim,
        final_time=final_time,
        mode=mode,
        advect_chemistry=advect_chemistry,
        stop_condition=stop_condition,
        step_backend=step_backend,
        step_backend_kwargs=step_backend_kwargs,
        before_step_callback=before_step_callback,
        history_callback=history_callback,
        emit_initial_history=False,
    )

    if (
        abs(
            float(
                np.asarray(
                    getattr(sim.fluid, runtime_fields(sim.par).time),
                    dtype=float,
                ),
            )
            - last_output_time_s,
        )
        > time_tol
    ):
        snapshot_filename = output_writer(sim, outindex)
        if snapshot_callback is not None:
            snapshot_callback(sim, snapshot_filename, outindex)
