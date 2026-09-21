# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Rsim execution subsystem helpers."""

import time
from typing import Any

import radhydropy.io as rio
from radhydropy.runtime_fields import runtime_fields


def _advance_until(
    sim: Any,
    final_time: Any,
    mode: str = "hydro_sources",
    *,
    advect_chemistry: bool = True,
    history_callback: Any = None,
    output_callback: Any = None,
    stop_condition: Any = None,
    step_backend: Any = None,
    step_backend_kwargs: Any = None,
    before_step_callback: Any = None,
    emit_initial_history: bool = True,
) -> dict[str, int]:
    """Advance ``sim`` to ``final_time`` using the shared step lifecycle.

    ``Evolve`` and explicit-time output scheduling both use this function so
    timestep estimation, callback ordering, counters, and progress reporting
    cannot drift apart.
    """
    if step_backend is None:
        step_backend = sim.Step
    if step_backend_kwargs is None:
        step_backend_kwargs = {}

    counters = {"hydro_steps": 0, "source_steps": 0}
    time_field = runtime_fields(sim.par).time
    progress_steps = 0
    if emit_initial_history and history_callback is not None:
        history_callback(sim)

    while getattr(sim.fluid, time_field) < final_time:
        if stop_condition is not None and stop_condition(sim):
            break
        if before_step_callback is not None:
            before_step_callback(sim)
        dt = sim.GetStepTime(final_time=final_time)
        step = step_backend(
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
            **step_backend_kwargs,
        )
        sim.last_step_dt = dt
        counters["hydro_steps"] += step["hydro_steps"]
        counters["source_steps"] += step["source_steps"]
        progress_steps += step["hydro_steps"]
        if progress_steps % 1000 == 0:
            final_time_value = float(final_time)
            progress_percent = (
                100.0
                if final_time_value == 0.0
                else 100.0 * float(getattr(sim.fluid, time_field)) / final_time_value
            )
            print(  # noqa: T201 - intentional periodic lifecycle progress output.
                f"--- hydro step {progress_steps}: "
                f"time={float(getattr(sim.fluid, time_field)):.6e} "
                f"dt={float(dt):.6e} ({progress_percent:.2f}%) ---",
                flush=True,
            )
        if history_callback is not None:
            history_callback(sim)
        if output_callback is not None:
            output_callback(sim, step)
    return counters


def Evolve(  # noqa: N802
    sim: Any,
    final_time: Any = None,
    mode: str = "hydro_sources",
    *,
    advect_chemistry: bool = True,
    history_callback: Any = None,
    output_callback: Any = None,
    stop_condition: Any = None,
    step_backend: Any = None,
    step_backend_kwargs: Any = None,
    before_step_callback: Any = None,
) -> dict[str, int]:
    """Evolve the simulation with a pluggable step backend."""
    if final_time is None:
        final_time = sim.par.simulation.final_time
    return _advance_until(
        sim,
        final_time=final_time,
        mode=mode,
        advect_chemistry=advect_chemistry,
        history_callback=history_callback,
        output_callback=output_callback,
        stop_condition=stop_condition,
        step_backend=step_backend,
        step_backend_kwargs=step_backend_kwargs,
        before_step_callback=before_step_callback,
    )


def Run(  # noqa: N802
    sim: Any,
    outputtime: Any = 0,
    mode: str = "hydro_sources",
    *,
    advect_chemistry: bool = True,
    stop_condition: Any = None,
    step_backend: Any = None,
    step_backend_kwargs: Any = None,
    before_step_callback: Any = None,
    history_callback: Any = None,
    snapshot_callback: Any = None,
) -> None:
    """Run the simulation loop and write periodic HDF5 outputs."""
    sim.WriteUsedParameters()
    if getattr(sim.par, "outputtimefilename", None):
        rio.run_with_output_times(
            sim,
            outputtime=outputtime,
            mode=mode,
            advect_chemistry=advect_chemistry,
            before_step_callback=before_step_callback,
            history_callback=history_callback,
            snapshot_callback=snapshot_callback,
            stop_condition=stop_condition,
            step_backend=step_backend,
            step_backend_kwargs=step_backend_kwargs,
        )
        return
    # Fixed-cadence output path: advance to `timesim` and write snapshots
    # whenever `outtime` reaches `outdeltatime`.
    print("--- Initization finished. Start running ... ---")  # noqa: T201 - intentional lifecycle output.
    print(  # noqa: T201 - intentional lifecycle timing output.
        "--- %s seconds ---" % (time.time() - getattr(sim, "_start_time", time.time())),
    )
    if before_step_callback is not None:
        before_step_callback(sim)
    initial_filename = rio.write_numbered_hdf5(sim, 0)
    if snapshot_callback is not None:
        snapshot_callback(sim, initial_filename, 0)
    output_state: dict[str, Any] = {}
    sim.Evolve(
        final_time=sim.par.simulation.final_time,
        mode=mode,
        advect_chemistry=advect_chemistry,
        before_step_callback=before_step_callback,
        history_callback=history_callback,
        output_callback=rio.hdf5_output_callback(
            sim,
            outputtime=outputtime,
            output_state=output_state,
            snapshot_callback=snapshot_callback,
        ),
        stop_condition=stop_condition,
        step_backend=step_backend,
        step_backend_kwargs=step_backend_kwargs,
    )
    if stop_condition is not None:
        final_index = output_state.get("outindex", 1)
        final_filename = rio.write_numbered_hdf5(
            sim,
            final_index,
        )
        if snapshot_callback is not None:
            snapshot_callback(sim, final_filename, final_index)
    print("--- Simulation finished. ---")  # noqa: T201 - intentional lifecycle output.
    print(  # noqa: T201 - intentional lifecycle timing output.
        "--- %s seconds ---" % (time.time() - getattr(sim, "_start_time", time.time())),
    )


def RunAll(  # noqa: N802
    sim: Any,
    outputtime: Any = 0,
    mode: str = "hydro_sources",
    *,
    advect_chemistry: bool = True,
    stop_condition: Any = None,
    step_backend: Any = None,
    step_backend_kwargs: Any = None,
    before_step_callback: Any = None,
    history_callback: Any = None,
    snapshot_callback: Any = None,
) -> None:
    """Run the full workflow from initial-condition read through outputs."""
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.Run(
        outputtime=outputtime,
        mode=mode,
        advect_chemistry=advect_chemistry,
        before_step_callback=before_step_callback,
        history_callback=history_callback,
        snapshot_callback=snapshot_callback,
        stop_condition=stop_condition,
        step_backend=step_backend,
        step_backend_kwargs=step_backend_kwargs,
    )
