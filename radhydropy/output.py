"""Simulation output scheduling helpers."""

import time

import numpy as np
import unyt

from radhydropy.units import code_unit_scales


def write_numbered_hdf5(sim, outindex):
    """Write ``Output_###.hdf5`` for the supplied simulation."""
    filename = (
        sim.par.output.directory
        + '/'
        + sim.par.output.filename_prefix
        + '_%03d' % outindex
        + '.hdf5'
    )
    from radhydropy.io import writehdf5
    writehdf5(sim, filename)


def hdf5_output_callback(sim, outputtime=0, output_state=None):
    """Return a callback that writes HDF5 snapshots at fixed cadence."""
    if output_state is None:
        output_state = {
            'outtime': 0.0 * sim.par.simulation.final_time,
            'outindex': 1,
            'last_output_time_s': float(np.asarray(sim.fluid.time, dtype=float)),
        }
    else:
        output_state.setdefault(
            'outtime',
            0.0 * sim.par.simulation.final_time,
        )
        output_state.setdefault('outindex', 1)
        output_state.setdefault(
            'last_output_time_s',
            float(np.asarray(sim.fluid.time, dtype=float)),
        )

    def callback(sim, step):
        dt = step["dt"]
        if getattr(dt, "shape", None) == (1,):
            dt = dt[0]
        if getattr(sim.par, 'verbose', 0) >= 1:
            print("time, dt", sim.fluid.time, dt)
        if output_state['outtime'] >= sim.par.output.cadence:
            sim.fluid.SetTemperature()
            write_numbered_hdf5(sim, output_state['outindex'])
            output_state['last_output_time_s'] = float(
                np.asarray(sim.fluid.time, dtype=float)
            )
            output_state['outtime'] = 0.0 * sim.par.simulation.final_time
            output_state['outindex'] += 1
        else:
            output_state['outtime'] += dt

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
):
    """Run a simulation using an explicit output-time list."""
    start = time.time()
    if output_writer is None:
        output_writer = write_numbered_hdf5
    if step_backend is None:
        step_backend = sim.Step
    if step_backend_kwargs is None:
        step_backend_kwargs = {}
    print("--- Initization finished. Start running ... ---")
    print("--- %s seconds ---" % (time.time() - start))
    output_writer(sim, 0)
    last_output_time_s = float(np.asarray(sim.fluid.time, dtype=float))

    current_time = sim.fluid.time
    final_time = sim.par.simulation.final_time
    time_tol = max(abs(float(np.asarray(final_time, dtype=float))) * 1.0e-12, 1.0e-30)
    from radhydropy.io import load_output_time_list
    output_times = load_output_time_list(getattr(sim.par, 'outputtimefilename', None))
    if output_times is None:
        output_times = []
    else:
        if hasattr(final_time, 'units'):
            target_unit = final_time.units
            sorted_values = np.unique(
                np.asarray(output_times.to_value(target_unit), dtype=float)
            )
        else:
            code_units = getattr(sim.par, 'CodeUnits', None)
            if code_units is None:
                code_units = getattr(
                    getattr(sim.par, 'units', None), 'CodeUnits', None
                )
            sorted_values = np.unique(
                np.asarray(output_times.to_value(unyt.s), dtype=float)
                / code_unit_scales(code_units)['time_s']
            )
        output_times = [
            value * final_time.units if hasattr(final_time, 'units') else value
            for value in sorted_values
            if (
                (value * final_time.units if hasattr(final_time, 'units') else value)
                > current_time
                and (value * final_time.units if hasattr(final_time, 'units') else value)
                <= final_time
            )
        ]

    outindex = 1
    for target_time in output_times:
        if stop_condition is not None and stop_condition(sim):
            break
        target_time_value = float(np.asarray(target_time, dtype=float))
        while float(np.asarray(sim.fluid.time, dtype=float)) < target_time_value - time_tol:
            if stop_condition is not None and stop_condition(sim):
                break
            dt = sim.GetStepTime(final_time=target_time)
            if getattr(sim.par, 'verbose', 0) >= 1:
                print("time, dt", sim.fluid.time, dt)
            step_backend(
                dt=dt,
                mode=mode,
                advect_chemistry=advect_chemistry,
                **step_backend_kwargs,
            )
        if stop_condition is not None and stop_condition(sim):
            break
        # Euler/source steps can cross a target by a roundoff- or CFL-sized
        # amount.  Treat the first state at or beyond the target as the
        # requested snapshot instead of silently dropping the output.
        if float(np.asarray(sim.fluid.time, dtype=float)) >= target_time_value - time_tol:
            sim.fluid.SetTemperature()
            output_writer(sim, outindex)
            last_output_time_s = float(np.asarray(sim.fluid.time, dtype=float))
            outindex += 1

    final_time_value = float(np.asarray(final_time, dtype=float))
    while float(np.asarray(sim.fluid.time, dtype=float)) < final_time_value - time_tol:
        if stop_condition is not None and stop_condition(sim):
            break
        dt = sim.GetStepTime(final_time=final_time)
        if getattr(sim.par, 'verbose', 0) >= 1:
            print("time, dt", sim.fluid.time, dt)
        step_backend(
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
            **step_backend_kwargs,
        )

    if stop_condition is not None and abs(float(np.asarray(sim.fluid.time, dtype=float)) - last_output_time_s) > time_tol:
        sim.fluid.SetTemperature()
        output_writer(sim, outindex)

    print("--- Simulation finished. ---")
    print("--- %s seconds ---" % (time.time() - start))
