"""Rsim execution subsystem helpers."""

import time
import radhydropy.io as rio

def Evolve(
    sim,
    final_time=None,
    mode="hydro_sources",
    advect_chemistry=True,
    history_callback=None,
    output_callback=None,
    stop_condition=None,
    step_backend=None,
    step_backend_kwargs=None,
):
    """Evolve the simulation with a pluggable step backend."""
    if final_time is None:
        final_time = sim.par.simulation.final_time
    if step_backend is None:
        step_backend = sim.Step
    if step_backend_kwargs is None:
        step_backend_kwargs = {}
    counters = {"hydro_steps": 0, "source_steps": 0}
    if history_callback is not None:
        history_callback(sim)
    while sim.fluid.time < final_time:
        if stop_condition is not None and stop_condition(sim):
            break
        dt = sim.GetStepTime(final_time=final_time)
        step = step_backend(
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
            **step_backend_kwargs,
        )
        counters["hydro_steps"] += step["hydro_steps"]
        counters["source_steps"] += step["source_steps"]
        if history_callback is not None:
            history_callback(sim)
        if output_callback is not None:
            output_callback(sim, step)
    return counters

def Run(
    sim,
    outputtime=0,
    mode="hydro_sources",
    advect_chemistry=True,
    stop_condition=None,
    step_backend=None,
    step_backend_kwargs=None,
):
    """Run the simulation loop and write periodic HDF5 outputs."""
    sim.WriteUsedParameters()
    if getattr(sim.par, 'outputtimefilename', None):
        rio.run_with_output_times(
            sim,
            outputtime=outputtime,
            mode=mode,
            advect_chemistry=advect_chemistry,
            stop_condition=stop_condition,
            step_backend=step_backend,
            step_backend_kwargs=step_backend_kwargs,
        )
        return
    # Fixed-cadence output path: advance to `timesim` and write snapshots
    # whenever `outtime` reaches `outdeltatime`.
    print("--- Initization finished. Start running ... ---") 
    print("--- %s seconds ---" % (
        time.time() - getattr(sim, "_start_time", time.time())
    ))
    rio.write_numbered_hdf5(sim, 0)
    sim.Evolve(
        final_time=sim.par.simulation.final_time,
        mode=mode,
        advect_chemistry=advect_chemistry,
        output_callback=rio.hdf5_output_callback(
            sim,
            outputtime=outputtime,
        ),
        stop_condition=stop_condition,
        step_backend=step_backend,
        step_backend_kwargs=step_backend_kwargs,
    )
    if stop_condition is not None:
        sim.fluid.SetTemperature()
        rio.write_numbered_hdf5(sim, 0)
    print("--- Simulation finished. ---") 
    print("--- %s seconds ---" % (
        time.time() - getattr(sim, "_start_time", time.time())
    ))

def RunAll(
    sim,
    outputtime=0,
    mode="hydro_sources",
    advect_chemistry=True,
    stop_condition=None,
    step_backend=None,
    step_backend_kwargs=None,
):
    """Run the full workflow from initial-condition read through outputs."""
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.Run(
        outputtime=outputtime,
        mode=mode,
        advect_chemistry=advect_chemistry,
        stop_condition=stop_condition,
        step_backend=step_backend,
        step_backend_kwargs=step_backend_kwargs,
    )
