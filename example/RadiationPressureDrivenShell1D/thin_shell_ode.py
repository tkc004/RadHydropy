"""Run the RadHydropy fixed-mass thin-shell radiation-pressure example."""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = PROJECT_ROOT / "example"
for path in (PROJECT_ROOT, EXAMPLE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "radhydropy-matplotlib")
)

import example_utils as eu
import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits


DEFAULT_CONFIG = Path(__file__).resolve().with_name("thin_shell_ode.yaml")
SPEED_OF_LIGHT = unyt.c.to_value(unyt.cm / unyt.s)


def _write_initial_condition(runparams, icparams):
    """Write a one-cell, fixed-mass shell IC in the normal example format."""
    code = CodeUnits.from_mapping(runparams["units"]["CodeUnits"])
    shell_mass = icparams["shell_mass"].to_value(unyt.g)
    par = SimpleNamespace(
        CodeUnits=code,
        units=SimpleNamespace(CodeUnits=code),
        unit_system=code.unit_system,
        coordsys=runparams["simulation"]["coordinate_system"],
        nogrid=runparams["mesh"]["grid_cells"],
        noghost=runparams["mesh"]["ghost_cells"],
        boxsize=icparams["box_size"],
        time=0.0 * unyt.s,
    )
    par.simulation = SimpleNamespace(
        current_time=par.time,
        box_size=icparams["box_size"],
        coordinate_system=runparams["simulation"]["coordinate_system"],
    )
    par.mesh = SimpleNamespace(
        grid_cells=runparams["mesh"]["grid_cells"],
        ghost_cells=runparams["mesh"]["ghost_cells"],
    )
    boxsize_cm = icparams["box_size"].to_value(unyt.cm)
    volume_cgs_cm3 = boxsize_cm**3
    mesh = SimpleNamespace(
        boundary=np.array([0.0, boxsize_cm]) * unyt.cm,
    )
    fluid = SimpleNamespace(
        rho_code=np.array([shell_mass / volume_cgs_cm3]) * unyt.g / unyt.cm**3,
        vel_code=np.array([0.0]) * unyt.cm / unyt.s,
        temp_code=np.array([icparams["temperature"].to_value(unyt.K)]) * unyt.K,
        mu=np.array([1.0]),
    )
    rio.writehdf5(
        SimpleNamespace(par=par, mesh=mesh, fluid=fluid),
        runparams["simulation"]["initial_condition_filename"],
    )


def _source_step(sim, shell_state, luminosity, photon_energy_cgs_erg, dt, **kwargs):
    """Advance one source-only RadHydropy timestep.

    The shell has one fixed control volume.  We intentionally do not call a
    hydrodynamic step: this removes gas-pressure and boundary contributions
    from the thin-shell momentum test while retaining the normal Rsim loop.
    """
    sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
    volume = float(np.asarray(sim.mesh.vol[sim.par.noghost], dtype=float))
    absorbed_rate = luminosity / photon_energy_cgs_erg / volume
    source_result = {
        "source_steps": 1,
        "absorbed_photon_rate": np.array([absorbed_rate]),
        "photon_energy_cgs_erg": np.array([photon_energy_cgs_erg]),
        "direction": 1,
    }
    sim.solver.ApplyRadiationPressure(
        dt, sim.mesh, sim.fluid, sim.par, source_result
    )
    sim._sync_hydro_state()
    sim.fluid.time += dt
    interior = sim.par.noghost
    shell_state["velocity"] = float(sim.fluid.vel_code[interior])
    shell_state["radius"] += shell_state["velocity"] * float(dt)
    return {"dt": dt, "hydro_steps": 0, "source_steps": 1}


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)
    runparams = config['par']
    icparams = config["initial_condition"]
    eu.clean_previous_outputs(runparams)
    Path(runparams["output"]["directory"]).mkdir(parents=True, exist_ok=True)
    _write_initial_condition(runparams, icparams)

    sim = Rsim(runparams)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    luminosity = runparams["radiation"]["radiation_pressure_source_luminosity"].to_value(
        unyt.erg / unyt.s
    )
    photon_energy_cgs_erg = (20.0 * unyt.eV).to_value(unyt.erg)
    shell_mass = icparams["shell_mass"].to_value(unyt.g)
    shell_state = {
        "radius": icparams["initial_radius"].to_value(unyt.cm),
        "velocity": 0.0,
    }
    history = {"time": [], "radius": [], "momentum": []}

    def record(simulation):
        interior = simulation.par.noghost
        history["time"].append(float(simulation.fluid.time))
        history["radius"].append(shell_state["radius"])
        history["momentum"].append(float(simulation.fluid.Mom_code[interior]))

    record(sim)

    def step_backend(dt, **kwargs):
        result = _source_step(
            sim,
            shell_state,
            luminosity,
            photon_energy_cgs_erg,
            dt,
            **kwargs,
        )
        record(sim)
        return result

    sim.Run(
        outputtime=0,
        mode="sources",
        step_backend=step_backend,
    )

    time_s = np.asarray(history["time"]) * float(
        (1.0 * sim.par.CodeUnits.time_unit).to_value(unyt.s)
    )
    radius_cgs_cm = np.asarray(history["radius"])
    momentum = np.asarray(history["momentum"]) * float(
        (1.0 * sim.par.CodeUnits.momentum_unit).to_value(unyt.g * unyt.cm / unyt.s)
    )
    force = luminosity / SPEED_OF_LIGHT
    expected_momentum = force * time_s
    acceleration = force / shell_mass
    expected_radius = radius_cgs_cm[0] + 0.5 * acceleration * time_s**2
    relative_error = np.divide(
        momentum - expected_momentum,
        expected_momentum,
        out=np.zeros_like(momentum),
        where=expected_momentum != 0.0,
    )

    figure = Path(runparams["output"]["savedir"]) / "RadiationPressureDrivenShell1D_ThinShellODE.jpg"
    time_myr = time_s / (1.0 * unyt.Myr).to_value(unyt.s)
    pc_cm = (1.0 * unyt.pc).to_value(unyt.cm)
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.0), sharex=True)
    axes[0].plot(time_myr, radius_cgs_cm / pc_cm, label="RadHydropy")
    axes[0].plot(time_myr, expected_radius / pc_cm, "--", label="exact thin-shell")
    axes[0].set_ylabel("shell radius [pc]")
    axes[1].plot(time_myr, momentum, label="RadHydropy shell momentum")
    axes[1].plot(time_myr, expected_momentum, "--", label=r"$Lt/c$")
    axes[1].set_ylabel(r"momentum [g cm s$^{-1}$]")
    axes[2].plot(time_myr, relative_error, label="relative error")
    axes[2].set_ylabel("momentum relative error")
    axes[2].set_xlabel("time [Myr]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print("final momentum relative error = %.6e" % relative_error[-1])
    print("figure = %s" % figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
