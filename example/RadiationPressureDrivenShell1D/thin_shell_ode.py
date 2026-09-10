"""Run the RadHydropy fixed-mass thin-shell radiation-pressure example."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

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
from radhydropy.arrays import as_named_array
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import quantity_to_value


DEFAULT_CONFIG = Path(__file__).resolve().with_name("thin_shell_ode.yaml")
SPEED_OF_LIGHT = unyt.c.to_value(unyt.cm / unyt.s)


def _build_initial_condition(config):
    """Write a one-cell, fixed-mass shell IC in the normal example format."""

    initial = config['initial_condition']
    sim = Rsim(config["par"])
    code = sim.par.units.CodeUnits
    grid_cells = int(config["par"]['mesh']['grid_cells'])
    if grid_cells != 1:
        raise ValueError('thin-shell IC requires exactly one active grid cell')
    sim.par.simulation.box_size_proper_code = quantity_to_value(
        initial['box_size_proper'], code.length_unit
    )
    sim.par.simulation.time_proper_code = 0.0
    boundary_proper_code = as_named_array(quantity_to_value(
        np.array([0.0, initial['box_size_proper'].to_value(unyt.cm)]) * unyt.cm,
        code.length_unit,
    ))
    area_proper_code = quantity_to_value(config["par"]['mesh']['area_proper'], code.area_unit)
    width_proper_code = np.diff(boundary_proper_code)
    volume_proper_code = width_proper_code * area_proper_code
    sim.mesh.boundary_proper_code = boundary_proper_code
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1]),
        boundary_proper_code=boundary_proper_code,
        width_proper_code=width_proper_code,
        area_proper_code=np.ones(1) * area_proper_code,
        volume_proper_code=volume_proper_code,
    )
    shell_mass = initial['shell_mass'].to_value(unyt.g)
    sim.fluid.rho_proper_code = as_named_array(
        np.array([shell_mass]) / volume_proper_code
    )
    sim.fluid.vel_proper_code = as_named_array(np.zeros(1, dtype=float))
    sim.fluid.temp_proper_code = as_named_array(quantity_to_value(
        np.array([initial['temperature_proper']]), code.temperature_unit
    ))
    sim.fluid.mu = np.ones(1)
    sim.fluid.SetFluidTime(0.0)
    sim.fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    sim.fluid.SetPressure()
    sim.fluid._refresh_runtime_state()
    return sim


def _write_initial_condition(config):
    sim = _build_initial_condition(config)
    rio.writehdf5(sim, config['par']['simulation']['initial_condition_filename'])


def _source_step(
    sim, shell_state, luminosity_cgs_erg_s, photon_energy_cgs_erg, dt, **kwargs
):
    """Advance one source-only RadHydropy timestep.

    The shell has one fixed control volume_proper_code.  We intentionally do not call a
    hydrodynamic step: this removes gas-pressure and boundary_proper_code contributions
    from the thin-shell momentum test while retaining the normal Rsim loop.
    """
    sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
    interior = sim.par.mesh.ghost_cells
    volume_proper_code = float(np.asarray(sim.mesh.geometry_state.volume_proper_code[interior], dtype=float))
    absorbed_rate = luminosity_cgs_erg_s / photon_energy_cgs_erg / volume_proper_code
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
    sim.fluid.time_proper_code += dt
    interior = sim.par.mesh.ghost_cells
    shell_state["vel_proper_code"] = float(sim.fluid.vel_proper_code[interior])
    shell_state["radius_proper_code"] += (
        shell_state["vel_proper_code"] * float(dt)
    )
    return {"dt": dt, "hydro_steps": 0, "source_steps": 1}


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)

    initial = config['initial_condition']
    eu.clean_previous_outputs(config)
    Path(config["par"]["output"]["directory"]).mkdir(parents=True, exist_ok=True)
    _write_initial_condition(config)

    sim = Rsim(config["par"])
    rio.readhdf5(
        sim.par,
        sim.mesh,
        sim.fluid,
        config["par"]['simulation']['initial_condition_filename'],
    )
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    code = sim.par.units.CodeUnits
    luminosity_cgs_erg_s = config["par"]["radiation"]["radiation_pressure_source_luminosity"].to_value(
        unyt.erg / unyt.s
    )
    photon_energy_cgs_erg = (20.0 * unyt.eV).to_value(unyt.erg)
    shell_mass_cgs_g = initial["shell_mass"].to_value(unyt.g)
    shell_state = {
        "radius_proper_code": quantity_to_value(
            initial["radius_shell_initial_proper"], code.length_unit
        ),
        "vel_proper_code": 0.0,
    }
    history = {
        "time_proper_code": [],
        "radius_proper_code": [],
        "momentum_proper_code": [],
    }

    def record(simulation):
        interior = simulation.par.mesh.ghost_cells
        history["time_proper_code"].append(
            float(simulation.fluid.time_proper_code)
        )
        history["radius_proper_code"].append(
            shell_state["radius_proper_code"]
        )
        history["momentum_proper_code"].append(
            float(simulation.fluid.Mom_code[interior])
        )

    record(sim)

    def step_backend(dt, **kwargs):
        result = _source_step(
            sim,
            shell_state,
            luminosity_cgs_erg_s,
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

    time_proper_cgs_s = np.asarray(history["time_proper_code"]) * float(
        (1.0 * sim.par.units.CodeUnits.time_unit).to_value(unyt.s)
    )
    radius_proper_cgs_cm = np.asarray(history["radius_proper_code"]) * float(
        (1.0 * sim.par.units.CodeUnits.length_unit).to_value(unyt.cm)
    )
    momentum_proper_cgs_g_cm_s = np.asarray(
        history["momentum_proper_code"]
    ) * float(
        (1.0 * sim.par.units.CodeUnits.momentum_unit).to_value(unyt.g * unyt.cm / unyt.s)
    )
    force_cgs_dyn = luminosity_cgs_erg_s / SPEED_OF_LIGHT
    expected_momentum_proper_cgs_g_cm_s = force_cgs_dyn * time_proper_cgs_s
    acceleration_proper_cgs_cm_s2 = force_cgs_dyn / shell_mass_cgs_g
    expected_radius_proper_cgs_cm = (
        radius_proper_cgs_cm[0]
        + 0.5 * acceleration_proper_cgs_cm_s2 * time_proper_cgs_s**2
    )
    relative_momentum_error_dimensionless = np.divide(
        momentum_proper_cgs_g_cm_s - expected_momentum_proper_cgs_g_cm_s,
        expected_momentum_proper_cgs_g_cm_s,
        out=np.zeros_like(momentum_proper_cgs_g_cm_s),
        where=expected_momentum_proper_cgs_g_cm_s != 0.0,
    )

    figure = Path(config["par"]["output"]["directory"]) / "RadiationPressureDrivenShell1D_ThinShellODE.jpg"
    time_proper_myr = time_proper_cgs_s / (1.0 * unyt.Myr).to_value(unyt.s)
    pc_cm = (1.0 * unyt.pc).to_value(unyt.cm)
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.0), sharex=True)
    axes[0].plot(time_proper_myr, radius_proper_cgs_cm / pc_cm, label="RadHydropy")
    axes[0].plot(time_proper_myr, expected_radius_proper_cgs_cm / pc_cm, "--", label="exact thin-shell")
    axes[0].set_ylabel("shell radius [pc]")
    axes[1].plot(time_proper_myr, momentum_proper_cgs_g_cm_s, label="RadHydropy shell momentum")
    axes[1].plot(time_proper_myr, expected_momentum_proper_cgs_g_cm_s, "--", label=r"$Lt/c$")
    axes[1].set_ylabel(r"momentum [g cm s$^{-1}$]")
    axes[2].plot(time_proper_myr, relative_momentum_error_dimensionless, label="relative error")
    axes[2].set_ylabel("momentum relative error")
    axes[2].set_xlabel("time [Myr]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print("final momentum relative error = %.6e" % relative_momentum_error_dimensionless[-1])
    print("figure = %s" % figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
