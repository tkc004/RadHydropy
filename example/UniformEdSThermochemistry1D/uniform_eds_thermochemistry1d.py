"""Few-cell uniform EdS Compton/atomic thermo-chemistry comparison."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT.parent))

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter
import copy
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
from tools import UniformEdSInitialCondition, analytic_compton_temperature
import example_utils as eu


CONFIG = EXAMPLE_ROOT / "uniform_eds_thermochemistry1d.yaml"


def run_case(runparams, icparams, units, cosmology, atomic_cooling):
    case = copy.deepcopy(runparams)
    label = "atomic_compton" if atomic_cooling else "compton_only"
    case["simulation"]["name"] = f"UniformEdSThermochemistry1D_{label}"
    case["simulation"]["initial_condition_filename"] = str(EXAMPLE_ROOT / f"{label}_InitialCondition.hdf5")
    case["output"]["filename_prefix"] = f"{label}_Output"
    case["thermochemistry"]["hydrogen_atomic_cooling"] = atomic_cooling
    case["output"]["directory"] = str(EXAMPLE_ROOT / "outputs")
    case["output"]["savedir"] = case["output"]["directory"]
    Path(case["output"]["directory"]).mkdir(parents=True, exist_ok=True)
    source_dt = float(case["_example"].get("source_timestep", 2.0))
    case.pop("_example", None)

    initial = UniformEdSInitialCondition(icparams, case["mesh"], units, cosmology)
    rio.writehdf5(initial, case["simulation"]["initial_condition_filename"])

    sim = Rsim(case)
    sim.par.cosmology = cosmology
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.fluid.SetFluidTime(sim.par.time)
    sim.SetInitFluid()
    sim.par.cosmology = cosmology

    # Rsim.Run normally obtains an outer timestep from the hydro CFL
    # estimator.  This is a source-only uniform-cell benchmark, so provide a
    # fixed outer timestep while retaining the Rsim.Run execution path.
    def fixed_step_time(dt=None, final_time=None):
        if dt is not None:
            return float(dt)
        remaining = float(final_time) - float(np.asarray(sim.fluid.time).flat[0])
        return min(source_dt, remaining)

    sim.GetStepTime = fixed_step_time

    physical = slice(sim.par.mesh.ghost_cells, sim.par.mesh.ghost_cells + sim.par.mesh.grid_cells)
    history = {
        "time_s": [],
        "temperature_K": [],
        "scale_factor": [],
        "source_solver": [],
        "relative_change": [],
    }

    def reset_conserved_from_temperature():
        """Keep the wrapper state in primitive/conserved sync between source steps.

        The benchmark deliberately exercises the existing source API.  Its
        source callback updates the primitive temperature, while this example
        restores the uniform-cell conserved arrays from that temperature
        before the next source-only step.  No library implementation is
        modified by this example-local adapter.
        """
        sim.fluid.SetPressure()
        sim.solver.SetConserved(sim.mesh, sim.fluid)

    def step_backend(**kwargs):
        cosmic_time = float(np.asarray(sim.fluid.time).flat[0])
        scale_factor = float(sim.par.cosmology.scale_factor(cosmic_time))
        sim.par.compton_cmb_redshift = 1.0 / scale_factor - 1.0
        reset_conserved_from_temperature()
        result = sim.Step(**kwargs)
        cosmic_time = float(np.asarray(sim.fluid.time).flat[0])
        scale_factor = float(sim.par.cosmology.scale_factor(cosmic_time))
        history["time_s"].append(
            cosmic_time * float(sim.par.units.CodeUnits.time_unit.to_value("s"))
        )
        history["scale_factor"].append(scale_factor)
        history["temperature_K"].append(
            float(np.mean(sim.fluid.temp[physical]))
        )
        history["source_solver"].append(result.get("source_solver", "explicit"))
        history["relative_change"].append(
            float(result.get("relative_change", 0.0))
        )
        if not np.all(np.isfinite(sim.fluid.temp[physical])):
            raise RuntimeError(
                f"non-finite temperature at cosmic time {cosmic_time:.8g}, "
                f"redshift {sim.par.compton_cmb_redshift:.8g}; "
                f"temperature={sim.fluid.temp[physical]}"
            )
        return result

    sim.Run(mode="sources", step_backend=step_backend)
    history = {key: np.asarray(value) for key, value in history.items()}
    history["xHI"] = np.asarray(sim.fluid.xHI[physical], dtype=float)
    return history, sim, physical


def main():
    config = eu.load_nested_example_config(CONFIG)
    runparams, icparams = config["par"], config["initial_condition"]
    runparams["_example"] = config["example"]
    units = CodeUnits.from_mapping(runparams["units"]["CodeUnits"])
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(runparams["gravity"]["cosmology_t_ref"]),
        a_ref=float(runparams["gravity"]["cosmology_a_ref"]),
    )

    compton, sim, physical = run_case(
        runparams, icparams, units, cosmology, atomic_cooling=False
    )
    atomic, _, _ = run_case(
        runparams, icparams, units, cosmology, atomic_cooling=True
    )

    initial_time_s = (
        float(icparams["initial_cosmic_time"])
        * float(units.time_unit.to_value("s"))
    )
    plot_time_s = np.linspace(
        initial_time_s, np.max(compton["time_s"]), 200
    )
    analytic = analytic_compton_temperature(
        compton["time_s"],
        float(icparams["temperature_K"]),
        float(icparams["initial_cosmic_time"]),
        cosmology,
        float(units.time_unit.to_value("s")),
        float(icparams["hydrogen_density_cm3"]),
        float(icparams["hydrogen_mass_fraction"]),
        float(icparams["xHI"]),
        float(runparams["hydrodynamics"]["gamma"]),
        float(runparams["thermochemistry"]["cmb_temperature_0"].to_value("K")),
        1.0 / (float(icparams["hydrogen_mass_fraction"]) * (2.0 - float(icparams["xHI"]))),
    )
    analytic_plot = analytic_compton_temperature(
        plot_time_s,
        float(icparams["temperature_K"]),
        float(icparams["initial_cosmic_time"]),
        cosmology,
        float(units.time_unit.to_value("s")),
        float(icparams["hydrogen_density_cm3"]),
        float(icparams["hydrogen_mass_fraction"]),
        float(icparams["xHI"]),
        float(runparams["hydrodynamics"]["gamma"]),
        float(runparams["thermochemistry"]["cmb_temperature_0"].to_value("K")),
        1.0 / (float(icparams["hydrogen_mass_fraction"]) * (2.0 - float(icparams["xHI"]))),
    )
    error = np.max(np.abs(compton["temperature_K"] - analytic) / analytic)
    print(f"Compton-only maximum relative error: {error:.6e}")
    for label, history in (("Compton-only", compton), ("atomic+Compton", atomic)):
        choices, counts = np.unique(history["source_solver"], return_counts=True)
        summary = ", ".join(
            f"{choice}={count}" for choice, count in zip(choices, counts)
        )
        print(f"{label} hybrid source choices: {summary}")
    if error > 2.0e-3:
        raise RuntimeError("Compton-only EdS comparison failed")
    if not np.all(np.isfinite(atomic["temperature_K"])):
        raise RuntimeError("atomic+Compton run produced non-finite temperature")
    if atomic["temperature_K"][-1] >= compton["temperature_K"][-1]:
        raise RuntimeError("atomic cooling did not cool below Compton-only run")

    figure = Path(runparams["output"]["savedir"]) / "UniformEdSThermochemistry1D.jpg"
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(plot_time_s / (1.0e6 * 365.25 * 86400.0), analytic_plot, "k-", label="EdS analytic Compton")
    plt.plot(compton["time_s"] / (1.0e6 * 365.25 * 86400.0), compton["temperature_K"], "o", ms=3, label="Rsim Compton")
    plt.plot(atomic["time_s"] / (1.0e6 * 365.25 * 86400.0), atomic["temperature_K"], "--", label="Rsim atomic + Compton")
    plt.xlabel("cosmic time [Myr]")
    plt.ylabel("physical temperature [K]")
    plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure, dpi=180)
    plt.close()
    print(f"figure = {figure}")


if __name__ == "__main__":
    main()
