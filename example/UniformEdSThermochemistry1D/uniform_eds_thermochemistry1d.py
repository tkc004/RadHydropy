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

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
from tools import UniformEdSInitialCondition, analytic_compton_temperature


CONFIG = EXAMPLE_ROOT / "uniform_eds_thermochemistry1d.yaml"


def run_case(runparams, icparams, units, cosmology, atomic_cooling):
    case = dict(runparams)
    label = "atomic_compton" if atomic_cooling else "compton_only"
    case["simname"] = f"UniformEdSThermochemistry1D_{label}"
    case["ICfilename"] = str(EXAMPLE_ROOT / f"{label}_InitialCondition.hdf5")
    case["outfileprefix"] = f"{label}_Output"
    case["hydrogen_atomic_cooling"] = atomic_cooling
    case["outdir"] = str(EXAMPLE_ROOT / "outputs")
    case["savedir"] = case["outdir"]
    Path(case["outdir"]).mkdir(parents=True, exist_ok=True)

    initial = UniformEdSInitialCondition(icparams, units, cosmology)
    rio.writehdf5(initial, case["ICfilename"])

    sim = Rsim(case)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.fluid.SetFluidTime(sim.par.time)
    sim.SetInitFluid()

    # Rsim.Run normally obtains an outer timestep from the hydro CFL
    # estimator.  This is a source-only uniform-cell benchmark, so provide a
    # fixed outer timestep while retaining the Rsim.Run execution path.
    source_dt = float(case.get("source_timestep", 2.0))

    def fixed_step_time(dt=None, final_time=None):
        if dt is not None:
            return float(dt)
        remaining = float(final_time) - float(np.asarray(sim.fluid.time).flat[0])
        return min(source_dt, remaining)

    sim.GetStepTime = fixed_step_time

    physical = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    history = {"time_s": [], "temperature_K": [], "scale_factor": []}

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
            cosmic_time * float(sim.par.CodeUnits.time_unit.to_value("s"))
        )
        history["scale_factor"].append(scale_factor)
        history["temperature_K"].append(
            float(np.mean(sim.fluid.temp[physical]))
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
    runparams, icparams = load_example_parameters(CONFIG, EXAMPLE_ROOT)
    units = CodeUnits.from_mapping(runparams["CodeUnits"])
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(runparams["cosmology_t_ref"]),
        a_ref=float(runparams["cosmology_a_ref"]),
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
        float(runparams["gamma"]),
        float(runparams["cmb_temperature_0"].to_value("K")),
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
        float(runparams["gamma"]),
        float(runparams["cmb_temperature_0"].to_value("K")),
        1.0 / (float(icparams["hydrogen_mass_fraction"]) * (2.0 - float(icparams["xHI"]))),
    )
    error = np.max(np.abs(compton["temperature_K"] - analytic) / analytic)
    print(f"Compton-only maximum relative error: {error:.6e}")
    if error > 2.0e-3:
        raise RuntimeError("Compton-only EdS comparison failed")
    if not np.all(np.isfinite(atomic["temperature_K"])):
        raise RuntimeError("atomic+Compton run produced non-finite temperature")
    if atomic["temperature_K"][-1] >= compton["temperature_K"][-1]:
        raise RuntimeError("atomic cooling did not cool below Compton-only run")

    figure = Path(runparams["savedir"]) / "UniformEdSThermochemistry1D.jpg"
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
