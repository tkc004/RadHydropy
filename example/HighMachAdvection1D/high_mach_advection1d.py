"""Run a uniform high-Mach advection test for the dual-energy scheme."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXAMPLE_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import radhydropy.io as rio

import tools as et


DEFAULT_CONFIG = HERE / "high_mach_advection1d.yaml"


def main(config_filename=DEFAULT_CONFIG, dual_energy=None, pressure_selection=None):
    runparams, icparams = load_example_parameters(config_filename, Path.cwd())
    if dual_energy is not None:
        runparams["dual_energy"] = bool(dual_energy)
        if not dual_energy:
            runparams["savedir"] = str(
                Path(runparams["savedir"]).with_name(
                    Path(runparams["savedir"]).name + "_no_dual_energy"
                )
            )
            runparams["outdir"] = runparams["savedir"]
    if pressure_selection is not None:
        runparams["dual_energy"] = True
        runparams["dual_energy_pressure_selection"] = pressure_selection
        runparams["savedir"] = str(
            Path(runparams["savedir"]).with_name(
                Path(runparams["savedir"]).name + "_conservative_pressure"
            )
        )
        runparams["outdir"] = runparams["savedir"]
    output = Path(runparams["savedir"])
    output.mkdir(parents=True, exist_ok=True)
    code_units = CodeUnits.from_mapping(runparams.get("CodeUnits"))
    initial = et.Simwrap(icparams, code_units=code_units)
    initial.fluid.eos = None
    rio.writehdf5(initial, runparams["ICfilename"])

    sim = Rsim(runparams)
    sim.RunAll()

    snapshots = sorted(output.glob("Output_*.hdf5"))
    history = []
    entropy_radius = None
    entropy_history = []
    density_history = []
    temperature_history = []
    for filename in snapshots:
        state = et.Simwrap(icparams, code_units=code_units)
        rio.readhdf5(state.par, state.mesh, state.fluid, filename)
        radius, density, temperature = et.primitive_profiles(state)
        _, entropy = et.entropy_profile(state)
        if entropy_radius is None:
            entropy_radius = radius
        entropy_history.append(entropy)
        density_history.append(density)
        temperature_history.append(temperature)
        history.append({
            "time": float(np.asarray(state.par.time).flat[0]),
            **et.energy_components(state),
        })
    if not history:
        history = [{"time": float(sim.fluid.time), **et.energy_components(sim)}]

    data = output / "HighMachAdvection1D_EnergyHistory.npz"
    np.savez(
        data,
        time_s=np.asarray([item["time"] for item in history]),
        total_energy=np.asarray([item["total"] for item in history]),
        kinetic_energy=np.asarray([item["kinetic"] for item in history]),
        thermal_energy=np.asarray([item["thermal"] for item in history]),
        pressure_fallback_count=float(sim.solver.dual_energy_pressure_fallback_count),
        synchronization_count=float(sim.solver.dual_energy_synchronization_count),
        floor_count=float(sim.solver.dual_energy_floor_count),
        floor_injected_energy=float(sim.solver.dual_energy_floor_injected_energy),
    )
    entropy_data = output / "HighMachAdvection1D_EntropyHistory.npz"
    entropy_values = np.asarray(entropy_history)
    np.savez(
        entropy_data,
        time_s=np.asarray([item["time"] for item in history]),
        radius=np.asarray(entropy_radius),
        entropy=entropy_values,
        density=np.asarray(density_history),
        temperature=np.asarray(temperature_history),
    )
    times = np.asarray([item["time"] for item in history])
    radius_scale = max(float(np.asarray(icparams["boxsize"])), 1.0)

    def save_profile_map(values, filename, title, colorbar_label):
        values = np.asarray(values)
        log_values = np.full_like(values, np.nan, dtype=float)
        positive = values > 0.0
        log_values[positive] = np.log10(values[positive])
        figure, axis = plt.subplots(figsize=(7.5, 5.0))
        image = axis.pcolormesh(
            np.asarray(entropy_radius) / radius_scale,
            times,
            np.ma.masked_invalid(log_values),
            shading="auto",
            cmap="viridis",
        )
        figure.colorbar(image, ax=axis, label=colorbar_label)
        axis.set_xlabel(r"Radius / $L_{\rm box}$")
        axis.set_ylabel("Time (s)")
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)

    entropy_figure = "HighMachAdvection1D_EntropyEvolution.jpg"
    save_profile_map(
        entropy_values,
        entropy_figure,
        "High-Mach advection entropy evolution",
        r"$\log_{10}[T/\rho^{\gamma-1}]$",
    )
    density_figure = "HighMachAdvection1D_DensityEvolution.jpg"
    save_profile_map(
        density_history,
        density_figure,
        "High-Mach advection density evolution",
        r"$\log_{10}(\rho\,[\mathrm{g\,cm^{-3}}])$",
    )
    temperature_figure = "HighMachAdvection1D_TemperatureEvolution.jpg"
    save_profile_map(
        temperature_history,
        temperature_figure,
        "High-Mach advection temperature evolution",
        r"$\log_{10}(T\,[\mathrm{K}])$",
    )
    initial_energy = history[0]
    final_energy = history[-1]
    print("high-Mach advection energy history = %s" % data)
    print("entropy evolution plot = %s" % (output / entropy_figure))
    print("density evolution plot = %s" % (output / density_figure))
    print("temperature evolution plot = %s" % (output / temperature_figure))
    print("initial Mach estimate > 1e4")
    print("relative total-energy change = %.6e" % (
        (final_energy["total"] - initial_energy["total"])
        / max(abs(initial_energy["total"]), 1.0e-300)
    ))
    print("dual-energy pressure fallbacks = %d" % sim.solver.dual_energy_pressure_fallback_count)
    print("dual-energy synchronizations = %d" % sim.solver.dual_energy_synchronization_count)
    print("dual-energy floor events = %d" % sim.solver.dual_energy_floor_count)
    print("dual-energy floor energy = %.6e" % sim.solver.dual_energy_floor_injected_energy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--without-dual-energy",
        action="store_true",
        help="disable the independent InternalEnergy evolution",
    )
    parser.add_argument(
        "--conservative-pressure",
        action="store_true",
        help="keep dual-energy evolution but always use pressure from E-K",
    )
    args = parser.parse_args()
    main(
        args.config,
        dual_energy=False if args.without_dual_energy else None,
        pressure_selection="conservative" if args.conservative_pressure else None,
    )
