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

import example_utils as eu  # noqa: E402

import radhydropy.io as rio  # noqa: E402
import tools as et  # noqa: E402
from radhydropy.rsim import Rsim  # noqa: E402
from radhydropy.units import CodeUnits, quantity_to_value  # noqa: E402

DEFAULT_CONFIG = HERE / "high_mach_advection1d.yaml"


def main(config_filename=DEFAULT_CONFIG, dual_energy=None, pressure_selection=None):
    config = eu.load_nested_example_config(config_filename)

    initial_condition = config["initial_condition"]
    exampleparams = config["example"]
    if dual_energy is not None:
        config["par"]["hydrodynamics"]["dual_energy"] = bool(dual_energy)
        if not dual_energy:
            config["par"]["output"]["directory"] = str(
                Path(config["par"]["output"]["directory"]).with_name(
                    Path(config["par"]["output"]["directory"]).name + "_no_dual_energy",
                ),
            )
            config["par"]["output"]["directory"] = config["par"]["output"]["directory"]
    if pressure_selection is not None:
        config["par"]["hydrodynamics"]["dual_energy"] = True
        config["par"]["hydrodynamics"]["dual_energy_pressure_selection"] = pressure_selection
        config["par"]["output"]["directory"] = str(
            Path(config["par"]["output"]["directory"]).with_name(
                Path(config["par"]["output"]["directory"]).name + "_conservative_pressure",
            ),
        )
        config["par"]["output"]["directory"] = config["par"]["output"]["directory"]
    output = Path(config["par"]["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    code_units = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    config["_code_units"] = code_units
    initial = et.build_initial_condition(config)
    rio.writehdf5(initial, config["par"]["simulation"]["initial_condition_filename"])

    sim = Rsim(config["par"])
    sim.RunAll()

    snapshots = sorted(output.glob("Output_*.hdf5"))
    history = []
    entropy_radius_proper_code = None
    entropy_history = []
    rho_proper_code_history = []
    temp_proper_code_history = []
    for filename in snapshots:
        state = et.build_initial_condition(config)
        rio.readhdf5(state.par, state.mesh, state.fluid, filename)
        radius_proper_code, rho_proper_code, temp_proper_code = et.primitive_profiles(state)
        _, entropy = et.entropy_profile(state)
        if entropy_radius_proper_code is None:
            entropy_radius_proper_code = radius_proper_code
        entropy_history.append(entropy)
        rho_proper_code_history.append(rho_proper_code)
        temp_proper_code_history.append(temp_proper_code)
        history.append(
            {
                "time_proper_code": float(np.asarray(state.fluid.time_proper_code).flat[0]),
                **et.energy_components(state),
            },
        )
    if not history:
        history = [
            {"time_proper_code": float(sim.fluid.time_proper_code), **et.energy_components(sim)},
        ]

    data = output / "HighMachAdvection1D_EnergyHistory.npz"
    np.savez(
        data,
        time_s=np.asarray([item["time_proper_code"] for item in history]),
        total_energy_proper_code=np.asarray([item["total_energy_proper_code"] for item in history]),
        kinetic_energy_proper_code=np.asarray(
            [item["kinetic_energy_proper_code"] for item in history],
        ),
        thermal_energy_proper_code=np.asarray(
            [item["thermal_energy_proper_code"] for item in history],
        ),
        pressure_fallback_count=float(sim.solver.dual_energy_pressure_fallback_count),
        synchronization_count=float(sim.solver.dual_energy_synchronization_count),
        floor_count=float(sim.solver.dual_energy_floor_count),
        floor_injected_energy=float(sim.solver.dual_energy_floor_injected_energy),
    )
    entropy_data = output / "HighMachAdvection1D_EntropyHistory.npz"
    entropy_values = np.asarray(entropy_history)
    np.savez(
        entropy_data,
        time_s=np.asarray([item["time_proper_code"] for item in history]),
        radius_proper_code=np.asarray(entropy_radius_proper_code),
        entropy=entropy_values,
        rho_proper_code=np.asarray(rho_proper_code_history),
        temp_proper_code=np.asarray(temp_proper_code_history),
    )
    times = np.asarray([item["time_proper_code"] for item in history])
    radius_scale_proper_code = max(
        quantity_to_value(initial_condition["box_size_proper"], code_units.length_unit),
        1.0,
    )

    def save_profile_map(values, filename, title, colorbar_label, **image_kwargs):
        values = np.asarray(values)
        log_values = np.full_like(values, np.nan, dtype=float)
        positive = values > 0.0
        log_values[positive] = np.log10(values[positive])
        figure, axis = plt.subplots(figsize=(7.5, 5.0))
        image = axis.pcolormesh(
            np.asarray(entropy_radius_proper_code) / radius_scale_proper_code,
            times,
            np.ma.masked_invalid(log_values),
            shading="auto",
            cmap="viridis",
            **image_kwargs,
        )
        figure.colorbar(image, ax=axis, label=colorbar_label)
        axis.set_xlabel(r"Radius / $L_{\rm box}$")
        axis.set_ylabel("Time (s)")
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)

    # Entropy is undefined only in exact vacuum.  Keep positive-density cells
    # visible: floor-dominated regions are part of this dual-energy test.  The
    # fixed limits expose those regions as saturated colors without allowing
    # extreme vacuum values to hide the gas-side entropy structure.
    entropy_plot_values = entropy_values.copy()
    density_history = np.asarray(rho_proper_code_history)
    entropy_plot_values[density_history <= 0.0] = np.nan
    entropy_figure = exampleparams.get(
        "entropy_plot_filename",
        "HighMachAdvection1D_EntropyEvolution.jpg",
    )
    save_profile_map(
        entropy_plot_values,
        entropy_figure,
        "High-Mach advection entropy evolution",
        r"$\log_{10}[T/\rho^{\gamma-1}]$",
        vmin=-2.0,
        vmax=2.0,
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
        np.asarray(temp_proper_code_history),
        temperature_figure,
        "High-Mach advection temperature evolution",
        r"$\log_{10}(T\,[\mathrm{K}])$",
    )
    history[0]
    history[-1]


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
