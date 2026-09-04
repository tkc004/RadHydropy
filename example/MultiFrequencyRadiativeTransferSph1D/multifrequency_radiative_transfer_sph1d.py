"""Pure-hydrogen multifrequency long-characteristic radiation example."""

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
example_root = Path(__file__).resolve().parents[1]
if str(example_root) not in sys.path:
    sys.path.insert(0, str(example_root))
static_dir = Path(__file__).resolve().parents[1] / "StaticStromgrenSpherePhotoheating1D"
if str(static_dir) not in sys.path:
    sys.path.insert(0, str(static_dir))

cache_dir = os.path.join(tempfile.gettempdir(), "radhydropy-cache")
mplconfig_dir = os.path.join(tempfile.gettempdir(), "radhydropy-matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
os.environ.setdefault("MPLCONFIGDIR", mplconfig_dir)

from example_utils import load_nested_example_parameters
from radhydropy.rsim import Rsim
import radhydropy.io as rio
from radhydropy.units import CodeUnits

tools_spec = importlib.util.spec_from_file_location(
    "static_stromgren_photoheating_tools",
    static_dir / "tools.py",
)
tools = importlib.util.module_from_spec(tools_spec)
assert tools_spec.loader is not None
tools_spec.loader.exec_module(tools)


DEFAULT_CONFIG = Path(__file__).with_name(
    "multifrequency_radiative_transfer_sph1d.yaml"
)

def _resolve_reference(config, config_filename, key):
    filename = config.get(key)
    if filename is None:
        return None
    path = Path(filename)
    if not path.is_absolute():
        path = Path(config_filename).resolve().parent / path
    radius_unit = config.get("reference_radius_unit", 5.4 * unyt.kpc)
    return tools.load_log_reference_profile(path, radius_unit)


def _save_plot(output_filename, config, figure_filename, config_filename):
    par, mesh, fluid = tools.load_output_state(output_filename, config)[:3]
    code = CodeUnits.from_mapping(config["CodeUnits"])
    interior = slice(par.noghost, par.noghost + par.nogrid)
    radius_kpc = (
        np.asarray(mesh.coordinate[interior], dtype=float) * code.length_unit
    ).to_value(unyt.kpc)
    xHI = np.asarray(fluid.xHI[interior], dtype=float)
    xHII = np.clip(1.0 - xHI, 1.0e-12, 1.0)
    temperature_cgs_K = (
        np.asarray(fluid.temp_code[interior], dtype=float) * code.temperature_unit
    ).to_value(unyt.K)
    ngamma_values = np.asarray(fluid.ngamma_code, dtype=float)
    if ngamma_values.ndim == 1:
        ngamma_values = ngamma_values[None, :]
    ngamma_cgs_cm3 = (
        ngamma_values[:, interior] * code.number_density_unit
    ).to_value(1.0 / unyt.cm**3)

    xhi_reference = _resolve_reference(
        config, config_filename, "neutral_fraction_reference_filename"
    )
    temperature_reference = _resolve_reference(
        config, config_filename, "temperature_reference_filename"
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.5), sharex=True)
    neutral_line, = axes[0].plot(
        radius_kpc, xHI, color="tab:blue", label=r"$x_{\rm HI}$"
    )
    if xhi_reference is not None:
        axes[0].scatter(
            xhi_reference["radius_kpc"],
            xhi_reference["value"],
            s=16,
            facecolors="none",
            edgecolors="black",
            label="static Stromgren reference",
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$x_{\rm HI}$")
    axes[0].set_ylim(1.0e-6, 1.1)
    axes[0].grid(True, which="both", alpha=0.25)
    ionized_axis = axes[0].twinx()
    ionized_line, = ionized_axis.plot(
        radius_kpc,
        xHII,
        color="tab:orange",
        linestyle="--",
        label=r"$x_{\rm HII}$",
    )
    ionized_axis.set_yscale("log")
    ionized_axis.set_ylabel(r"$x_{\rm HII}$", color="tab:orange")
    ionized_axis.tick_params(axis="y", colors="tab:orange")
    ionized_axis.set_ylim(1.0e-6, 1.1)
    axes[0].legend(
        [neutral_line, ionized_line],
        [neutral_line.get_label(), ionized_line.get_label()],
        frameon=False,
        loc="best",
    )
    axes[1].plot(radius_kpc, temperature_cgs_K, color="tab:red")
    if temperature_reference is not None:
        axes[1].scatter(
            temperature_reference["radius_kpc"],
            temperature_reference["value"],
            s=16,
            facecolors="none",
            edgecolors="black",
            label="static Stromgren reference",
        )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Temperature [K]")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False)
    photon_axis = axes[2]
    for group, values in enumerate(ngamma_cgs_cm3):
        photon_axis.plot(radius_kpc, values, label=f"group {group + 1}")
    photon_axis.set_yscale("log")
    photon_axis.set_xlabel("Radius [kpc]")
    photon_axis.set_ylabel(r"$n_\gamma$ [cm$^{-3}$]")
    photon_axis.grid(True, which="both", alpha=0.25)
    photon_axis.legend(frameon=False)
    network_name = config.get("thermochemistry_network", "hydrogen")
    title = "H/He" if network_name == "hydrogen_helium" else "Pure-H"
    if config.get("metal_pie_enabled", False):
        title += " + metal PIE"
    radiation_temperature = float(
        config.get("stellar_spectrum_blackbody_temperature_cgs_K", 1.0e5)
    )
    fig.suptitle(
        rf"{title} multifrequency radiation "
        rf"($T_{{\rm rad}}={radiation_temperature:.0f}$ K)"
    )
    fig.savefig(figure_filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(config_filename=DEFAULT_CONFIG):
    runparams, icparams = load_nested_example_parameters(config_filename, Path.cwd())
    config = {**runparams, **icparams}
    output_dir = Path.cwd()
    ic_filename = Path(runparams.get("ICfilename", "InitialCondition.hdf5"))
    if not ic_filename.is_absolute():
        ic_filename = output_dir / ic_filename
    runparams["ICfilename"] = str(ic_filename)
    runparams["outdir"] = str(output_dir)
    runparams["savedir"] = str(output_dir)
    config.update(runparams)

    tools.write_initial_condition(config, runparams)
    runtime_only = {
        "absolute_tolerance", "boxsize", "evolution_timestep",
        "explicit_tolerance", "final_time",
        "figure_filename",
        "box_size", "chemistry_timestep", "coordinate_system",
        "current_time", "grid_cells", "plot_filename",
        "hydrogen_initial_collisional_equilibrium", "hydrogen_number_density",
        "initial_temperature", "neutral_fraction_reference_filename",
        "number_of_cells", "reference_radius_unit", "relative_tolerance",
        "temperature_reference_filename", "sigma_gamma", "epsilon_gamma",
        "source_photon_rate", "alpha_B_coefficient",
    }
    runtime = {
        key: value for key, value in runparams.items()
        if key not in runtime_only
    }
    runtime["nogrid"] = runparams["number_of_cells"]
    sim = Rsim(runtime)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.EvolveStaticThermochemistry(
        runparams["final_time"],
        runparams["evolution_timestep"],
    )
    output_filename = output_dir / (
        f"{runparams.get('outfileprefix', 'Output')}_000.hdf5"
    )
    rio.writehdf5(sim, output_filename)
    figure_filename = Path(runparams["savedir"]) / config.get(
        "figure_filename", "MultiFrequencyRadiativeTransferSph1D.jpg"
    )
    _save_plot(output_filename, config, figure_filename, config_filename)
    print(f"output file = {output_filename}")
    print(f"figure = {figure_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
