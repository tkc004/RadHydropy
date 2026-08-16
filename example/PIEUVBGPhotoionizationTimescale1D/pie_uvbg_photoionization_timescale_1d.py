"""Test thermal approach to HM12 PIE equilibrium over an ionization timescale."""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import unyt

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[2]
EXAMPLE_ROOT = EXAMPLE_DIR.parents[1]
for path in (PROJECT_ROOT, EXAMPLE_ROOT, EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/radhydropy-matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import radhydropy.io as rio
from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
from radhydropy.thermo_networks.pie import MetalPIETable
from radhydropy.units import CodeUnits

from tools import clean_outputs, load_history


DEFAULT_CONFIG = EXAMPLE_DIR / "pie_uvbg_photoionization_timescale_1d.yaml"


def _equilibrium_temperature(table, density, redshift, metallicity):
    temperatures = np.logspace(
        table.log_temperature[0], table.log_temperature[-1], 4096
    )
    heating, cooling = table.rates(
        temperatures,
        density,
        redshift=redshift,
        metallicity=metallicity,
    )
    net = heating - cooling
    crossings = np.where(net[:-1] * net[1:] <= 0.0)[0]
    if len(crossings) == 0:
        raise RuntimeError("HM12 table has no zero-net-heating temperature")
    index = int(crossings[0])
    log_temperature = np.interp(
        0.0,
        [net[index], net[index + 1]],
        [np.log10(temperatures[index]), np.log10(temperatures[index + 1])],
    )
    return 10.0**log_temperature


def _write_initial_condition(config, runparams, icparams, output_dir):
    from tools import Simwrap

    code_units = CodeUnits.from_mapping(runparams["CodeUnits"])
    ric = Simwrap(icparams, code_units)
    rio.writehdf5(ric, output_dir / "InitialCondition.hdf5")


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    runparams, icparams = load_example_parameters(config_filename)
    output_dir = EXAMPLE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_figure = output_dir / "PIEUVBGPhotoionizationTimescale1D.jpg"
    if legacy_figure.exists():
        legacy_figure.unlink()
    table_filename = str(
        (config_filename.parent / runparams["metal_pie_table_filename"]).resolve()
    )

    table = MetalPIETable(table_filename)
    redshift = float(runparams["metal_pie_redshift"])
    metallicity = float(runparams["metallicity"])
    photoionization_timescale_yr = float(
        runparams["pie_uvbg_photoionization_timescale"].to_value(unyt.yr)
    )
    timesim_yr = float(runparams["timesim"].to_value(unyt.yr))
    output_times_yr = np.array(
        [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0,
         10000.0, 20000.0, 30000.0, 50000.0, 70000.0, 100000.0]
    )
    output_time_file = EXAMPLE_DIR / (
        "pie_uvbg_photoionization_timescale_1d_output_times.txt"
    )

    initial_temperatures = (1.0e3, 1.0e4, 2.0e4, 1.0e5)
    densities = (0.1, 1.0, 10.0)
    results = []
    for density in densities:
        equilibrium_temperature = _equilibrium_temperature(
            table, density, redshift, metallicity
        )
        for initial_temperature in initial_temperatures:
            case_name = f"nH_{density:g}_T_{initial_temperature:g}"
            case_dir = output_dir / case_name
            clean_outputs(case_dir)
            case_runparams = dict(runparams)
            case_runparams["outdir"] = str(case_dir)
            case_runparams["savedir"] = str(case_dir)
            case_runparams["ICfilename"] = str(case_dir / "InitialCondition.hdf5")
            case_runparams["metal_pie_table_filename"] = table_filename
            case_runparams["outputtimefilename"] = str(output_time_file)
            case_icparams = dict(icparams)
            case_icparams["nHini"] = density
            case_icparams["tempini"] = initial_temperature * unyt.K
            _write_initial_condition(
                config_filename, case_runparams, case_icparams, case_dir
            )
            sim = Rsim(case_runparams)
            sim.RunAll(outputtime=0, mode="hydro_sources")
            history = load_history(case_dir)
            if len(history) < 2:
                raise RuntimeError(f"expected evolved snapshots in {case_dir}")
            # The legacy HDF5 time attribute is zero in this cgs setup. The
            # output list is explicit, so reconstruct the physical times of
            # the saved snapshots from that list.
            scheduled_times = np.concatenate(([0.0], output_times_yr, [timesim_yr]))
            time_yr = scheduled_times[:len(history)]
            temperature = np.array([item["temperature_K"] for item in history])
            timescale_ratio = time_yr / photoionization_timescale_yr
            error = np.abs(temperature - equilibrium_temperature) / equilibrium_temperature
            results.append(
                {
                    "label": rf"$n_H={density:g},\ T_0={initial_temperature:.0e}$",
                    "density": density,
                    "initial_temperature": initial_temperature,
                    "equilibrium_temperature": equilibrium_temperature,
                    "time_yr": time_yr,
                    "timescale_ratio": timescale_ratio,
                    "temperature": temperature,
                    "error": error,
                }
            )

    colors = {density: f"C{index}" for index, density in enumerate(densities)}
    linestyles = ["-", "--", ":", "-."]
    for density in densities:
        density_results = [
            result for result in results if result["density"] == density
        ]
        equilibrium_temperature = density_results[0]["equilibrium_temperature"]
        fig, (ax_temp, ax_error) = plt.subplots(
            2, 1, figsize=(8, 7), sharex=True
        )
        for result in density_results:
            linestyle = linestyles[
                initial_temperatures.index(result["initial_temperature"])
            ]
            label = rf"$T_0={result['initial_temperature']:.0e}\ {{\rm K}}$"
            ax_temp.plot(
                result["time_yr"],
                result["temperature"],
                color="tab:blue",
                linestyle=linestyle,
                linewidth=1.4,
                label=label,
            )
            ax_error.plot(
                result["time_yr"],
                result["error"],
                color="tab:red",
                linestyle=linestyle,
                linewidth=1.2,
                label=label,
            )
        color = colors[density]
        ax_temp.axhline(
            equilibrium_temperature,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=rf"PIE equilibrium: $T={equilibrium_temperature:.3g}\ {{\rm K}}$",
        )
        ax_temp.plot(
            [photoionization_timescale_yr, 10.0 * photoionization_timescale_yr],
            [equilibrium_temperature, equilibrium_temperature],
            color=color,
            linestyle="None",
            marker="s",
            markersize=5,
        )
        ax_temp.axvline(
            photoionization_timescale_yr,
            color="0.5",
            linestyle="--",
            linewidth=1.0,
            label=r"$\tau_{\rm pi}$",
        )
        ax_temp.axvline(
            10.0 * photoionization_timescale_yr,
            color="0.5",
            linestyle=":",
            linewidth=1.0,
            label=r"$10\tau_{\rm pi}$",
        )
        ax_error.axhline(
            0.01,
            color="0.5",
            linestyle="--",
            linewidth=1.0,
            label="1%",
        )
        ax_error.axvline(
            photoionization_timescale_yr,
            color="0.5",
            linestyle="--",
            linewidth=1.0,
        )
        ax_error.axvline(
            10.0 * photoionization_timescale_yr,
            color="0.5",
            linestyle=":",
            linewidth=1.0,
        )
        ax_temp.set_xscale("log")
        ax_temp.set_yscale("log")
        ax_error.set_xscale("log")
        ax_error.set_yscale("log")
        ax_temp.set_ylabel("temperature [K]")
        ax_error.set_xlabel("time [yr]")
        ax_error.set_ylabel(r"$|T-T_{\rm PIE}|/T_{\rm PIE}$")
        ax_temp.grid(alpha=0.25)
        ax_error.grid(alpha=0.25)
        ax_temp.legend(frameon=False, fontsize=8, ncol=2)
        ax_error.legend(frameon=False, fontsize=8, ncol=2)
        fig.suptitle(rf"HM12 PIE timescale test: $n_H={density:g}\ {{\rm cm^{{-3}}}}$")
        fig.tight_layout()
        figure = output_dir / f"PIEUVBGPhotoionizationTimescale1D_nH_{density:g}.jpg"
        fig.savefig(figure, dpi=180)
        plt.close(fig)

    for result in results:
        one_tau = int(np.argmin(np.abs(result["timescale_ratio"] - 1.0)))
        ten_tau = int(np.argmin(np.abs(result["timescale_ratio"] - 10.0)))
        print(
            f"{result['label']}: T_PIE={result['equilibrium_temperature']:.6e} K, "
            f"error(1 tau)={result['error'][one_tau]:.6e}, "
            f"error(10 tau)={result['error'][ten_tau]:.6e}"
        )
    print(f"figures = {output_dir}/PIEUVBGPhotoionizationTimescale1D_nH_*.jpg")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().config)
