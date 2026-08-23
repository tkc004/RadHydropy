"""Adiabatic gas collapse from the z=100 LCDM correlation-function IC."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.example_config import load_example_parameters
from radhydropy.gravity import Gravity
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name(
    "cosmological_gas_correlation_z100.yaml"
)


def load_correlation_table(config_filename, runparams):
    filename = Path(runparams["linear_correlation_table_filename"])
    if not filename.is_absolute():
        filename = Path(config_filename).resolve().parent / filename
    return et.load_lcdm_correlation_table(filename)


def plot_density_evolution(times, radius, density, virial_radius, scale_factors,
                           filename):
    selected = np.unique(
        np.linspace(0, len(times) - 1, min(9, len(times))).astype(int)
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, selected.size))
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 8.0),
        gridspec_kw={"height_ratios": (3.0, 1.25)},
    )
    for color, index in zip(colors, selected):
        axes[0].loglog(radius, np.maximum(density[index], 1.0e-30),
                       color=color, lw=1.7, label="t = %.2f Gyr" % times[index])
        if np.isfinite(virial_radius[index]) and virial_radius[index] > 0.0:
            axes[0].axvline(
                virial_radius[index] / scale_factors[index],
                color=color, ls="--", lw=0.9, alpha=0.65,
            )
    axes[0].set_ylabel(r"proper gas density [code mass / kpc$^3$]")
    axes[0].set_title(
        "Adiabatic gas collapse from the z=100 LCDM correlation IC\n"
        "solid: gas density; dashed: corresponding virial radius"
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(loc="best", fontsize=8, ncol=3)
    finite = np.isfinite(virial_radius) & (virial_radius > 0.0)
    axes[1].plot(times[finite], virial_radius[finite], "k.-", label=r"$r_{200}$")
    axes[1].set_xlabel("cosmic time [Gyr]")
    axes[1].set_ylabel("proper radius [kpc]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def run(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    runparams, icparams = load_example_parameters(config_filename)
    units = CodeUnits.from_mapping(runparams["CodeUnits"])
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(runparams["cosmology_t_ref"]),
        a_ref=float(runparams["cosmology_a_ref"]),
    )
    correlation_table = load_correlation_table(config_filename, runparams)
    output_dir = Path(runparams["savedir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ic_filename = output_dir / "InitialCondition.hdf5"

    initial = et.Simwrap(
        icparams, units, cosmology, correlation_table=correlation_table
    )
    rio.writehdf5(initial, ic_filename)
    dm = et.make_dark_matter(
        icparams, units, cosmology, correlation_table=correlation_table
    )

    baryon_fraction = float(icparams["baryon_fraction"])
    gas_mass = float(np.sum(initial.fluid.rho * initial.mesh.vol))
    dm_mass = float(np.sum(dm.mass))
    measured_fraction = gas_mass / max(gas_mass + dm_mass, 1.0e-30)
    if not np.isclose(measured_fraction, baryon_fraction, rtol=0.02):
        raise RuntimeError(
            "initial gas/total mass fraction does not match baryon_fraction"
        )
    initial_temperature = float(np.median(initial.fluid.temp)) / float(
        cosmology.scale_factor(float(icparams["initial_cosmic_time"]))
    ) ** 2
    expected_temperature = float(icparams["cmb_temperature_0"]) * (
        1.0 / float(cosmology.scale_factor(float(icparams["initial_cosmic_time"])))
    )
    if not np.isclose(initial_temperature, expected_temperature, rtol=1.0e-8):
        raise RuntimeError("initial gas temperature is not the z=100 CMB temperature")

    local = dict(runparams)
    local.update({"ICfilename": str(ic_filename), "outdir": str(output_dir),
                  "savedir": str(output_dir)})
    sim = Rsim(local)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.fluid.time = float(np.asarray(sim.par.time).flat[0])
    sim.par.gravity = Gravity(
        selfgravity=True, cosmological=True, cosmology=sim.par.cosmology,
        dark_matter=dm, code_units=sim.par.CodeUnits,
    )
    sim.par.dark_matter = dm
    sim.par.dark_matter_background_fraction = 1.0 - baryon_fraction
    sim.par.gas_background_fraction = baryon_fraction

    initial_time = float(icparams["initial_cosmic_time"])
    final_time = float(runparams["final_cosmic_time"])
    target_tau = float(cosmology.supercomoving_time(final_time))
    cadence = float(runparams.get("gas_profile_cadence", 0.10))
    next_snapshot = initial_time
    gas_profiles = []
    radius_history = []

    def save_snapshot(cosmic_time):
        gas_profiles.append(et.gas_density_profile(sim, cosmic_time, cosmology))
        radius_history.append(et.profiles(sim, dm, cosmic_time, cosmology, icparams))

    save_snapshot(initial_time)
    next_snapshot += cadence
    while float(sim.fluid.time) < target_tau - 1.0e-12:
        dt = min(float(sim.GetStepTime()), target_tau - float(sim.fluid.time))
        sim.Step(dt=dt, mode="hydro")
        cosmic_time = float(
            cosmology.cosmic_time_from_supercomoving(float(sim.fluid.time))
        )
        if cosmic_time >= next_snapshot or cosmic_time >= final_time - 1.0e-10:
            save_snapshot(cosmic_time)
            while next_snapshot <= cosmic_time + 1.0e-12:
                next_snapshot += cadence

    times = np.asarray([item["time_Gyr"] for item in gas_profiles])
    radius = np.asarray(gas_profiles[0]["radius_comoving_kpc"])
    density = np.asarray([item["density_proper_code"] for item in gas_profiles])
    scale_factors = np.asarray([item["scale_factor"] for item in gas_profiles])
    virial_radius = np.asarray([item["rvir_kpc"] for item in radius_history])
    data_file = output_dir / "CosmologicalGasCorrelationZ100.npz"
    np.savez(data_file, time_Gyr=times, scale_factor=scale_factors,
             radius_comoving_kpc=radius, density_proper_code=density,
             rvir_proper_kpc=virial_radius)
    figure = output_dir / "CosmologicalGasCorrelationZ100.jpg"
    plot_density_evolution(times, radius, density, virial_radius, scale_factors, figure)
    print("initial gas fraction = %.8g" % measured_fraction)
    print("initial gas temperature = %.8g K" % initial_temperature)
    print("final cosmic time = %.8g Gyr" % times[-1])
    print("data = %s" % data_file)
    print("figure = %s" % figure)
    return data_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    run(parser.parse_args().config)
