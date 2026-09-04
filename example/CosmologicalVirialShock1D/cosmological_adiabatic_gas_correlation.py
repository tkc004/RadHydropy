"""Adiabatic gas plus live-DM correlation-function collapse experiment.

This is the gas-bearing companion to ``cosmological_dark_matter_only.py``.
It starts at z=100 with the tabulated LCDM correlation-function perturbation,
evolves 1024 live dark-matter shells and the Eulerian gas mesh, and saves the
gas density profile at regular cosmic-time intervals.
"""

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
from example_utils import load_nested_example_config
from radhydropy.gravity import Gravity
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name(
    "cosmological_adiabatic_gas_correlation_z100.yaml"
)


def load_correlation_table(config_filename, par):
    filename = Path(par["linear_correlation_table_filename"])
    if not filename.is_absolute():
        filename = Path(config_filename).resolve().parent / filename
    return et.load_lcdm_correlation_table(filename)


def _snapshot(sim, dm, cosmic_time, cosmology, icparams):
    gas = et.gas_density_profile(sim, cosmic_time, cosmology)
    radii = et.profiles(sim, dm, cosmic_time, cosmology, icparams)
    return gas, radii


def run(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    config = load_nested_example_config(config_filename)
    par = config["par"]
    initial_condition = config["initial_condition"]
    units = CodeUnits.from_mapping(par["units"]["CodeUnits"])
    gravity = par["gravity"]
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(gravity["cosmology_t_ref"]),
        a_ref=float(gravity["cosmology_a_ref"]),
    )
    correlation_table = load_correlation_table(config_filename, par)

    output_dir = Path(par["output"]["savedir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ic_filename = output_dir / "InitialCondition.hdf5"

    initial = et.build_initial_condition(
        {"par": par, "initial_condition": initial_condition}, units, cosmology,
        correlation_table=correlation_table,
    )
    rio.writehdf5(initial, ic_filename)
    dm = et.make_dark_matter(
        initial_condition, units, cosmology,
        correlation_table=correlation_table,
    )

    # The initial density is split explicitly into f_b and 1-f_b.  This
    # check is intentionally printed for this experiment because using the
    # full matter density in both components would double-count gravity.
    baryon_fraction = float(initial_condition["baryon_fraction"])
    gas_mass = float(np.sum(initial.fluid.rho_code * initial.mesh.vol))
    dm_mass = float(np.sum(dm.mass))
    measured_fraction = gas_mass / max(gas_mass + dm_mass, 1.0e-30)
    print("initial gas mass = %.8g code masses" % gas_mass)
    print("initial dark-matter mass = %.8g code masses" % dm_mass)
    print("initial gas fraction = %.8g (configured %.8g)" % (
        measured_fraction, baryon_fraction,
    ))
    if hasattr(initial.fluid, "xHI"):
        print("initial CMB temperature = %.8g K" % float(np.median(
            np.asarray(initial.fluid.temp_code) /
            float(cosmology.scale_factor(float(initial_condition["initial_cosmic_time"])))**2
        )))
        print("initial electron fraction = %.8g" % float(np.median(
            1.0 - np.asarray(initial.fluid.xHI)
        )))
    if not np.isclose(measured_fraction, baryon_fraction, rtol=0.02):
        raise RuntimeError(
            "gas/total initial mass fraction does not match the configured "
            "cosmic baryon fraction"
        )

    local = dict(par)
    local["simulation"] = dict(par["simulation"])
    local["simulation"]["initial_condition_filename"] = str(ic_filename)
    local["output"] = dict(par["output"])
    local["output"].update({
        "directory": str(output_dir), "savedir": str(output_dir),
    })
    local["thermochemistry"] = dict(par.get("thermochemistry", {}))
    local["thermochemistry"].update({
        "metal_pie_enabled": False,
        "cie_cooling": False,
        "thermochemistry_network": "hydrogen",
    })
    sim = Rsim(local)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.fluid.time = float(np.asarray(sim.par.time).flat[0])
    sim.par.gravity = Gravity(
        selfgravity=True,
        cosmological=True,
        cosmology=sim.par.cosmology,
        dark_matter=dm,
        code_units=sim.par.CodeUnits,
    )
    sim.par.dark_matter = dm
    # The gas density already contains only f_b of the homogeneous matter
    # density.  The live shell mass contains only (1-f_b), so its homogeneous
    # contribution must be subtracted with the same fraction in Poisson's
    # equation.
    sim.par.dark_matter_background_fraction = 1.0 - baryon_fraction
    sim.par.gas_background_fraction = baryon_fraction

    initial_time = float(initial_condition["initial_cosmic_time"])
    final_time = float(par["simulation"]["final_time"])
    target_tau = float(cosmology.supercomoving_time(final_time))
    cadence = float(par.get("gas_profile_cadence", 0.10))
    next_snapshot = initial_time
    gas_profiles = []
    radius_history = []

    def save_snapshot(cosmic_time):
        gas, radii = _snapshot(sim, dm, cosmic_time, cosmology, initial_condition)
        gas_profiles.append(gas)
        radius_history.append(radii)

    save_snapshot(initial_time)
    next_snapshot += cadence
    steps = 0
    while float(sim.fluid.time) < target_tau - 1.0e-12:
        cosmic_start = float(
            cosmology.cosmic_time_from_supercomoving(float(sim.fluid.time))
        )
        scale_start = float(cosmology.scale_factor(cosmic_start))
        sim.par.compton_cmb_redshift = 1.0 / scale_start - 1.0
        dt = min(
            float(sim.GetStepTime()),
            target_tau - float(sim.fluid.time),
        )
        sim.Step(dt=dt, mode="hydro_sources")
        steps += 1
        cosmic_time = float(
            cosmology.cosmic_time_from_supercomoving(float(sim.fluid.time))
        )
        if steps == 1 or steps % 500 == 0:
            print(
                "step=%d cosmic_time=%.6g dt=%.6g crossing_dt=%.6g"
                % (steps, cosmic_time, dt, dm.crossing_timestep()),
                flush=True,
            )
        if cosmic_time >= next_snapshot or cosmic_time >= final_time - 1.0e-10:
            save_snapshot(cosmic_time)
            while next_snapshot <= cosmic_time + 1.0e-12:
                next_snapshot += cadence

    times = np.asarray([item["time_Gyr"] for item in gas_profiles])
    radius_comoving = np.asarray(gas_profiles[0]["radius_comoving_kpc"])
    density_proper = np.asarray(
        [item["density_proper_code"] for item in gas_profiles]
    )
    scale_factors = np.asarray([item["scale_factor"] for item in gas_profiles])
    rvir_proper = np.asarray([item["rvir_kpc"] for item in radius_history])
    np.savez(
        output_dir / "AdiabaticGasDensityProfiles.npz",
        time_Gyr=times,
        scale_factor=scale_factors,
        radius_comoving_kpc=radius_comoving,
        density_proper_code=density_proper,
        rvir_proper_kpc=rvir_proper,
        rvir_comoving_kpc=rvir_proper / scale_factors,
        target_radius_kpc=np.asarray([item["rtarget_kpc"] for item in radius_history]),
        mvir=np.asarray([item["mvir"] for item in radius_history]),
        gas_fraction=np.full(times.size, baryon_fraction),
    )
    figure = plot_gas_density_evolution(
        times, radius_comoving, density_proper, rvir_proper,
        scale_factors, output_dir / "AdiabaticGasDensityProfiles.jpg",
    )
    print("steps = %d, dark-matter shells = %d, gas cells = %d" % (
        steps, dm.number_of_shells, int(par["mesh"]["grid_cells"]),
    ))
    print("final cosmic time = %.8g Gyr" % times[-1])
    print("profile data = %s" % (output_dir / "AdiabaticGasDensityProfiles.npz"))
    print("figure = %s" % figure)
    return output_dir / "AdiabaticGasDensityProfiles.npz"


def plot_gas_density_evolution(
    times, radius_comoving, density_proper, rvir_proper, scale_factors,
    filename,
):
    """Plot selected gas profiles and the virial radius at matching times."""
    times = np.asarray(times, dtype=float)
    radius_comoving = np.asarray(radius_comoving, dtype=float)
    density_proper = np.asarray(density_proper, dtype=float)
    rvir_proper = np.asarray(rvir_proper, dtype=float)
    scale_factors = np.asarray(scale_factors, dtype=float)
    selected = np.unique(
        np.linspace(0, times.size - 1, min(9, times.size)).astype(int)
    )
    colors = plt.get_cmap("viridis")(
        np.linspace(0.05, 0.95, selected.size)
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 8.0),
        gridspec_kw={"height_ratios": (3.0, 1.25)},
    )
    axis = axes[0]
    for color, index in zip(colors, selected):
        density = np.maximum(density_proper[index], 1.0e-30)
        label = "t = %.2f Gyr" % times[index]
        axis.loglog(radius_comoving, density, color=color, lw=1.7, label=label)
        if np.isfinite(rvir_proper[index]) and rvir_proper[index] > 0.0:
            axis.axvline(
                rvir_proper[index] / scale_factors[index],
                color=color, ls="--", lw=0.9, alpha=0.65,
            )
    axis.set_ylabel(r"proper gas density [code mass / kpc$^3$]")
    axis.set_title(
        "Adiabatic gas collapse from the z=100 LCDM correlation IC\n"
        "solid: gas density; dashed: corresponding virial radius"
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend(loc="best", fontsize=8, ncol=3)

    axis = axes[1]
    finite = np.isfinite(rvir_proper) & (rvir_proper > 0.0)
    axis.plot(times[finite], rvir_proper[finite], "k.-", label=r"$r_{200}$")
    axis.set_xlabel("cosmic time [Gyr]")
    axis.set_ylabel("proper radius [kpc]")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    run(parser.parse_args().config)


