"""Cosmological spherical-collapse comparison: adiabatic versus PIE cooling."""

import argparse
import copy
import os
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
from radhydropy.cosmology import EinsteinDeSitter, LambdaCDM
from example_utils import load_nested_example_parameters
from radhydropy.gravity import Gravity
from radhydropy.rsim import Rsim
from radhydropy.thermo_networks.pie import MetalPIETable
from radhydropy.units import CodeUnits
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name("cosmological_virial_shock1d.yaml")


def load_correlation_table(config_filename, runparams):
    filename = runparams.get("linear_correlation_table_filename")
    if not filename:
        return None
    filename = Path(filename)
    if not filename.is_absolute():
        filename = Path(config_filename).resolve().parent / filename
    return et.load_lcdm_correlation_table(filename)


def run_case(
    runparams, icparams, units, cosmology, table, radiative,
    correlation_table=None,
):
    case = "radiative" if radiative else "adiabatic"
    case_dir = Path(runparams["savedir"]) / case
    case_dir.mkdir(parents=True, exist_ok=True)
    local = copy.deepcopy(runparams)
    local.update({
        "savedir": str(case_dir), "outdir": str(case_dir),
        "ICfilename": str(case_dir / "InitialCondition.hdf5"),
        # Load the PIE table at startup so the network can switch to it at
        # z=10 without reconstructing the Rsim parameter object.
        "metal_pie_enabled": bool(radiative),
        "cie_cooling": bool(radiative),
        "thermochemistry_network": "cie_cooling" if radiative else "hydrogen",
    })
    initial = et.Simwrap(
        icparams, units, cosmology, table,
        correlation_table=correlation_table,
    )
    rio.writehdf5(initial, local["ICfilename"])
    dm = et.make_dark_matter(
        icparams, units, cosmology, correlation_table=correlation_table
    )

    sim = Rsim(local)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    # ``SetInitFluid`` initializes the hydro state time to zero; cosmological
    # ICs carry a non-zero (often negative) supercomoving start time.
    sim.fluid.time = float(np.asarray(sim.par.time).flat[0])
    sim.par.gravity = Gravity(
        selfgravity=True, cosmological=True, cosmology=sim.par.cosmology,
        dark_matter=dm, code_units=sim.par.CodeUnits,
    )
    sim.par.dark_matter = dm
    sim.par.dark_matter_background_fraction = 1.0 - float(icparams["baryon_fraction"])
    sim.par.gas_background_fraction = float(icparams["baryon_fraction"])

    t0 = float(icparams["initial_cosmic_time"])
    tf = float(runparams["final_cosmic_time"])
    target = float(cosmology.supercomoving_time(tf))
    cadence = float(runparams["snapshot_cadence"])
    next_output = t0
    history = []
    while float(sim.fluid.time) < target - 1.0e-12:
        tau = float(sim.fluid.time)
        cosmic_time = float(cosmology.cosmic_time_from_supercomoving(tau))
        if radiative:
            a = float(cosmology.scale_factor(cosmic_time))
            redshift = max(0.0, 1.0 / a - 1.0)
            if redshift > float(icparams["uv_background_on_redshift"]):
                sim.par.thermochemistry_network = "cie_cooling"
                sim.par.cie_cooling = True
                sim.par.metal_pie_enabled = True
            else:
                sim.par.thermochemistry_network = "pie_uvbg_cooling"
                sim.par.cie_cooling = False
                sim.par.metal_pie_enabled = True
                sim.par.metal_pie_redshift = redshift
        dt = float(sim.GetStepTime())
        dt = min(dt, target - tau)
        sim.Step(dt=dt, mode="hydro_sources" if radiative else "hydro")
        cosmic_time = float(cosmology.cosmic_time_from_supercomoving(float(sim.fluid.time)))
        if cosmic_time >= next_output or cosmic_time >= tf - 1.0e-10:
            history.append(et.profiles(sim, dm, cosmic_time, cosmology, icparams))
            next_output += cadence

    result = {key: np.asarray([row[key] for row in history]) for key in history[0]}
    np.savez(case_dir / "mass_radius_history.npz", **result)
    final_profile = et.density_profiles(sim, dm, cosmic_time, cosmology)
    np.savez(case_dir / "density_profile_final.npz", **final_profile)
    return result, final_profile


def plot_histories(histories, filename):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    for axis, (label, history) in zip(axes, (("Adiabatic", histories["adiabatic"]), ("Radiative PIE", histories["radiative"]))):
        time = history["time_Gyr"]
        axis.plot(time, history["mvir"], color="black", label=r"$M(<r_{\rm vir})$")
        axis.plot(time, history["mshock"], color="tab:red", label=r"$M(<r_{\rm shock})$")
        axis.plot(time, history["mdisc"], color="tab:blue", label=r"$M(<r_{\rm disc})$")
        axis.set_yscale("log")
        axis.set_ylabel(r"total mass [$10^{10}\,M_\odot$]")
        axis.set_title(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("cosmic time [Gyr]")
    fig.suptitle("Mass interior to virial, shock, and centrifugal/disc radii")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_radius_histories(histories, filename):
    """Plot the three evolving radii used by the enclosed-mass diagnostic."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    for axis, (label, history) in zip(
        axes,
        (("Adiabatic", histories["adiabatic"]), ("Radiative CIE → PIE", histories["radiative"])),
    ):
        time = history["time_Gyr"]
        # The virial and disc radii can coincide when the centrifugal
        # balance lies outside the measured halo.  Draw r_vir last, with a
        # dashed line and markers, so it cannot disappear underneath r_disc.
        axis.plot(time, history["rshock_kpc"], color="tab:red", lw=1.8,
                  label=r"$r_{\rm shock}$", zorder=2)
        axis.plot(time, history["rdisc_kpc"], color="tab:blue", lw=1.8,
                  label=r"$r_{\rm disc}$", zorder=2)
        axis.plot(time, history["rtarget_kpc"], color="0.45", lw=1.2,
                  ls=":", label=r"$r(M_{\rm target})$", zorder=1)
        axis.plot(time, history["rvir_kpc"], color="black", lw=2.0,
                  ls="--", marker="o", markevery=max(1, len(time) // 12),
                  ms=3.0, label=r"$r_{\rm vir}$", zorder=4)
        axis.set_ylabel("radius [kpc]")
        axis.set_title(label)
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("cosmic time [Gyr]")
    fig.suptitle("Evolution of shock, virial, and disc radii")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_density_profiles(profiles, filename):
    """Plot final physical gas and dark-matter density profiles."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    for axis, (label, profile) in zip(
        axes, (("Adiabatic", profiles["adiabatic"]),
               ("Radiative CIE → PIE", profiles["radiative"])),
    ):
        axis.loglog(profile["gas_radius_kpc"], profile["gas_density_code"],
                    color="tab:red", label="gas")
        axis.loglog(profile["dm_radius_kpc"], profile["dm_density_code"],
                    color="black", marker="o", ms=2.5, linestyle="None",
                    label="dark matter shells")
        axis.set_ylabel(r"density [code mass / kpc$^3$]")
        axis.set_title(label + " at t = %.2f Gyr" % profile["time_Gyr"])
        axis.grid(alpha=0.25, which="both")
        axis.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("proper radius [kpc]")
    fig.suptitle("Final radial gas and dark-matter density profiles")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def main(config_filename=DEFAULT_CONFIG):
    runparams, icparams = load_nested_example_parameters(config_filename)
    units = CodeUnits.from_mapping(runparams["CodeUnits"])
    if runparams.get("cosmology_type") in ("lambda_cdm", "LambdaCDM", "lcdm"):
        cosmology = LambdaCDM.from_code_units(
            units, t_ref=float(runparams["cosmology_t_ref"]),
            a_ref=float(runparams["cosmology_a_ref"]),
            omega_m=float(runparams["cosmology_omega_m"]),
            omega_lambda=float(runparams["cosmology_omega_lambda"]),
            hubble_ref=float(runparams["cosmology_hubble_ref"]),
        )
    else:
        cosmology = EinsteinDeSitter.from_code_units(
            units, t_ref=float(runparams["cosmology_t_ref"]),
            a_ref=float(runparams["cosmology_a_ref"]),
        )
    table_path = Path(runparams["metal_pie_table_filename"])
    if not table_path.is_absolute():
        table_path = Path(config_filename).parent / table_path
    table = MetalPIETable(table_path)
    runparams["metal_pie_table_filename"] = str(table_path.resolve())
    correlation_table = load_correlation_table(config_filename, runparams)
    outputs = {
        "adiabatic": run_case(
            runparams, icparams, units, cosmology, table, False,
            correlation_table=correlation_table,
        ),
        "radiative": run_case(
            runparams, icparams, units, cosmology, table, True,
            correlation_table=correlation_table,
        ),
    }
    histories = {key: value[0] for key, value in outputs.items()}
    density_profiles = {key: value[1] for key, value in outputs.items()}
    figure = Path(runparams["savedir"]) / "CosmologicalVirialShock1D.jpg"
    plot_histories(histories, figure)
    radius_figure = Path(runparams["savedir"]) / "CosmologicalVirialShock1D_Radii.jpg"
    plot_radius_histories(histories, radius_figure)
    density_figure = Path(runparams["savedir"]) / "CosmologicalVirialShock1D_Densities.jpg"
    plot_density_profiles(density_profiles, density_figure)
    print("figure = %s" % figure)
    print("radius figure = %s" % radius_figure)
    print("density figure = %s" % density_figure)
    print("histories = %s" % (Path(runparams["savedir"]) / "{adiabatic,radiative}" / "mass_radius_history.npz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
