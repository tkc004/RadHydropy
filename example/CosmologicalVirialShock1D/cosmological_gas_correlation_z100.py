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
    if np.any(finite):
        axes[1].plot(times[finite], virial_radius[finite], "k.-", label=r"$r_{200}$")
    else:
        axes[1].text(
            0.5, 0.5, "no resolved $r_{200}$ yet",
            transform=axes[1].transAxes, ha="center", va="center",
        )
    if times.size > 1:
        axes[1].set_xlim(times[0], times[-1])
    axes[1].set_xlabel("cosmic time [Gyr]")
    axes[1].set_ylabel("proper radius [kpc]")
    axes[1].grid(alpha=0.25)
    if np.any(finite):
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_mass_history(history, filename):
    """Plot masses interior to the measured virial, shock, and disc radii."""
    time = history["time_Gyr"]
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    axis.plot(time, history["mvir"], color="black", lw=1.8,
              label=r"$M(<r_{\rm vir})$")
    axis.plot(time, history["mshock"], color="tab:red", lw=1.8,
              label=r"$M(<r_{\rm shock})$")
    axis.plot(time, history["mdisc"], color="tab:blue", lw=1.8,
              label=r"$M(<r_{\rm disc})$")
    axis.set_yscale("log")
    axis.set_xlabel("cosmic time [Gyr]")
    axis.set_ylabel(r"total mass [$10^{10}\,M_\odot$]")
    axis.set_title("Mass interior to virial, shock, and centrifugal/disc radii\n"
                   "adiabatic gas + live dark matter")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_radius_history(history, filename):
    """Plot the evolving shock, virial, disc, and target-mass radii."""
    time = history["time_Gyr"]
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    axis.plot(time, history["rshock_kpc"], color="tab:red", lw=1.8,
              label=r"$r_{\rm shock}$")
    axis.plot(time, history["rdisc_kpc"], color="tab:blue", lw=1.8,
              label=r"$r_{\rm disc}$")
    axis.plot(time, history["rtarget_kpc"], color="0.45", lw=1.2,
              ls=":", label=r"$r(M_{\rm target})$")
    axis.plot(time, history["rvir_kpc"], color="black", lw=2.0,
              ls="--", marker="o", markevery=max(1, len(time) // 12),
              ms=3.0, label=r"$r_{\rm vir}$")
    axis.set_yscale("log")
    axis.set_xlabel("cosmic time [Gyr]")
    axis.set_ylabel("proper radius [kpc]")
    axis.set_title("Evolution of shock, virial, and disc radii\n"
                   "adiabatic gas + live dark matter")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_temperature_evolution(times, radius, temperature, virial_radius,
                               splashback_radius, scale_factors,
                               virial_temperature, filename):
    """Plot physical gas temperature profiles and the evolving virial radius."""
    selected = np.unique(
        np.linspace(0, len(times) - 1, min(9, len(times))).astype(int)
    )
    colors = plt.get_cmap("plasma")(np.linspace(0.05, 0.95, selected.size))
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 8.0),
        gridspec_kw={"height_ratios": (3.0, 1.25)},
    )
    for color, index in zip(colors, selected):
        proper_radius = radius * scale_factors[index]
        axes[0].loglog(
            proper_radius, np.maximum(temperature[index], 1.0e-30),
            color=color, lw=1.7, label="t = %.2f Gyr" % times[index],
        )
        if np.isfinite(virial_radius[index]) and virial_radius[index] > 0.0:
            axes[0].axvline(
                virial_radius[index],
                color=color, ls="--", lw=0.9, alpha=0.65,
            )
        if np.isfinite(virial_temperature[index]) and virial_temperature[index] > 0.0:
            axes[0].axhline(
                virial_temperature[index], color=color, ls=":", lw=1.0,
                alpha=0.7,
            )
        if np.isfinite(splashback_radius[index]) and splashback_radius[index] > 0.0:
            axes[0].axvline(
                splashback_radius[index], color=color, ls="-.", lw=1.2, alpha=0.85,
            )
            temperature_at_splashback = np.interp(
                splashback_radius[index], proper_radius, temperature[index],
                left=np.nan, right=np.nan,
            )
            if np.isfinite(temperature_at_splashback) and temperature_at_splashback > 0.0:
                axes[0].plot(
                    splashback_radius[index], temperature_at_splashback,
                    marker="s", ms=4.5, color=color, mec="black", mew=0.35,
                    linestyle="None", zorder=5,
    )
    axes[0].set_xlabel("proper radius [kpc]")
    axes[0].set_ylabel("physical gas temperature [K]")
    axes[0].set_title(
        "Gas temperature evolution from the z=100 LCDM IC\n"
        "solid T; dotted Tvir; dashed r200; dash-dot + squares rsp"
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(loc="best", fontsize=8, ncol=3)
    finite = np.isfinite(virial_radius) & (virial_radius > 0.0)
    if np.any(finite):
        axes[1].plot(times[finite], virial_radius[finite], "k.-", label=r"$r_{200}$")
    else:
        axes[1].text(
            0.5, 0.5, "no resolved $r_{200}$ yet",
            transform=axes[1].transAxes, ha="center", va="center",
        )
    if times.size > 1:
        axes[1].set_xlim(times[0], times[-1])
    axes[1].set_xlabel("cosmic time [Gyr]")
    axes[1].set_ylabel("proper radius [kpc]")
    axes[1].grid(alpha=0.25)
    if np.any(finite):
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_temperature_density_evolution(times, density, temperature, filename):
    """Plot gas temperature against physical gas density at each snapshot."""
    selected = np.unique(
        np.linspace(0, len(times) - 1, min(9, len(times))).astype(int)
    )
    colors = plt.get_cmap("plasma")(np.linspace(0.05, 0.95, selected.size))
    fig, axis = plt.subplots(figsize=(7.5, 6.0))
    for color, index in zip(colors, selected):
        rho = np.asarray(density[index], dtype=float)
        temp = np.asarray(temperature[index], dtype=float)
        valid = (
            np.isfinite(rho) & np.isfinite(temp)
            & (rho > 0.0) & (temp > 0.0)
        )
        if np.any(valid):
            order = np.argsort(rho[valid])
            axis.loglog(
                rho[valid][order], temp[valid][order],
                color=color, lw=1.5, marker=".", ms=3.0,
                label="t = %.2f Gyr" % times[index],
            )
    axis.set_xlabel(r"physical gas density [code mass / kpc$^3$]")
    axis.set_ylabel("physical gas temperature [K]")
    axis.set_title("Gas temperature-density evolution")
    axis.grid(alpha=0.25, which="both")
    axis.legend(loc="best", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def plot_dark_matter_density_evolution(dm_profiles, filename, bin_count=128):
    """Plot smoothed mass-binned physical DM profiles versus radius.

    The saved NPZ retains every live shell.  This figure intentionally uses
    fewer radial bins so that one-shell Poisson structure does not look like
    physical density oscillations.
    """
    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    selected = np.unique(
        np.linspace(0, len(dm_profiles) - 1, min(9, len(dm_profiles))).astype(int)
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, selected.size))
    all_comoving = np.concatenate([
        np.asarray(profile["dm_radius_kpc"], dtype=float)
        / float(profile["scale_factor"])
        for profile in dm_profiles
    ])
    plot_bin_count = min(int(bin_count), 48)
    bin_edges = np.geomspace(
        max(1.0e-8, np.nanmin(all_comoving) * 0.9),
        np.nanmax(all_comoving) * 1.1,
        plot_bin_count + 1,
    )
    bin_radii = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    for index, color in zip(selected, colors):
        profile = dm_profiles[index]
        scale_factor = float(profile["scale_factor"])
        radius = np.asarray(profile["dm_radius_kpc"], dtype=float)
        density = np.asarray(profile["dm_density_code"], dtype=float)
        mass = np.asarray(profile["dm_mass"], dtype=float)
        core_mass = float(profile.get("dm_central_core_mass", 0.0))
        core_radius = float(profile.get("dm_central_core_radius_kpc", 0.0)) / scale_factor
        comoving_radius = radius / scale_factor
        valid = (
            np.isfinite(comoving_radius) & np.isfinite(density)
            & np.isfinite(mass) & (comoving_radius > 0.0) & (mass > 0.0)
        )
        shell_radius = comoving_radius[valid]
        shell_mass = mass[valid]
        if shell_radius.size < 1:
            continue
        mass_in_bin, _ = np.histogram(
            shell_radius, bins=bin_edges, weights=shell_mass
        )
        if core_mass > 0.0 and core_radius > 0.0:
            core_bin = int(np.searchsorted(bin_edges, core_radius, side="right") - 1)
            if 0 <= core_bin < mass_in_bin.size:
                mass_in_bin[core_bin] += core_mass
        bin_volume = (
            4.0 * np.pi / 3.0 * scale_factor**3
            * np.diff(bin_edges**3)
        )
        binned_density = mass_in_bin / np.maximum(bin_volume, 1.0e-30)
        valid_bins = binned_density > 0.0
        axis.loglog(
            bin_radii[valid_bins], binned_density[valid_bins],
            color=color, lw=1.6, label="t = %.2f" % profile["time_Gyr"],
        )
    axis.set_xlabel("comoving radius [kpc]")
    axis.set_ylabel(r"dark-matter density [code mass / kpc$^3$]")
    axis.set_title("Live dark-matter density evolution")
    axis.grid(alpha=0.25, which="both")
    axis.legend(title="cosmic time [Gyr]", fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def run(config_filename=DEFAULT_CONFIG, final_time_override=None):
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
    figure_prefix = str(
        runparams.get("figure_prefix", "CosmologicalGasCorrelationZ100")
    )
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
    initial_a = float(cosmology.scale_factor(initial_time))
    sim.par.mu_inflow = float(icparams.get("mu", 0.59))
    minimum_temperature = runparams.get("minimum_temperature", 2.0)
    if hasattr(minimum_temperature, "to_value"):
        minimum_temperature = float(minimum_temperature.to_value("K"))
    else:
        minimum_temperature = float(minimum_temperature)

    transition_redshift = runparams.get("thermochemistry_transition_redshift")
    transition_tau = None
    if transition_redshift is not None:
        transition_redshift = float(transition_redshift)
        transition_scale_factor = 1.0 / (1.0 + transition_redshift)
        transition_time = float(
            cosmology.t_ref
            * (transition_scale_factor / cosmology.a_ref) ** 1.5
        )
        transition_tau = float(cosmology.supercomoving_time(transition_time))

    active_network = None

    def configure_thermochemistry(cosmic_time):
        """Select the pre/post-transition source at the current redshift."""
        nonlocal active_network
        if transition_redshift is None:
            return
        scale_factor = float(cosmology.scale_factor(cosmic_time))
        redshift = max(0.0, 1.0 / scale_factor - 1.0)
        if redshift > transition_redshift:
            sim.par.thermochemistry_network = "hydrogen"
            sim.par.metal_pie_enabled = False
            sim.par.hydrogen_chemistry = True
            sim.par.hydrogen_recombination = True
            sim.par.hydrogen_collisional_ionization = True
            sim.par.hydrogen_atomic_cooling = True
            sim.par.hydrogen_update_mu = True
            sim.par.hydrogen_thermal_coupling = True
            sim.par.compton_cmb_enabled = True
        else:
            sim.par.thermochemistry_network = "pie_uvbg_cooling"
            sim.par.metal_pie_enabled = True
            sim.par.metal_pie_redshift = transition_redshift
            sim.par.hydrogen_chemistry = False
            sim.par.hydrogen_update_mu = False
            sim.par.hydrogen_thermal_coupling = False
            sim.par.compton_cmb_enabled = False
        if sim.par.thermochemistry_network != active_network:
            active_network = sim.par.thermochemistry_network
            print(
                "thermochemistry=%s at z=%.6g"
                % (active_network, redshift),
                flush=True,
            )

    def update_cosmic_boundary(cosmic_time):
        """Set the outer cosmic gas state in supercomoving hydro units."""
        scale_factor = float(cosmology.scale_factor(cosmic_time))
        background_physical = float(cosmology.background_density(cosmic_time))
        temperature_initial = float(icparams["cmb_temperature_0"])
        temperature_physical = temperature_initial * (initial_a / scale_factor) ** 2

        # The boundary is specified physically, then converted explicitly to
        # the hydro representation.  For this gamma=5/3 supercomoving case,
        # rho_code = rho_phys*a^3 and T_code = T_phys*a^2; both happen to be
        # constant for a homogeneous adiabatic background, as they should.
        sim.par.rho_inflow = baryon_fraction * background_physical * scale_factor**3
        sim.par.vel_inflow = 0.0
        sim.par.temp_inflow = temperature_physical * scale_factor**2
        sim.par.compton_cmb_redshift = 1.0 / scale_factor - 1.0
        # Hydro stores supercomoving temperature; keep the physical floor at
        # the configured value as the scale factor changes.
        sim.par.hydro_temperature_floor = minimum_temperature * scale_factor**2

    configure_thermochemistry(initial_time)
    update_cosmic_boundary(initial_time)

    final_time = (
        float(final_time_override)
        if final_time_override is not None
        else float(runparams["final_cosmic_time"])
    )
    target_tau = float(cosmology.supercomoving_time(final_time))
    cadence = float(runparams.get("gas_profile_cadence", 0.10))
    next_snapshot = initial_time
    gas_profiles = []
    radius_history = []
    dm_profiles = []
    steps = 0

    def save_snapshot(cosmic_time):
        gas_profile = et.gas_density_profile(sim, cosmic_time, cosmology)
        first = int(sim.par.noghost)
        last = first + int(sim.par.nogrid)
        scale_factor = float(cosmology.scale_factor(cosmic_time))
        gas_profile["temperature_physical_K"] = (
            np.asarray(sim.fluid.temp[first:last], dtype=float) / scale_factor**2
        )
        gas_profiles.append(gas_profile)
        radius_history.append(et.profiles(sim, dm, cosmic_time, cosmology, icparams))
        dm_profile = et.density_profiles(sim, dm, cosmic_time, cosmology)
        dm_profile["scale_factor"] = scale_factor
        dm_profiles.append(dm_profile)

    save_snapshot(initial_time)
    next_snapshot += cadence
    while float(sim.fluid.time) < target_tau - 1.0e-12:
        cosmic_start = float(
            cosmology.cosmic_time_from_supercomoving(float(sim.fluid.time))
        )
        configure_thermochemistry(cosmic_start)
        update_cosmic_boundary(cosmic_start)
        dt = min(float(sim.GetStepTime()), target_tau - float(sim.fluid.time))
        if transition_tau is not None and float(sim.fluid.time) < transition_tau:
            dt = min(dt, transition_tau - float(sim.fluid.time))
        # This run has active Compton/atomic or PIE thermal sources.  Using
        # hydro-only mode would select the networks but never apply their
        # energy update.
        sim.Step(dt=dt, mode="hydro_sources")
        steps += 1
        cosmic_time = float(
            cosmology.cosmic_time_from_supercomoving(float(sim.fluid.time))
        )
        if steps == 1 or steps % 100 == 0:
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
    radius = np.asarray(gas_profiles[0]["radius_comoving_kpc"])
    density = np.asarray([item["density_proper_code"] for item in gas_profiles])
    temperature = np.asarray(
        [item["temperature_physical_K"] for item in gas_profiles]
    )
    scale_factors = np.asarray([item["scale_factor"] for item in gas_profiles])
    virial_radius = np.asarray([item["rvir_kpc"] for item in radius_history])
    splashback_radius = np.asarray(
        [item["rsplashback_kpc"] for item in radius_history]
    )
    virial_temperature = np.asarray([item["tvir_K"] for item in radius_history])
    history = {
        key: np.asarray([item[key] for item in radius_history])
        for key in radius_history[0]
    }
    data_file = output_dir / (figure_prefix + ".npz")
    np.savez(data_file, **history, scale_factor=scale_factors,
             radius_comoving_kpc=radius, density_proper_code=density,
             temperature_physical_K=temperature,
             rvir_proper_kpc=virial_radius,
             virial_temperature_K=virial_temperature)
    figure = output_dir / (figure_prefix + ".jpg")
    radius_figure = output_dir / (figure_prefix + "_Radii.jpg")
    plot_mass_history(history, figure)
    plot_radius_history(history, radius_figure)
    temperature_figure = output_dir / (figure_prefix + "_Temperatures.jpg")
    plot_temperature_evolution(
        times, radius, temperature, virial_radius, splashback_radius,
        scale_factors,
        virial_temperature,
        temperature_figure,
    )
    temperature_density_figure = output_dir / (
        figure_prefix + "_TemperatureDensity.jpg"
    )
    plot_temperature_density_evolution(
        times, density, temperature, temperature_density_figure,
    )
    dm_figure = output_dir / (figure_prefix + "_DarkMatterDensities.jpg")
    plot_dark_matter_density_evolution(
        dm_profiles, dm_figure,
        bin_count=int(runparams.get("dm_density_bins", 128)),
    )
    dm_data_file = output_dir / (figure_prefix + "_DarkMatterDensities.npz")
    np.savez(
        dm_data_file,
        time_Gyr=np.asarray([item["time_Gyr"] for item in dm_profiles]),
        scale_factor=np.asarray([item["scale_factor"] for item in dm_profiles]),
        radius_kpc=np.asarray([item["dm_radius_kpc"] for item in dm_profiles]),
        density_code=np.asarray([item["dm_density_code"] for item in dm_profiles]),
        mass=np.asarray([item["dm_mass"] for item in dm_profiles]),
        central_core_mass=np.asarray([
            item.get("dm_central_core_mass", 0.0) for item in dm_profiles
        ]),
        central_core_radius_kpc=np.asarray([
            item.get("dm_central_core_radius_kpc", 0.0) for item in dm_profiles
        ]),
    )
    print("initial gas fraction = %.8g" % measured_fraction)
    print("initial gas temperature = %.8g K" % initial_temperature)
    print("final cosmic time = %.8g Gyr" % times[-1])
    print("data = %s" % data_file)
    print("figure = %s" % figure)
    print("radius figure = %s" % radius_figure)
    print("temperature figure = %s" % temperature_figure)
    print("temperature-density figure = %s" % temperature_density_figure)
    print("dark-matter figure = %s" % dm_figure)
    print("dark-matter data = %s" % dm_data_file)
    return data_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--final-time", type=float, default=None,
        help="override final cosmic time in Gyr for a short debug run",
    )
    args = parser.parse_args()
    run(args.config, final_time_override=args.final_time)
