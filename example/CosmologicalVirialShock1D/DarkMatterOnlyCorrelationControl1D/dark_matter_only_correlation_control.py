"""Gas-free DM control for the z=100 correlation-function IC."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = EXAMPLE_DIR.parent
PROJECT_ROOT = EXAMPLE_ROOT.parent
sys.path[:0] = [str(EXAMPLE_DIR), str(EXAMPLE_ROOT), str(PROJECT_ROOT)]

from example_utils import load_nested_example_config
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.units import CodeUnits, quantity_to_value
import virial_shock_tools as et


CONFIG = Path(__file__).with_name("dark_matter_only_correlation_control.yaml")


def main(config_filename=CONFIG):
    config_filename = Path(config_filename).resolve()
    config = load_nested_example_config(config_filename)
    par = config["par"]
    ic = config["initial_condition"]
    units = CodeUnits.from_mapping(par["units"]["CodeUnits"])
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=quantity_to_value(par["cosmology"]["cosmology_t_ref"], units.time_unit),
        a_ref=float(par["cosmology"]["cosmology_a_ref"]),
    )
    table_path = config["example"]["correlation_table_filename"]
    table_path = (config_filename.parent / table_path).resolve()
    table = et.load_lcdm_correlation_table(table_path)
    config["_code_unit_system"] = units
    config["_cosmology"] = cosmology
    config["_correlation_table"] = table
    config["_dark_matter_softening"] = float(par["dark_matter"]["softening"])
    shells = et.make_dark_matter(config)

    initial = quantity_to_value(ic["time_cosmic"], units.time_unit)
    final = quantity_to_value(par["simulation"]["final_time"], units.time_unit)
    tau = float(cosmology.supercomoving_time(initial))
    final_tau = float(cosmology.supercomoving_time(final))
    timestep = float(par["dark_matter"]["timestep"])
    initial_mass = shells.total_mass
    times = [initial]
    radii = [float(cosmology.scale_factor(initial)) * shells.radius.copy()]
    masses = [shells.mass.copy()]

    while tau < final_tau - 1.0e-12:
        dt = min(timestep, final_tau - tau)
        start = float(cosmology.cosmic_time_from_supercomoving(tau))
        end = float(cosmology.cosmic_time_from_supercomoving(tau + dt))
        a_start = float(cosmology.scale_factor(start))
        a_end = float(cosmology.scale_factor(end))
        rho_comoving = float(cosmology.background_density(start)) * a_start**3
        background = lambda radius_comoving_code, rho_comoving_code=rho_comoving: (
            4.0 * np.pi / 3.0 * rho_comoving_code * np.asarray(radius_comoving_code, dtype=float)**3
        )
        shells.step(
            dt,
            crossing_safety_factor=float(par["dark_matter"]["crossing_safety_factor"]),
            background_enclosed_mass=background,
            scale_factor=a_start,
            scale_factor_end=a_end,
            cosmological=True,
        )
        tau += dt
        times.append(end)
        radii.append(a_end * shells.radius.copy())
        masses.append(shells.mass.copy())

    output_dir = config_filename.parent / par["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(times)
    radii = np.asarray(radii)
    masses = np.asarray(masses)
    mean_density = np.asarray([
        float(cosmology.background_density(time_cosmic_code)) for time_cosmic_code in times
    ])
    density_contrast = np.empty_like(radii)
    enclosed_mass = np.empty_like(radii)
    density_plot_radius = []
    density_plot_contrast = []
    for row, (radius_comoving_code, mass_comoving_code) in enumerate(zip(radii, masses)):
        order = np.argsort(radius_comoving_code)
        radius_comoving_code = radius_comoving_code[order]
        mass_comoving_code = mass_comoving_code[order]
        edges = np.empty(radius_comoving_code.size + 1)
        edges[1:-1] = np.sqrt(radius_comoving_code[:-1] * radius_comoving_code[1:])
        edges[0] = radius_comoving_code[0]**2 / edges[1]
        edges[-1] = radius_comoving_code[-1]**2 / edges[-2]
        shell_volume = 4.0 * np.pi / 3.0 * np.diff(edges**3)
        rho_comoving_code = np.divide(
            mass_comoving_code, shell_volume,
            out=np.full_like(mass_comoving_code, np.inf), where=shell_volume > 0.0,
        )
        density_contrast[row] = rho_comoving_code / mean_density[row]
        enclosed_mass[row] = np.cumsum(mass_comoving_code)

        # The shell-by-shell profile becomes visually dominated by sampling
        # noise after shell crossing. Aggregate only the diagnostic profile
        # into logarithmic radial bins; retain the full-resolution enclosed
        # mass data and panel.
        bin_edges = np.geomspace(
            max(radius_comoving_code[0], np.finfo(float).tiny),
            radius_comoving_code[-1],
            int(par["output"]["density_plot_bins"]) + 1,
        )
        bin_index = np.clip(
            np.digitize(radius_comoving_code, bin_edges) - 1, 0, bin_edges.size - 2
        )
        binned_mass = np.bincount(
            bin_index, weights=mass, minlength=bin_edges.size - 1
        )
        binned_volume = np.bincount(
            bin_index, weights=shell_volume, minlength=bin_edges.size - 1
        )
        valid = binned_volume > 0.0
        bin_radius = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        bin_density_contrast = np.full(bin_radius.shape, np.nan)
        bin_density_contrast[valid] = (
            binned_mass[valid] / binned_volume[valid] / mean_density[row]
        )
        density_plot_radius.append(bin_radius[valid])
        density_plot_contrast.append(bin_density_contrast[valid])

    data_file = output_dir / "DarkMatterOnlyCorrelationControl.npz"
    np.savez(data_file, time_cosmic_code=times, radius_comoving_code=radii,
             density_contrast=density_contrast, enclosed_mass=enclosed_mass)
    figure = output_dir / "DarkMatterOnlyCorrelationControl.jpg"
    selected = np.unique(np.linspace(0, len(times) - 1, min(9, len(times))).astype(int))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, selected.size))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for color, row in zip(colors, selected):
        order = np.argsort(radii[row])
        label = "t = %.2f" % times[row]
        axes[0].loglog(
            density_plot_radius[row], density_plot_contrast[row],
            color=color, label=label,
        )
        axes[1].loglog(radii[row][order], enclosed_mass[row][order], color=color, label=label)
    axes[0].axhline(1.0, color="black", ls="--", lw=0.8)
    axes[0].set(xlabel="proper radius [kpc]", ylabel="DM density / background",
                title="Gas-free DM density contrast")
    axes[1].set(xlabel="proper radius [kpc]", ylabel="enclosed DM mass [code mass]",
                title="Gas-free DM enclosed mass")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(title="cosmic time [Gyr]", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure, dpi=220)
    plt.close(fig)
    print("initial DM mass = %.8g code masses" % initial_mass)
    print("final DM mass = %.8g code masses" % shells.total_mass)
    print("mass error = %.8g" % (shells.total_mass - initial_mass))
    print("shell crossings = %d" % shells.total_crossing_event_count)
    print("origin reflections = %d" % shells.total_origin_reflection_count)
    print("data = %s" % data_file)
    print("figure = %s" % figure)


if __name__ == "__main__":
    main()
