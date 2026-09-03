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
from radhydropy.units import CodeUnits
import tools as et


CONFIG = Path(__file__).with_name("dark_matter_only_correlation_control.yaml")


def main(config_filename=CONFIG):
    config_filename = Path(config_filename).resolve()
    config = load_nested_example_config(config_filename)
    par = config["par"]
    ic = config["initial_condition"]
    units = CodeUnits.from_mapping(par["units"]["CodeUnits"])
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(par["gravity"]["cosmology_t_ref"]),
        a_ref=float(par["gravity"]["cosmology_a_ref"]),
    )
    table_path = config["example"]["correlation_table_filename"]
    table_path = (config_filename.parent / table_path).resolve()
    table = et.load_lcdm_correlation_table(table_path)
    shells = et.make_dark_matter(
        ic, units, cosmology, correlation_table=table,
        softening=float(par["dark_matter"]["softening"]),
    )

    initial = float(ic["initial_cosmic_time"])
    final = float(par["simulation"]["final_time"])
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
        background = lambda radius, rho=rho_comoving: (
            4.0 * np.pi / 3.0 * rho * np.asarray(radius, dtype=float)**3
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
        float(cosmology.background_density(time)) for time in times
    ])
    density_contrast = np.empty_like(radii)
    enclosed_mass = np.empty_like(radii)
    density_plot_radius = []
    density_plot_contrast = []
    for row, (radius, mass) in enumerate(zip(radii, masses)):
        order = np.argsort(radius)
        radius = radius[order]
        mass = mass[order]
        edges = np.empty(radius.size + 1)
        edges[1:-1] = np.sqrt(radius[:-1] * radius[1:])
        edges[0] = radius[0]**2 / edges[1]
        edges[-1] = radius[-1]**2 / edges[-2]
        shell_volume = 4.0 * np.pi / 3.0 * np.diff(edges**3)
        density = np.divide(
            mass, shell_volume,
            out=np.full_like(mass, np.inf), where=shell_volume > 0.0,
        )
        density_contrast[row] = density / mean_density[row]
        enclosed_mass[row] = np.cumsum(mass)

        # The shell-by-shell profile becomes visually dominated by sampling
        # noise after shell crossing. Aggregate only the diagnostic profile
        # into logarithmic radial bins; retain the full-resolution enclosed
        # mass data and panel.
        bin_edges = np.geomspace(
            max(radius[0], np.finfo(float).tiny),
            radius[-1],
            int(par["output"]["density_plot_bins"]) + 1,
        )
        bin_index = np.clip(
            np.digitize(radius, bin_edges) - 1, 0, bin_edges.size - 2
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
    np.savez(data_file, time=times, radius_kpc=radii,
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
