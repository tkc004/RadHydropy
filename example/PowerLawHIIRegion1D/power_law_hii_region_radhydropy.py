"""RadHydropy power-law H II region versus the analytic solution."""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt

EXAMPLE_DIR = Path(__file__).resolve().parent
EXAMPLE_ROOT = EXAMPLE_DIR.parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
for path in (PROJECT_ROOT, EXAMPLE_ROOT, EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits, code_quantity_to_cgs
import example_utils as eu

import power_law_hii_region_analytic as analytic


DEFAULT_CONFIG = EXAMPLE_DIR / "power_law_hii_region_radhydropy.yaml"


def density_profile(radius_cgs_cm, nc, rc, w):
    radius_cgs_cm = np.asarray(radius_cgs_cm, dtype=float)
    return nc * np.where(radius_cgs_cm < rc, 1.0, (radius_cgs_cm / rc) ** (-w))


def build_initial_condition(config):
    """Build the physical IC object consumed by ``writehdf5``."""
    from radhydropy.eos import EOS
    from radhydropy.fluid import Fluid
    from radhydropy.mesh import Mesh

    par_config = config['par']
    initial = config['initial_condition']
    code = CodeUnits.from_mapping(par_config['units']['CodeUnits'])
    ncell = int(initial['number_of_cells'])
    boxsize = initial['boxsize']
    boundary = np.linspace(0.0, boxsize.to_value(unyt.cm), ncell + 1) * unyt.cm
    radius = 0.5 * (boundary[1:] + boundary[:-1])
    n_h = density_profile(
        radius.to_value(unyt.cm),
        initial['core_number_density'].to_value(1.0 / unyt.cm**3),
        initial['core_radius'].to_value(unyt.cm),
        initial['density_power_law_exponent'],
    ) / unyt.cm**3

    par = SimpleNamespace(
        coordsys=par_config['simulation']['coordinate_system'],
        boundcond=par_config['boundary']['condition'],
        nogrid=ncell,
        noghost=int(par_config['mesh']['ghost_cells']),
        boxsize=boxsize,
        area=par_config['mesh']['area'],
        EOStype=par_config['hydrodynamics']['eos_type'],
        gamma=par_config['hydrodynamics']['gamma'],
        CodeUnits=code,
        simulation=SimpleNamespace(
            coordinate_system="spherical",
            current_time=0.0 * unyt.yr,
            box_size=boxsize,
        ),
        mesh=SimpleNamespace(
            grid_cells=ncell,
            ghost_cells=int(par_config['mesh']['ghost_cells']),
            area=par_config['mesh']['area'],
        ),
        hydrodynamics=SimpleNamespace(gamma=par_config['hydrodynamics']['gamma']),
        units=SimpleNamespace(CodeUnits=code),
    )
    mesh = Mesh()
    mesh.boundary = boundary
    fluid = Fluid()
    fluid.eos = EOS(par_config['hydrodynamics']['eos_type'], par_config['hydrodynamics']['gamma'], code)
    fluid.rho_code = (n_h * unyt.mp).to(unyt.g / unyt.cm**3)
    fluid.vel_code = np.zeros(ncell) * unyt.cm / unyt.s
    fluid.temp_code = np.ones(ncell) * initial['initial_temperature']
    fluid.xHI = np.ones(ncell)
    fluid.mu = np.ones(ncell)
    fluid.ngamma_code = np.zeros(ncell) / unyt.cm**3
    fluid.SetFluidTime(0.0 * unyt.yr)
    return SimpleNamespace(par=par, mesh=mesh, fluid=fluid)


def write_initial_condition(config, filename):
    filename = Path(filename)
    filename.unlink(missing_ok=True)
    rio.writehdf5(build_initial_condition(config), filename)


def load_snapshot(filename, par_config):
    from radhydropy.fluid import Fluid
    from radhydropy.mesh import Mesh
    from radhydropy.params import Par

    par = Par(par_config)
    mesh = Mesh()
    fluid = Fluid()
    rio.readhdf5(par, mesh, fluid, filename)
    ghost_cells = par.mesh.ghost_cells
    grid_cells = par.mesh.grid_cells
    code_units = par.units.CodeUnits
    if ghost_cells > 0:
        mesh.boundary = mesh.boundary[ghost_cells:-ghost_cells]
    mesh.SetUpMesh(par)
    return par, mesh, fluid


def front_radius_cgs_cm(mesh, fluid, par, neutral_fraction=0.5):
    first = par.mesh.ghost_cells
    interior = slice(first, first + par.mesh.grid_cells)
    radius = np.asarray(
        code_quantity_to_cgs(mesh.coordinate[interior], par.units.CodeUnits, "length_cgs_cm"),
        dtype=float,
    )
    xhi = np.asarray(fluid.xHI[interior], dtype=float)
    crossings = np.where((xhi[:-1] <= neutral_fraction) & (xhi[1:] > neutral_fraction))[0]
    if crossings.size == 0:
        return np.nan
    left = int(crossings[0])
    right = left + 1
    weight = (neutral_fraction - xhi[left]) / (xhi[right] - xhi[left])
    return radius[left] + weight * (radius[right] - radius[left])


def shock_radius_cgs_cm(
    mesh,
    fluid,
    par,
    core_number_density=1.0e6,
    core_radius_cgs_cm=2.1e16,
    density_power_law_exponent=1.0,
):
    """Estimate the leading shock from the compressed neutral shell.

    The shock diagnostic is the outer edge of the first neutral shell whose
    density exceeds the original power-law density by five percent.  The
    shell is searched outside the ionization front; the unshocked profile is
    therefore not mistaken for a shock.  This is a profile diagnostic, not a
    replacement for a Riemann shock detector.
    """
    first = par.mesh.ghost_cells
    interior = slice(first, first + par.mesh.grid_cells)
    radius_cgs_cm = np.asarray(
        code_quantity_to_cgs(mesh.coordinate[interior], par.units.CodeUnits, "length_cgs_cm"),
        dtype=float,
    )
    rho_cgs = np.asarray(
        code_quantity_to_cgs(fluid.rho_code[interior], par.units.CodeUnits, "density_cgs_g_cm3"),
        dtype=float,
    )
    xhi = np.asarray(fluid.xHI[interior], dtype=float)
    front = front_radius_cgs_cm(mesh, fluid, par)
    if not np.isfinite(front):
        return np.nan

    initial_nh = density_profile(
        radius_cgs_cm,
        core_number_density,
        core_radius_cgs_cm,
        density_power_law_exponent,
    )
    compression = rho_cgs / (initial_nh * (1.0 * unyt.mp).to_value(unyt.g))
    neutral = (radius_cgs_cm > front) & (xhi > 0.5)
    candidates = np.where(neutral & (compression > 1.05))[0]
    if candidates.size == 0:
        return np.nan

    peak = candidates[np.argmax(compression[candidates])]
    shell = np.where(
        neutral & (np.arange(radius_cgs_cm.size) >= peak) & (compression > 1.05)
    )[0]
    if shell.size == 0:
        return np.nan
    return float(radius_cgs_cm[shell[-1]])


def output_files(outdir, prefix):
    return sorted(Path(outdir).glob(f"{prefix}_*.hdf5"))


def apply_piecewise_isothermal_state(sim, config):
    """Keep neutral gas cold and ionized gas at the configured 1e4 K."""
    sim.fluid.eos.apply_piecewise_isothermal_state(
        sim.fluid,
        sim.par,
        config['initial_condition']["neutral_temperature"],
        config['initial_condition']["ionized_temperature"],
        config['initial_condition'].get("isothermal_ionized_fraction_threshold"),
    )
    sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=getattr(sim.par, "verbose", 0))


def make_isothermal_step_backend(sim, config):
    def step_backend(dt=None, mode="hydro_sources", advect_chemistry=True):
        result = sim.Step(
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
        )
        apply_piecewise_isothermal_state(sim, config)
        return result

    return step_backend


def save_profile_plot(snapshots, output, exponent):
    """Plot density and velocity profiles using one style per output time."""
    line_styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharex=True)
    density_axis, velocity_axis = axes

    for index, (time_yr, par, mesh, fluid) in enumerate(snapshots):
        first = par.mesh.ghost_cells
        interior = slice(first, first + par.mesh.grid_cells)
        radius_cgs_cm = np.asarray(
            code_quantity_to_cgs(
                mesh.coordinate[interior], par.units.CodeUnits, "length_cgs_cm"
            ),
            dtype=float,
        )
        radius_pc = radius_cgs_cm / (1.0 * unyt.pc).to_value(unyt.cm)
        rho_cgs = np.asarray(
            code_quantity_to_cgs(
                fluid.rho_code[interior], par.units.CodeUnits, "density_cgs_g_cm3"
            ),
            dtype=float,
        )
        velocity_cgs_cm_s = np.asarray(
            code_quantity_to_cgs(
                fluid.vel_code[interior], par.units.CodeUnits, "velocity_cgs_cm_s"
            ),
            dtype=float,
        )
        style = line_styles[index % len(line_styles)]
        color = colors[index % len(colors)]
        label = f"{time_yr:.0f} yr"
        density_axis.plot(
            radius_pc,
            rho_cgs / (1.0 * unyt.mp).to_value(unyt.g),
            linestyle=style,
            color=color,
            label=label,
        )
        velocity_axis.plot(
            radius_pc,
            velocity_cgs_cm_s / 1.0e5,
            linestyle=style,
            color=color,
            label=label,
        )

    for axis in axes:
        axis.set_xlim(0.0, 1.0)
        axis.grid(True, which="both", alpha=0.25)
        axis.set_xlabel("radius (pc)")
    density_axis.set_yscale("log")
    density_axis.set_ylabel(r"hydrogen number density (cm$^{-3}$)")
    velocity_axis.set_ylabel("radial velocity (km/s)")
    velocity_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    density_axis.legend(title="snapshot", fontsize=8)
    figure.suptitle(f"Power-law H II region profiles, w={exponent:g}")
    figure.tight_layout()
    figure.savefig(output, dpi=180)


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    runtime = config['par']
    initial = config['initial_condition']
    example = config['example']
    output_config = runtime['output']
    outdir = Path(output_config['directory'])
    outdir.mkdir(parents=True, exist_ok=True)
    Path(output_config['savedir']).mkdir(parents=True, exist_ok=True)
    for filename in output_files(outdir, output_config['filename_prefix']):
        filename.unlink()
    write_initial_condition(config, runtime['simulation']['initial_condition_filename'])

    sim = Rsim(runtime)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    apply_piecewise_isothermal_state(sim, config)
    sim.Run(
        outputtime=0,
        mode="hydro_sources",
        step_backend=make_isothermal_step_backend(sim, config),
    )

    times_yr = []
    radii_cm = []
    shock_radii_cm = []
    snapshots = []
    nc = initial['core_number_density'].to_value(1.0 / unyt.cm**3)
    rc = initial['core_radius'].to_value(unyt.cm)
    exponent = float(initial['density_power_law_exponent'])
    for filename in output_files(outdir, output_config['filename_prefix']):
        par, mesh, fluid = load_snapshot(filename, runtime)
        time_s = code_quantity_to_cgs(fluid.time, par.units.CodeUnits, "time_s")
        time_yr = float(time_s) / (1.0 * unyt.yr).to_value(unyt.s)
        times_yr.append(time_yr)
        radii_cm.append(front_radius_cgs_cm(mesh, fluid, par))
        shock_radii_cm.append(
            shock_radius_cgs_cm(
                mesh,
                fluid,
                par,
                core_number_density=nc,
                core_radius_cgs_cm=rc,
                density_power_law_exponent=exponent,
            )
        )
        snapshots.append((time_yr, par, mesh, fluid))

    times_yr = np.asarray(times_yr)
    radii_cm = np.asarray(radii_cm)
    shock_radii_cm = np.asarray(shock_radii_cm)
    end_time_yr = float(runtime["simulation"]["final_time"].to_value(unyt.yr))
    source_rate_s = runtime['radiation']['radiative_transfer_source_photon_rate'].to_value(1.0 / unyt.s)
    analytic_time_s, analytic_radius_cgs_cm, _ = analytic.calculate_front(
        source_rate_s,
        nc,
        rc,
        exponent,
        end_time_yr,
    )
    analytic_time_yr = analytic_time_s / analytic.SECONDS_PER_YEAR
    analytic_valid = (
        np.isfinite(analytic_time_yr)
        & np.isfinite(analytic_radius_cgs_cm)
        & (analytic_time_yr > 0.0)
        & (analytic_radius_cgs_cm > 0.0)
    )
    valid = np.isfinite(radii_cm) & (times_yr > 0.0)
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    axis.plot(
        analytic_time_yr[analytic_valid],
        analytic_radius_cgs_cm[analytic_valid],
        color="black",
        linewidth=2.0,
        zorder=1,
        label=f"analytic, w={exponent:g}",
    )
    axis.plot(
        times_yr[valid],
        radii_cm[valid],
        "^",
        markerfacecolor="none",
        markeredgecolor="C1",
        zorder=3,
        label="ionization front",
    )
    shock_valid = np.isfinite(shock_radii_cm) & (times_yr > 0.0)
    axis.plot(
        times_yr[shock_valid],
        shock_radii_cm[shock_valid],
        "o",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor="C2",
        zorder=3,
        label="shock front",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1.0e2, end_time_yr)
    axis.set_ylim(1.0e16, 1.0e20)
    axis.set_xlabel("time (yr)")
    axis.set_ylabel("ionization-front radius (cm)")
    axis.set_title(f"RadHydropy vs analytic power-law H II region, w={exponent:g}")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = Path(output_config["savedir"]) / example["front_plot_filename"]
    figure.savefig(output, dpi=180)
    profile_output = Path(output_config["savedir"]) / example["profile_plot_filename"]
    save_profile_plot(snapshots, profile_output, exponent)
    print(f"wrote {output}")
    print(f"wrote {profile_output}")
    print(f"snapshots = {len(times_yr)}")
    if np.any(valid):
        print(f"final simulated front = {radii_cm[valid][-1]:.6e} cm")
    else:
        print("final simulated front = unavailable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
