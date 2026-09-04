"""Cheap pre-crossing gas/live-DM linear-growth consistency test.

The calculation evolves a low-amplitude copy of the LCDM correlation-function
perturbation with negligible gas pressure.  It compares enclosed gas and dark
matter overdensities and peculiar velocities with the Einstein--de Sitter
growing mode, while aborting before the first collisionless-shell crossing.
"""

import argparse
import copy
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "radhydropy-matplotlib")
)
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
from radhydropy.dark_matter import DarkMatterShells, prepare_enclosed_gas_mass
from example_utils import load_nested_example_config
from radhydropy.gravity import Gravity
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name(
    "cosmological_gas_dm_linear_growth.yaml"
)


class SmoothEnclosedMassForGas:
    """Delegate shell dynamics while smoothing gas force sampling in r^3."""

    def __init__(self, shells):
        self.shells = shells

    def __getattr__(self, name):
        return getattr(self.shells, name)

    def gravitating_enclosed_mass(self, radius=None,
                                  include_shell_mass_with_fixed=False):
        if radius is None:
            return self.shells.gravitating_enclosed_mass(
                radius,
                include_shell_mass_with_fixed=include_shell_mass_with_fixed,
            )
        shell_radius = np.asarray(self.shells.radius, dtype=float)
        shell_enclosed = np.asarray(
            self.shells.gravitating_enclosed_mass(
                shell_radius,
                include_shell_mass_with_fixed=include_shell_mass_with_fixed,
            ),
            dtype=float,
        )
        total = float(np.sum(self.shells.mass))
        if self.shells.fixed_enclosed_mass is not None:
            total += float(self.shells.fixed_enclosed_mass)
        outer_radius = shell_radius[-1] + 0.5 * (
            shell_radius[-1] - shell_radius[-2]
        )
        interpolation_radius = np.concatenate((
            [0.0], shell_radius, [outer_radius],
        ))
        interpolation_mass = np.concatenate((
            [0.0], shell_enclosed, [total],
        ))
        requested = np.asarray(radius, dtype=float)
        return np.interp(
            requested**3,
            interpolation_radius**3,
            interpolation_mass,
            left=0.0,
            right=total,
        )


class LinearGrowthDiagnosticSolver(Solver):
    """Record the smallest local face factor on every hydro step."""

    def __init__(self):
        super().__init__()
        self.positivity_factors = []

    def _positivity_limited_face_fluxes(
        self, fluid, dt, mesh, par, mass_face, mom_face, energy_face,
        geometric_mom=None, angular_face=None, **kwargs,
    ):
        factor = super()._positivity_limited_face_fluxes(
            fluid, dt, mesh, par, mass_face, mom_face, energy_face,
            geometric_mom=geometric_mom, angular_face=angular_face, **kwargs,
        )
        self.positivity_factors.append(float(factor))
        return factor


def _load_correlation_table(config_filename, par):
    filename = Path(par["linear_correlation_table_filename"])
    if not filename.is_absolute():
        filename = Path(config_filename).resolve().parent / filename
    return et.load_lcdm_correlation_table(filename)


def _set_background_state(sim, cosmology, cosmic_time, baryon_fraction,
                          initial_temperature_code, mu):
    """Synchronize the analytic EdS outer reservoir and its active cell."""
    scale_factor = float(cosmology.scale_factor(cosmic_time))
    background_comoving = (
        float(cosmology.background_density(cosmic_time)) * scale_factor**3
    )
    sim.par.rho_inflow = baryon_fraction * background_comoving
    sim.par.vel_inflow = 0.0
    sim.par.temp_inflow = initial_temperature_code
    sim.par.mu_inflow = mu

    first = int(sim.par.mesh.ghost_cells)
    index = first + int(sim.par.mesh.grid_cells) - 1
    rho = float(sim.par.rho_inflow)
    velocity = 0.0
    pressure = float(np.asarray(
        sim.fluid.eos.pressure(rho, initial_temperature_code, mu),
        dtype=float,
    ))
    volume = float(np.asarray(sim.mesh.vol[index], dtype=float))
    sim.fluid.rho_code[index] = rho
    sim.fluid.vel_code[index] = velocity
    sim.fluid.temp_code[index] = initial_temperature_code
    sim.fluid.mu[index] = mu
    sim.fluid.pre_code[index] = pressure
    sim.fluid.Mass_code[index] = rho * volume
    sim.fluid.Mom_code[index] = 0.0
    sim.fluid.Energy_code[index] = float(np.asarray(
        sim.fluid.eos.total_energy_density(rho, velocity, pressure),
        dtype=float,
    )) * volume


def _fit_amplitude(measured, reference, mask):
    """Return the least-squares amplitude of ``measured`` versus reference."""
    measured = np.asarray(measured, dtype=float)[mask]
    reference = np.asarray(reference, dtype=float)[mask]
    finite = np.isfinite(measured) & np.isfinite(reference)
    measured = measured[finite]
    reference = reference[finite]
    denominator = float(np.dot(reference, reference))
    if measured.size == 0 or denominator <= 0.0:
        return float("nan")
    return float(np.dot(measured, reference) / denominator)


def _matched_cell_density(boundaries, coordinates, target_enclosed_mass):
    """Choose piecewise-constant densities exact at every cell centre."""
    shell_volume = 4.0 * np.pi / 3.0 * np.diff(boundaries**3)
    partial_volume = 4.0 * np.pi / 3.0 * (
        coordinates**3 - boundaries[:-1]**3
    )
    density = np.empty_like(coordinates)
    mass_before = 0.0
    for index in range(coordinates.size):
        density[index] = (
            target_enclosed_mass[index] - mass_before
        ) / max(partial_volume[index], 1.0e-300)
        mass_before += density[index] * shell_volume[index]
    if np.any(density <= 0.0):
        raise RuntimeError("matched gas-density quadrature became non-positive")
    return density


def _matched_shell_mass(target_enclosed_mass):
    """Choose shell masses whose half-shell enclosed values are exact."""
    mass = np.empty_like(target_enclosed_mass)
    mass_before = 0.0
    for index, target in enumerate(target_enclosed_mass):
        mass[index] = 2.0 * (target - mass_before)
        mass_before += mass[index]
    if np.any(mass <= 0.0):
        raise RuntimeError("matched dark-matter quadrature became non-positive")
    return mass


def _make_matched_initial_state(config, units, cosmology,
                                correlation_table):
    """Build gas cells and one volume-centred DM shell per identical cell."""
    icparams = config["initial_condition"]
    par = config["par"]
    initial = et.Simwrap(
        config, units, cosmology, correlation_table=correlation_table
    )
    # A uniform origin-centred mesh avoids allowing logarithmic innermost-cell
    # truncation error to dominate a deliberately tiny growing-mode signal.
    boundaries = np.linspace(
        0.0, float(icparams["rmax"]), int(par["mesh"]["grid_cells"]) + 1
    )
    coordinates = et.cell_centres(boundaries)
    initial.mesh.boundary = boundaries
    initial.mesh.coordinate = coordinates
    initial.mesh.area = 4.0 * np.pi * boundaries[:-1]**2
    initial.mesh.vol = 4.0 * np.pi / 3.0 * np.diff(boundaries**3)
    cosmic_time = float(icparams["initial_cosmic_time"])
    scale_factor = float(cosmology.scale_factor(cosmic_time))
    hubble = float(cosmology.hubble(cosmic_time))
    background_comoving = (
        float(cosmology.background_density(cosmic_time)) * scale_factor**3
    )
    baryon_fraction = float(icparams["baryon_fraction"])
    length_unit_mpc_h = (
        float(units.length_in_cgs) / 3.0856775814913673e24
        * float(icparams.get("correlation_h", 0.674))
    )
    _, mean_delta = et.density_contrast_profile(
        coordinates, icparams, cosmology,
        correlation_table=correlation_table,
        length_unit_mpc_h=length_unit_mpc_h,
    )
    enclosed_volume = 4.0 * np.pi / 3.0 * coordinates**3
    target_total_mass = (
        background_comoving * enclosed_volume * (1.0 + mean_delta)
    )
    initial.fluid.rho_code = _matched_cell_density(
        boundaries, coordinates, baryon_fraction * target_total_mass
    )
    initial.fluid.vel_code = -(
        scale_factor**2 * hubble * mean_delta * coordinates / 3.0
    )
    initial.fluid.temp_code = np.full(
        int(par["mesh"]["grid_cells"]),
        float(icparams["cie_initial_temperature"]) * scale_factor**2,
    )

    shell_radius = coordinates.copy()
    shells = DarkMatterShells(
        radius=shell_radius,
        velocity=-scale_factor**2 * hubble * mean_delta * shell_radius / 3.0,
        mass=_matched_shell_mass(
            (1.0 - baryon_fraction) * target_total_mass
        ),
        angular_momentum=np.zeros_like(shell_radius),
        softening=float(par.get("dark_matter", {}).get("softening", 0.0)),
        code_units=units,
    )
    return initial, shells


def _snapshot(sim, dm, cosmic_time, cosmology, icparams, correlation_table,
              initial_scale_factor, diagnostic_min, diagnostic_max):
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    x = np.asarray(sim.mesh.coordinate[first:last], dtype=float)
    rho_code = np.asarray(sim.fluid.rho_code[first:last], dtype=float)
    scale_factor = float(cosmology.scale_factor(cosmic_time))
    hubble = float(cosmology.hubble(cosmic_time))
    growth = scale_factor / initial_scale_factor
    background_comoving = (
        float(cosmology.background_density(cosmic_time)) * scale_factor**3
    )
    baryon_fraction = float(icparams["baryon_fraction"])
    dm_fraction = 1.0 - baryon_fraction
    volume = 4.0 * np.pi / 3.0 * x**3

    gas_mass = prepare_enclosed_gas_mass(
        sim.mesh, sim.fluid.rho_code, sim.par
    )(x)
    dm_x = np.asarray(dm.radius, dtype=float)
    dm_mass = dm.enclosed_mass(dm_x)
    delta_gas = gas_mass / np.maximum(
        baryon_fraction * background_comoving * volume, 1.0e-300
    ) - 1.0
    delta_dm = dm_mass / np.maximum(
        dm_fraction * background_comoving * (4.0 * np.pi / 3.0) * dm_x**3,
        1.0e-300,
    ) - 1.0

    length_unit_mpc_h = (
        float(sim.par.CodeUnits.length_in_cgs)
        / 3.0856775814913673e24
        * float(icparams.get("correlation_h", 0.674))
    )
    _, mean_delta_initial = et.density_contrast_profile(
        x,
        icparams,
        cosmology,
        correlation_table=correlation_table,
        length_unit_mpc_h=length_unit_mpc_h,
    )
    delta_analytic = growth * mean_delta_initial

    gas_velocity = np.asarray(sim.fluid.vel_code[first:last], dtype=float) / scale_factor
    gas_velocity_analytic = -(
        scale_factor * hubble * x * delta_analytic / 3.0
    )

    _, dm_mean_delta_initial = et.density_contrast_profile(
        dm_x,
        icparams,
        cosmology,
        correlation_table=correlation_table,
        length_unit_mpc_h=length_unit_mpc_h,
    )
    dm_velocity = np.asarray(dm.velocity, dtype=float) / scale_factor
    dm_velocity_analytic = -(
        scale_factor * hubble * dm_x * growth * dm_mean_delta_initial / 3.0
    )

    density_signal = np.abs(delta_analytic) > 1.0e-7
    gas_valid = (
        (x >= diagnostic_min) & (x <= diagnostic_max) & density_signal
    )
    dm_density_analytic = growth * dm_mean_delta_initial
    dm_density_valid = (
        (dm_x >= diagnostic_min) & (dm_x <= diagnostic_max)
        & (np.abs(dm_density_analytic) > 1.0e-7)
    )
    dm_velocity_valid = (
        (dm_x >= diagnostic_min) & (dm_x <= diagnostic_max)
        & (np.abs(dm_velocity_analytic) > 1.0e-12)
    )
    velocity_valid = gas_valid & (np.abs(gas_velocity_analytic) > 1.0e-12)
    gas_density_amplitude = _fit_amplitude(
        delta_gas, delta_analytic, gas_valid
    )
    dm_density_amplitude = _fit_amplitude(
        delta_dm, dm_density_analytic, dm_density_valid
    )

    return {
        "time_Gyr": float(
            cosmic_time * sim.par.CodeUnits.time_unit.to_value("Gyr")
        ),
        "scale_factor": scale_factor,
        "growth_factor": growth,
        "radius_comoving_kpc": x,
        "delta_bar_gas": delta_gas,
        "delta_bar_dm": delta_dm,
        "delta_bar_analytic": delta_analytic,
        "gas_peculiar_velocity_km_s": gas_velocity,
        "gas_velocity_analytic_km_s": gas_velocity_analytic,
        "dm_radius_comoving_kpc": dm_x.copy(),
        "delta_bar_dm_analytic": dm_density_analytic,
        "dm_peculiar_velocity_km_s": dm_velocity,
        "dm_velocity_analytic_km_s": dm_velocity_analytic,
        "delta_gas_growth_amplitude": gas_density_amplitude,
        "delta_dm_growth_amplitude": dm_density_amplitude,
        "delta_gas_over_dm": gas_density_amplitude / dm_density_amplitude,
        "gas_velocity_growth_amplitude": _fit_amplitude(
            gas_velocity, gas_velocity_analytic, velocity_valid
        ),
        "dm_velocity_growth_amplitude": _fit_amplitude(
            dm_velocity, dm_velocity_analytic, dm_velocity_valid
        ),
        "minimum_shell_separation_comoving_kpc": float(
            np.min(np.diff(dm_x))
        ),
        "predicted_crossing_dt": float(dm.crossing_timestep(safety_factor=1.0)),
    }


def _save_outputs(history, output_dir, force_mode, positivity_factors,
                  diagnostic_min, diagnostic_max):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        key: np.asarray([snapshot[key] for snapshot in history])
        for key in history[0]
    }
    positivity_factors = np.asarray(positivity_factors, dtype=float)
    data["positivity_limiter_factor"] = positivity_factors
    data_filename = output_dir / "CosmologicalGasDMLinearGrowth.npz"
    np.savez(data_filename, **data)

    final = history[-1]
    radius = final["radius_comoving_kpc"]
    dm_radius = final["dm_radius_comoving_kpc"]
    gas_plot = (radius >= diagnostic_min) & (radius <= diagnostic_max)
    dm_plot = (dm_radius >= diagnostic_min) & (dm_radius <= diagnostic_max)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    axes[0, 0].plot(radius[gas_plot], final["delta_bar_analytic"][gas_plot], "k-", label="analytic")
    axes[0, 0].plot(radius[gas_plot], final["delta_bar_gas"][gas_plot], "C0--", label="gas")
    axes[0, 0].plot(dm_radius[dm_plot], final["delta_bar_dm_analytic"][dm_plot], "k:", label="analytic at DM shells")
    axes[0, 0].plot(dm_radius[dm_plot], final["delta_bar_dm"][dm_plot], "C1:", label="dark matter")
    axes[0, 0].set_ylabel(r"enclosed $\bar{\delta}$")
    axes[0, 0].set_title("Final enclosed overdensity")

    axes[0, 1].plot(
        radius[gas_plot], final["gas_velocity_analytic_km_s"][gas_plot], "k-", label="analytic gas grid"
    )
    axes[0, 1].plot(
        radius[gas_plot], final["gas_peculiar_velocity_km_s"][gas_plot], "C0--", label="gas"
    )
    axes[0, 1].plot(
        dm_radius[dm_plot], final["dm_peculiar_velocity_km_s"][dm_plot], "C1:", label="dark matter"
    )
    axes[0, 1].set_ylabel(r"peculiar velocity [km s$^{-1}$]")
    axes[0, 1].set_title("Final peculiar velocity")

    growth = data["growth_factor"]
    axes[1, 0].axhline(1.0, color="k", lw=1.0)
    axes[1, 0].plot(growth, data["delta_gas_growth_amplitude"], "o-", label="gas / analytic")
    axes[1, 0].plot(growth, data["delta_dm_growth_amplitude"], "s-", label="DM / analytic")
    axes[1, 0].plot(growth, data["delta_gas_over_dm"], "^-", label="gas / DM")
    axes[1, 0].set_ylabel("fitted overdensity amplitude")
    axes[1, 0].set_xlabel(r"linear growth $a/a_i$")

    axes[1, 1].axhline(1.0, color="k", lw=1.0)
    axes[1, 1].plot(growth, data["gas_velocity_growth_amplitude"], "o-", label="gas / analytic")
    axes[1, 1].plot(growth, data["dm_velocity_growth_amplitude"], "s-", label="DM / analytic")
    axes[1, 1].set_ylabel("fitted peculiar-velocity amplitude")
    axes[1, 1].set_xlabel(r"linear growth $a/a_i$")

    for axis in axes.flat:
        axis.set_xscale("log")
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8)
    axes[0, 0].set_xlabel("comoving radius [kpc]")
    axes[0, 1].set_xlabel("comoving radius [kpc]")
    figure.suptitle("Pre-crossing coupled gas/DM EdS linear growth")
    figure.tight_layout()
    figure_filename = output_dir / "CosmologicalGasDMLinearGrowth.jpg"
    figure.savefig(figure_filename, dpi=220)
    plt.close(figure)

    report_filename = output_dir / "CosmologicalGasDMLinearGrowth.txt"
    report_filename.write_text(
        "dm_force_sampling %s\n"
        "final_time_Gyr %.10g\n"
        "final_growth_factor %.10g\n"
        "delta_gas_growth_amplitude %.10g\n"
        "delta_dm_growth_amplitude %.10g\n"
        "delta_gas_over_dm %.10g\n"
        "gas_velocity_growth_amplitude %.10g\n"
        "dm_velocity_growth_amplitude %.10g\n"
        "minimum_shell_separation_comoving_kpc %.10g\n"
        "minimum_positivity_limiter_factor %.10g\n"
        "median_positivity_limiter_factor %.10g\n"
        "fraction_hydro_steps_limited %.10g\n"
        % (
            force_mode,
            final["time_Gyr"],
            final["growth_factor"],
            final["delta_gas_growth_amplitude"],
            final["delta_dm_growth_amplitude"],
            final["delta_gas_over_dm"],
            final["gas_velocity_growth_amplitude"],
            final["dm_velocity_growth_amplitude"],
            final["minimum_shell_separation_comoving_kpc"],
            float(np.min(positivity_factors)),
            float(np.median(positivity_factors)),
            float(np.mean(positivity_factors < 1.0 - 1.0e-12)),
        ),
        encoding="utf-8",
    )
    return data_filename, figure_filename, report_filename


def run(config_filename=DEFAULT_CONFIG, final_time_override=None,
        smooth_force_override=None, resolution_override=None,
        output_suffix=None):
    config_filename = Path(config_filename).resolve()
    config = load_nested_example_config(config_filename)
    par = config["par"]
    initial_condition = config["initial_condition"]
    example = config["example"]
    if resolution_override is not None:
        resolution = int(resolution_override)
        if resolution < 8 or resolution > 1024:
            raise ValueError("resolution must be between 8 and 1024")
        initial_condition = dict(initial_condition)
        par = copy.deepcopy(par)
        par["mesh"]["grid_cells"] = resolution
        initial_condition["dark_matter_shells"] = resolution
    if int(initial_condition["dark_matter_shells"]) > 1024 or int(par["mesh"]["grid_cells"]) > 1024:
        raise ValueError("linear-growth test is limited to at most 1024 gas cells and shells")
    if int(initial_condition["dark_matter_shells"]) != int(par["mesh"]["grid_cells"]):
        raise ValueError("linear-growth quadrature requires one DM shell per gas cell")

    units = CodeUnits.from_mapping(par["units"]["CodeUnits"])
    gravity = par["gravity"]
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(gravity["cosmology_t_ref"]),
        a_ref=float(gravity["cosmology_a_ref"]),
    )
    correlation_table = _load_correlation_table(config_filename, par)
    smooth_force = (
        bool(par["hydrodynamics"].get("smooth_dm_force_for_gas", True))
        if smooth_force_override is None else bool(smooth_force_override)
    )
    output_dir = Path(par["output"]["savedir"])
    if not output_dir.is_absolute():
        output_dir = config_filename.parent / output_dir
    if output_suffix:
        output_dir = output_dir.with_name(output_dir.name + str(output_suffix))
    if resolution_override is not None:
        output_dir = output_dir.with_name(
            "%s_%d" % (output_dir.name, int(resolution_override))
        )
    if not smooth_force:
        output_dir = output_dir.with_name(output_dir.name + "_raw_shell_force")
    output_dir.mkdir(parents=True, exist_ok=True)
    ic_filename = output_dir / "InitialCondition.hdf5"

    initial, dm = _make_matched_initial_state(
        {"par": par, "initial_condition": initial_condition},
        units, cosmology, correlation_table
    )
    rio.writehdf5(initial, ic_filename)
    initial_shell_mass_order = np.asarray(dm.mass, dtype=float).copy()
    if np.unique(initial_shell_mass_order).size != dm.number_of_shells:
        raise RuntimeError("shell masses must be unique for the crossing guard")

    local = copy.deepcopy(par)
    local["simulation"]["initial_condition_filename"] = str(ic_filename)
    local["output"].update({
        "directory": str(output_dir), "savedir": str(output_dir),
    })
    sim = Rsim(local)
    diagnostic_solver = LinearGrowthDiagnosticSolver()
    sim.solver = diagnostic_solver
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.fluid.time = float(np.asarray(sim.par.time).flat[0])
    gravity_dm = SmoothEnclosedMassForGas(dm) if smooth_force else dm
    sim.par.gravity = Gravity(
        selfgravity=True,
        cosmological=True,
        cosmology=sim.par.cosmology,
        dark_matter=gravity_dm,
        code_units=sim.par.CodeUnits,
    )
    sim.par.dark_matter = dm
    baryon_fraction = float(initial_condition["baryon_fraction"])
    sim.par.dark_matter_background_fraction = 1.0 - baryon_fraction
    sim.par.gas_background_fraction = baryon_fraction

    initial_time = float(initial_condition["initial_cosmic_time"])
    final_time = (
        float(final_time_override)
        if final_time_override is not None
        else float(par["simulation"]["final_time"])
    )
    if final_time <= initial_time:
        raise ValueError("final cosmic time must exceed the initial time")
    initial_scale_factor = float(cosmology.scale_factor(initial_time))
    initial_temperature_code = float(initial_condition["cie_initial_temperature"]) * initial_scale_factor**2
    diagnostic_min = float(example["diagnostic_radius_min_comoving_kpc"])
    diagnostic_max = float(example["diagnostic_radius_max_comoving_kpc"])
    snapshot_count = int(example.get("snapshot_count", 9))
    snapshot_times = np.geomspace(initial_time, final_time, snapshot_count)
    snapshot_taus = np.asarray(cosmology.supercomoving_time(snapshot_times), dtype=float)

    _set_background_state(
        sim, cosmology, initial_time, baryon_fraction,
            initial_temperature_code, float(initial_condition["mu"]),
    )
    sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
    sim.solver.SetConserved(sim.mesh, sim.fluid)

    history = [_snapshot(
        sim, dm, initial_time, cosmology, initial_condition, correlation_table,
        initial_scale_factor, diagnostic_min, diagnostic_max,
    )]
    steps = 0
    for target_tau in snapshot_taus[1:]:
        while float(sim.fluid.time) < target_tau - 1.0e-13:
            cosmic_time = float(
                cosmology.cosmic_time_from_supercomoving(float(sim.fluid.time))
            )
            _set_background_state(
                sim, cosmology, cosmic_time, baryon_fraction,
                initial_temperature_code, float(initial_condition["mu"]),
            )
            sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
            sim.solver.SetConserved(sim.mesh, sim.fluid)
            dt = min(float(sim.GetStepTime()), target_tau - float(sim.fluid.time))
            crossing_dt = float(dm.crossing_timestep(safety_factor=1.0))
            if np.isfinite(crossing_dt) and crossing_dt <= dt * (1.0 + 1.0e-12):
                raise RuntimeError(
                    "predicted dark-matter shell crossing before requested endpoint "
                    "at cosmic time %.8g" % cosmic_time
                )
            sim.Step(dt=dt, mode="hydro")
            steps += 1
            if not np.array_equal(dm.mass, initial_shell_mass_order):
                raise RuntimeError("dark-matter shell crossing occurred during linear test")
            if np.any(np.diff(dm.radius) <= 0.0):
                raise RuntimeError("dark-matter shell radii ceased to be strictly ordered")

        cosmic_time = float(
            cosmology.cosmic_time_from_supercomoving(float(sim.fluid.time))
        )
        _set_background_state(
            sim, cosmology, cosmic_time, baryon_fraction,
            initial_temperature_code, float(initial_condition["mu"]),
        )
        history.append(_snapshot(
            sim, dm, cosmic_time, cosmology, initial_condition, correlation_table,
            initial_scale_factor, diagnostic_min, diagnostic_max,
        ))

    force_mode = "volume_linear" if smooth_force else "raw_step"
    data, figure, report = _save_outputs(
        history, output_dir, force_mode, diagnostic_solver.positivity_factors,
        diagnostic_min, diagnostic_max,
    )
    final = history[-1]
    print("steps = %d" % steps)
    print("DM force sampling = %s" % force_mode)
    print(
        "DM crossing batch fraction = %.8g"
        % float(getattr(sim.par, "dark_matter_crossing_batch_fraction", 0.0))
    )
    print("final cosmic time = %.8g Gyr" % final["time_Gyr"])
    print("gas/analytic overdensity = %.8g" % final["delta_gas_growth_amplitude"])
    print("DM/analytic overdensity = %.8g" % final["delta_dm_growth_amplitude"])
    print("gas/DM overdensity = %.8g" % final["delta_gas_over_dm"])
    print("gas/analytic velocity = %.8g" % final["gas_velocity_growth_amplitude"])
    print("DM/analytic velocity = %.8g" % final["dm_velocity_growth_amplitude"])
    print(
        "positivity limiter: min=%.8g median=%.8g limited_fraction=%.8g"
        % (
            np.min(diagnostic_solver.positivity_factors),
            np.median(diagnostic_solver.positivity_factors),
            np.mean(np.asarray(diagnostic_solver.positivity_factors) < 1.0 - 1.0e-12),
        )
    )
    print("data = %s" % data)
    print("figure = %s" % figure)
    print("report = %s" % report)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the cheap pre-crossing coupled gas/DM growth test."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument(
        "--raw-shell-force", action="store_true",
        help="use the production stepwise shell enclosed mass instead of the smooth control",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="use this matched number of gas cells and DM shells; default is the YAML resolution",
    )
    parser.add_argument("--output-suffix", default=None)
    arguments = parser.parse_args()
    run(
        arguments.config,
        arguments.final_time,
        smooth_force_override=False if arguments.raw_shell_force else None,
        resolution_override=arguments.resolution,
        output_suffix=arguments.output_suffix,
    )
