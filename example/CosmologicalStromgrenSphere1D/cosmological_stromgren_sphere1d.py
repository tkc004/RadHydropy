# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Run the cosmological Strömgren-sphere front-location benchmark."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "radhydropy-matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "example"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

from cosmological_initial_condition import build_initial_condition

from example import example_utils as eu
from example.CosmologicalStromgrenSphere1D.cosmological_stromgren_analytic import (
    front_radius_comoving,
    stromgren_radius_proper,
)
from radhydropy import io as rio
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits

DEFAULT_CONFIG = Path(__file__).with_name("cosmological_stromgren_sphere1d.yaml")
IONIZED_FRONT_NEUTRAL_FRACTION = 0.5


def _front_and_resolution(sim):
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    boundary = np.asarray(sim.mesh.boundary_comoving_code[first : last + 1], dtype=float)
    radius = 0.5 * (boundary[:-1] + boundary[1:])
    xhi = np.asarray(sim.fluid.xHI[first:last], dtype=float)
    ionized = xhi <= IONIZED_FRONT_NEUTRAL_FRACTION
    if not np.any(ionized):
        front = 0.0
        index = 0
    elif np.all(ionized):
        front = boundary[-1]
        index = len(xhi) - 1
    else:
        index = int(np.where(ionized)[0][-1])
        right = index + 1
        weight = (IONIZED_FRONT_NEUTRAL_FRACTION - xhi[index]) / (xhi[right] - xhi[index])
        front = radius[index] + weight * (radius[right] - radius[index])
    smoothing = 0.5 * (boundary[index + 1] - boundary[index])
    tau = float(np.asarray(sim.fluid.tau_supercomoving_code).flat[0])
    _, scale_factor, _ = sim.par.cosmology.model.background_state_from_supercomoving(tau)
    return front, smoothing, float(scale_factor)


def _write_output_times(filename, cosmology, initial_scale_factor, final_scale_factor, units):
    scale_factors = np.linspace(initial_scale_factor, final_scale_factor, 25)
    times = cosmology.cosmic_time_from_scale_factor(scale_factors)
    taus = cosmology.supercomoving_time(times)
    values_myr = taus * units.time_unit.to_value(unyt.Myr)
    filename.write_text("Myr\n" + "\n".join(f"{value:.16e}" for value in values_myr) + "\n")
    return float(taus[-1])


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    par = config["par"]
    example = config["example"]
    units = CodeUnits.from_mapping(par["units"]["CodeUnits"])
    output_dir = Path(par["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    eu.clean_previous_outputs(config)

    initial_a = float(example["initial_scale_factor"])
    final_a = float(example["final_scale_factor"])
    h0 = example["hubble_constant"].to_value(1 / unyt.s)
    time_ref = 2.0 / (3.0 * h0) / units.time_unit.to_value(unyt.s)
    cosmology = EinsteinDeSitter.from_code_units(units, t_ref=time_ref, a_ref=1.0)
    initial_time = float(cosmology.cosmic_time_from_scale_factor(initial_a))
    initial_tau = float(cosmology.supercomoving_time(initial_time))
    final_tau = _write_output_times(
        output_dir / "output_times.txt",
        cosmology,
        initial_a,
        final_a,
        units,
    )

    par["cosmology"].update(
        cosmology_t_ref=time_ref * units.time_unit,
        cosmology_a_ref=1.0,
    )
    par["simulation"]["final_time"] = final_tau * units.time_unit
    config["initial_condition"]["time_cosmic"] = initial_time * units.time_unit
    par["simulation"]["initial_condition_filename"] = str(output_dir / "InitialCondition.hdf5")
    par["output"]["time_list_filename"] = str(output_dir / "output_times.txt")
    par["output"]["directory"] = str(output_dir)
    config["_code_cosmology"] = cosmology
    config["_initial_tau_supercomoving_code"] = initial_tau

    rho_m_code = cosmology.background_density(initial_time)
    rho_h_code = 0.1 * rho_m_code * initial_a**3
    grid_cells = int(par["mesh"]["grid_cells"])
    config["_rho_comoving_code"] = np.full(grid_cells, rho_h_code)
    gamma = float(par["hydrodynamics"]["gamma"])
    temperature = example["temperature"].to_value(unyt.K)
    config["_temp_supercomoving_code"] = np.full(
        grid_cells,
        temperature * initial_a ** (3.0 * (gamma - 1.0)),
    )
    config["_vel_supercomoving_code"] = np.zeros(grid_cells)

    writer = build_initial_condition(config)
    writer.set_field("xHI", np.ones(grid_cells))
    writer.set_field(
        "ngamma_comoving_code",
        np.zeros(grid_cells),
    )
    writer.write(par["simulation"]["initial_condition_filename"], validate=True)

    sim = Rsim(par)
    sim.RunAll(outputtime=0, mode="sources", advect_chemistry=False)

    radius_errors = []
    ratios = []
    redshifts = []
    source_rate = example["source_photon_rate"]
    alpha_b = example["alpha_B"]
    n_h_initial = (
        rho_h_code / initial_a**3 * units.density_unit / unyt.mp
    ).to(1.0 / unyt.cm**3)
    analytic_scale_factors = np.linspace(initial_a, final_a, 400)
    analytic_front_comoving = front_radius_comoving(
        analytic_scale_factors,
        initial_a,
        initial_time * units.time_unit,
        source_rate,
        n_h_initial,
        alpha_b,
    )
    analytic_front_proper = analytic_front_comoving * analytic_scale_factors
    analytic_stromgren = stromgren_radius_proper(
        analytic_scale_factors,
        initial_a,
        n_h_initial,
        source_rate,
        alpha_b,
    )
    analytic_ratio = (analytic_front_proper / analytic_stromgren).to_value("")
    analytic_redshifts = 1.0 / analytic_scale_factors - 1.0
    for filename in sorted(output_dir.glob("Output_*.hdf5")):
        snapshot = rio.loadhdf5(config, filename)
        front_comoving, smoothing_comoving, scale_factor = _front_and_resolution(snapshot)
        front_proper = front_comoving * scale_factor * units.length_unit
        smoothing_proper = smoothing_comoving * scale_factor * units.length_unit
        stromgren = (
            3.0
            * source_rate
            / (4.0 * np.pi * alpha_b * n_h_initial**2 * (initial_a / scale_factor) ** 6)
        ) ** (1.0 / 3.0)
        radius_errors.append(float((smoothing_proper / stromgren).to_value("")))
        ratios.append(float((front_proper / stromgren).to_value("")))
        redshifts.append(1.0 / scale_factor - 1.0)

    order = np.argsort(redshifts)
    figure = output_dir / "CosmologicalStromgrenSphere1D_IFrontRatio.jpg"
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(
        np.asarray(redshifts)[order],
        np.asarray(ratios)[order],
        yerr=np.asarray(radius_errors)[order],
        fmt="s",
        color="magenta",
        ecolor="magenta",
        capsize=2.5,
        label="simulation, $x_{\\rm HI}=0.5$",
    )
    ax.plot(
        analytic_redshifts,
        analytic_ratio,
        color="black",
        lw=1.8,
        label="analytic, recombination-only",
    )
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$r_I/r_S$")
    ax.set_xlim(0.0, 9.0)
    ax.grid(visible=True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
