"""Plot and verify a generated z=100 correlation-function IC file."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

from radhydropy.cosmology import EinsteinDeSitter
import radhydropy.io as rio
from radhydropy.rsim import Rsim
from example_utils import load_nested_example_config
from radhydropy.units import CodeUnits, quantity_to_value
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name(
    "cosmological_dark_matter_correlation_z100.yaml"
)


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    config = load_nested_example_config(config_filename)
    par = config["par"]
    initial_condition = config["initial_condition"]
    example = config["example"]
    cosmology_config = par["cosmology"]
    units = CodeUnits.from_mapping(par["units"]["CodeUnits"])
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=quantity_to_value(cosmology_config["cosmology_t_ref"], units.time_unit),
        a_ref=float(cosmology_config["cosmology_a_ref"]),
    )
    table_filename = Path(example["linear_correlation_table_filename"])
    if not table_filename.is_absolute():
        table_filename = config_filename.parent / table_filename
    table = et.load_lcdm_correlation_table(table_filename)
    config["_code_unit_system"] = units
    config["_cosmology"] = cosmology
    config["_correlation_table"] = table

    filename = Path(par["simulation"]["initial_condition_filename"])
    if not filename.is_absolute():
        filename = config_filename.parent / filename
    snapshot = Rsim(config["par"])
    rio.readhdf5(snapshot.par, snapshot.mesh, snapshot.fluid, str(filename))
    first = int(snapshot.par.mesh.ghost_cells)
    last = first + int(snapshot.par.mesh.grid_cells)
    boundary_comoving_code = np.asarray(
        snapshot.mesh.boundary_comoving_code[first:last + 1], dtype=float
    )
    rho_comoving_code = np.asarray(
        snapshot.fluid.rho_comoving_code[first:last], dtype=float
    )
    temp_supercomoving_code = np.asarray(
        snapshot.fluid.temp_supercomoving_code[first:last], dtype=float
    )
    vel_supercomoving_code = np.asarray(
        snapshot.fluid.vel_supercomoving_code[first:last], dtype=float
    )

    radius_comoving_code = et.cell_centres(boundary_comoving_code)
    initial_time = quantity_to_value(initial_condition["time_cosmic"], units.time_unit)
    scale_factor = float(cosmology.scale_factor(initial_time))
    redshift = 1.0 / scale_factor - 1.0
    length_unit_mpc_h = (
        float(units.length_in_cgs)
        / float((1.0 * unyt.Mpc).to_value("cm"))
        * float(initial_condition.get("correlation_h", 0.674))
    )
    expected_delta, expected_mean_delta = et.density_contrast_profile(
        radius_comoving_code, config,
        length_unit_mpc_h=length_unit_mpc_h,
    )
    rho_background = float(cosmology.background_density(initial_time))
    fb = float(initial_condition["baryon_fraction"])
    actual_delta = rho_comoving_code / (rho_background * scale_factor**3 * fb) - 1.0
    expected_velocity = (
        -scale_factor**2 * float(cosmology.hubble(initial_time))
        * expected_mean_delta * radius_comoving_code / 3.0
    )
    if bool(initial_condition.get("cmb_equilibrium_initial", False)):
        expected_temperature = float(
            initial_condition.get("cmb_temperature_0", 2.7255 * unyt.K).to_value(unyt.K)
        ) / scale_factor
    else:
        expected_temperature = float(
            initial_condition.get("cie_temperature_proper", 10.0 * unyt.K).to_value(unyt.K)
        )

    radius_perturbation_comoving_code = et.radius_perturbation_comoving_code(config)
    clipped_edges = np.clip(boundary_comoving_code, 0.0, radius_perturbation_comoving_code)
    shell_volume = 4.0 * np.pi / 3.0 * np.diff(clipped_edges**3)
    target_volume = 4.0 * np.pi / 3.0 * radius_perturbation_comoving_code**3
    target_mean_delta = np.sum(
        (rho_comoving_code - rho_background * scale_factor**3 * fb) * shell_volume
    ) / (rho_background * scale_factor**3 * fb * target_volume)

    rho_proper_cgs_g_cm3 = rho_comoving_code * float(units.density_unit) / scale_factor**3
    temperature_proper_cgs_K = (
        temp_supercomoving_code * float(units.temperature_unit) / scale_factor**2
    )
    hubble_vel_proper_cgs_cm_s = float(cosmology.hubble(initial_time)) * scale_factor * radius_comoving_code
    vel_peculiar_proper_cgs_cm_s = vel_supercomoving_code / scale_factor
    vel_radial_proper_cgs_cm_s = hubble_vel_proper_cgs_cm_s + vel_peculiar_proper_cgs_cm_s
    velocity_to_km_s = float(units.velocity_in_cgs) / 1.0e5
    proper_radius_kpc = (
        scale_factor * radius_comoving_code * float(units.length_in_cgs)
        / float((1.0 * unyt.kpc).to_value("cm"))
    )

    density_error = np.max(np.abs(actual_delta - expected_delta))
    velocity_error = np.max(np.abs(vel_supercomoving_code - expected_velocity))
    temperature_error = np.max(np.abs(temperature_proper_cgs_K - expected_temperature))
    if density_error > 1.0e-10 or velocity_error > 1.0e-10:
        raise RuntimeError("stored density or velocity does not match the IC construction")
    if temperature_error > 1.0e-10:
        raise RuntimeError("stored temperature does not match the requested cold IC")
    if abs(target_mean_delta - float(initial_condition["initial_overdensity"])) > 2.0e-4:
        raise RuntimeError("stored target overdensity is inconsistent with the requested normalization")

    output = filename.with_name("CosmologicalCorrelationInitialCondition.jpg")
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    axes[0, 0].semilogx(radius_comoving_code, actual_delta, label="stored IC")
    axes[0, 0].semilogx(radius_comoving_code, expected_delta, "--", label="correlation table")
    axes[0, 0].set_ylabel(r"$\delta(r)$")
    axes[0, 0].set_xlabel("comoving radius [code length]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].loglog(proper_radius_kpc, rho_proper_cgs_g_cm3)
    axes[0, 1].set_ylabel(r"gas density [g cm$^{-3}$]")
    axes[0, 1].set_xlabel("proper radius [kpc]")
    axes[1, 0].loglog(proper_radius_kpc, temperature_proper_cgs_K)
    axes[1, 0].axhline(expected_temperature, color="black", ls="--", lw=1.0)
    axes[1, 0].set_ylabel("gas temperature [K]")
    axes[1, 0].set_xlabel("proper radius [kpc]")
    axes[1, 1].semilogx(
        proper_radius_kpc, hubble_vel_proper_cgs_cm_s * velocity_to_km_s,
        label="Hubble flow",
    )
    axes[1, 1].semilogx(
        proper_radius_kpc, vel_peculiar_proper_cgs_cm_s / scale_factor * velocity_to_km_s,
        label="peculiar",
    )
    axes[1, 1].semilogx(
        proper_radius_kpc, vel_radial_proper_cgs_cm_s * velocity_to_km_s,
        label="total physical",
    )
    axes[1, 1].set_ylabel("radial velocity [km/s]")
    axes[1, 1].set_xlabel("proper radius [kpc]")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25, which="both")
    fig.suptitle("Stored z=100 correlation-function initial condition")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print("figure = %s" % output)
    print("scale factor = %.12g, redshift = %.8g" % (scale_factor, redshift))
    print("target radius = %.8g code lengths" % radius_perturbation_comoving_code)
    print("target enclosed overdensity = %.12g (requested %.12g)" % (
        target_mean_delta, float(initial_condition["initial_overdensity"])
    ))
    print("max density-profile error = %.6e" % density_error)
    print("max peculiar-velocity error = %.6e code velocity" % velocity_error)
    print("temperature range = [%.8g, %.8g] K" % (
        temperature_physical.min(), temperature_physical.max()
    ))
    print("verification = PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
