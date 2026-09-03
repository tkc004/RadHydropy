"""Plot and verify a generated z=100 correlation-function IC file."""

import argparse
from pathlib import Path
import sys

import h5py
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
from example_utils import load_nested_example_parameters
from radhydropy.units import CodeUnits
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name(
    "cosmological_dark_matter_correlation_z100.yaml"
)


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    runparams, icparams = load_nested_example_parameters(config_filename)
    units = CodeUnits.from_mapping(runparams["CodeUnits"])
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(runparams["cosmology_t_ref"]),
        a_ref=float(runparams["cosmology_a_ref"]),
    )
    table_filename = Path(runparams["linear_correlation_table_filename"])
    if not table_filename.is_absolute():
        table_filename = config_filename.parent / table_filename
    table = et.load_lcdm_correlation_table(table_filename)

    filename = Path(runparams["ICfilename"])
    with h5py.File(filename, "r") as handle:
        boundary = handle["Data/Boundary"][:] / float(units.length_in_cgs)
        density = handle["Data/Density"][:] / float(units.density_unit)
        temperature = handle["Data/Temperature"][:] / float(units.temperature_unit)
        velocity = handle["Data/Velocity"][:] / float(units.velocity_unit)

    radius = et.cell_centres(boundary)
    initial_time = float(icparams["initial_cosmic_time"])
    scale_factor = float(cosmology.scale_factor(initial_time))
    redshift = 1.0 / scale_factor - 1.0
    length_unit_mpc_h = (
        float(units.length_in_cgs)
        / float((1.0 * unyt.Mpc).to_value("cm"))
        * float(icparams.get("correlation_h", 0.674))
    )
    expected_delta, expected_mean_delta = et.density_contrast_profile(
        radius, icparams, cosmology,
        correlation_table=table,
        length_unit_mpc_h=length_unit_mpc_h,
    )
    rho_background = float(cosmology.background_density(initial_time))
    fb = float(icparams["baryon_fraction"])
    actual_delta = density / (rho_background * scale_factor**3 * fb) - 1.0
    expected_velocity = (
        -scale_factor**2 * float(cosmology.hubble(initial_time))
        * expected_mean_delta * radius / 3.0
    )
    expected_temperature = float(icparams.get("cie_initial_temperature", 10.0))

    target_radius = et.perturbation_radius(icparams, cosmology)
    clipped_edges = np.clip(boundary, 0.0, target_radius)
    shell_volume = 4.0 * np.pi / 3.0 * np.diff(clipped_edges**3)
    target_volume = 4.0 * np.pi / 3.0 * target_radius**3
    target_mean_delta = np.sum(
        (density - rho_background * scale_factor**3 * fb) * shell_volume
    ) / (rho_background * scale_factor**3 * fb * target_volume)

    density_physical = density * float(units.density_unit) / scale_factor**3
    temperature_physical = (
        temperature * float(units.temperature_unit) / scale_factor**2
    )
    hubble_velocity = float(cosmology.hubble(initial_time)) * scale_factor * radius
    total_velocity = hubble_velocity + velocity / scale_factor
    velocity_to_km_s = float(units.velocity_in_cgs) / 1.0e5
    proper_radius_kpc = (
        scale_factor * radius * float(units.length_in_cgs)
        / float((1.0 * unyt.kpc).to_value("cm"))
    )

    density_error = np.max(np.abs(actual_delta - expected_delta))
    velocity_error = np.max(np.abs(velocity - expected_velocity))
    temperature_error = np.max(np.abs(temperature_physical - expected_temperature))
    if density_error > 1.0e-10 or velocity_error > 1.0e-10:
        raise RuntimeError("stored density or velocity does not match the IC construction")
    if temperature_error > 1.0e-10:
        raise RuntimeError("stored temperature does not match the requested cold IC")
    if abs(target_mean_delta - float(icparams["initial_overdensity"])) > 2.0e-4:
        raise RuntimeError("stored target overdensity is inconsistent with the requested normalization")

    output = filename.with_name("CosmologicalCorrelationInitialCondition.jpg")
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    axes[0, 0].semilogx(radius, actual_delta, label="stored IC")
    axes[0, 0].semilogx(radius, expected_delta, "--", label="correlation table")
    axes[0, 0].set_ylabel(r"$\delta(r)$")
    axes[0, 0].set_xlabel("comoving radius [code length]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].loglog(proper_radius_kpc, density_physical)
    axes[0, 1].set_ylabel(r"gas density [g cm$^{-3}$]")
    axes[0, 1].set_xlabel("proper radius [kpc]")
    axes[1, 0].loglog(proper_radius_kpc, temperature_physical)
    axes[1, 0].axhline(expected_temperature, color="black", ls="--", lw=1.0)
    axes[1, 0].set_ylabel("gas temperature [K]")
    axes[1, 0].set_xlabel("proper radius [kpc]")
    axes[1, 1].semilogx(
        proper_radius_kpc, hubble_velocity * velocity_to_km_s,
        label="Hubble flow",
    )
    axes[1, 1].semilogx(
        proper_radius_kpc, velocity / scale_factor * velocity_to_km_s,
        label="peculiar",
    )
    axes[1, 1].semilogx(
        proper_radius_kpc, total_velocity * velocity_to_km_s,
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
    print("target radius = %.8g code lengths" % target_radius)
    print("target enclosed overdensity = %.12g (requested %.12g)" % (
        target_mean_delta, float(icparams["initial_overdensity"])
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
