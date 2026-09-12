"""Generate the z=100 correlation-function cosmological initial condition."""

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

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter, LambdaCDM
from example_utils import load_nested_example_config
from radhydropy.units import CodeUnits, quantity_to_value
import virial_shock_tools as et


DEFAULT_CONFIG = Path(__file__).with_name(
    "cosmological_dark_matter_correlation_z100_lambda_cdm.yaml"
)


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    config = load_nested_example_config(config_filename)
    par = config["par"]
    initial_condition = config["initial_condition"]
    example = config["example"]
    cosmology_config = par["cosmology"]
    units = CodeUnits.from_mapping(par["units"]["CodeUnits"])
    if cosmology_config.get("cosmology_type") == "lambda_cdm":
        cosmology = LambdaCDM.from_code_units(
            units, t_ref=quantity_to_value(cosmology_config["cosmology_t_ref"], units.time_unit),
            a_ref=float(cosmology_config["cosmology_a_ref"]),
            omega_m=float(cosmology_config["cosmology_omega_m"]),
            omega_lambda=float(cosmology_config["cosmology_omega_lambda"]),
            hubble_ref=float(cosmology_config["cosmology_hubble_ref"]),
        )
    else:
        cosmology = EinsteinDeSitter.from_code_units(
            units, t_ref=quantity_to_value(cosmology_config["cosmology_t_ref"], units.time_unit),
            a_ref=float(cosmology_config["cosmology_a_ref"]),
        )

    table_filename = Path(example["linear_correlation_table_filename"])
    if not table_filename.is_absolute():
        table_filename = config_filename.parent / table_filename
    correlation_table = et.load_lcdm_correlation_table(table_filename)
    config["_code_unit_system"] = units
    config["_cosmology"] = cosmology
    config["_correlation_table"] = correlation_table

    initial_writer = et.build_initial_condition(config)
    initial = initial_writer.simulation
    output = Path(par["simulation"]["initial_condition_filename"])
    output.parent.mkdir(parents=True, exist_ok=True)
    initial_writer.write(output)

    length_unit_mpc_h = (
        float(units.length_in_cgs)
        / float((1.0 * unyt.Mpc).to_value("cm"))
        * float(initial_condition.get("correlation_h", 0.674))
    )
    radius_comoving_code = np.asarray(initial.mesh.x_comoving_code, dtype=float)
    delta, mean_delta = et.density_contrast_profile(
        radius_comoving_code,
        config,
        length_unit_mpc_h=length_unit_mpc_h,
    )
    initial_time = quantity_to_value(initial_condition["time_cosmic"], units.time_unit)
    scale_factor = float(cosmology.scale_factor(initial_time))
    velocity_peculiar_supercomoving_code = np.asarray(
        initial.fluid.vel_supercomoving_code, dtype=float
    )
    hubble_velocity = float(cosmology.hubble(initial_time)) * scale_factor * radius_comoving_code

    figure = output.with_name("CosmologicalCorrelationInitialCondition.jpg")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].semilogx(radius_comoving_code, delta, label=r"$\delta(r)$")
    axes[0].semilogx(radius_comoving_code, mean_delta, label=r"$\bar{\delta}(<r)$")
    axes[0].set_xlabel("comoving radius [code length]")
    axes[0].set_ylabel("linear density contrast")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)
    axes[1].semilogx(radius_comoving_code, hubble_velocity, label="quiet Hubble flow")
    axes[1].semilogx(
        radius_comoving_code,
        velocity_peculiar_supercomoving_code,
        label="peculiar velocity",
    )
    axes[1].set_xlabel("comoving radius [code length]")
    axes[1].set_ylabel("initial velocity [code units]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)
    fig.suptitle("z=100 correlation-function initial condition")
    fig.tight_layout()
    fig.savefig(figure, dpi=200)
    plt.close(fig)

    print("initial condition = %s" % output)
    print("diagnostic figure = %s" % figure)
    print("initial scale factor = %.8g" % scale_factor)
    print("initial redshift = %.8g" % (1.0 / scale_factor - 1.0))
    print("target enclosed overdensity = %.8g" % float(initial_condition["initial_overdensity"]))
    print("mean overdensity at outermost cell = %.8g" % float(mean_delta[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
