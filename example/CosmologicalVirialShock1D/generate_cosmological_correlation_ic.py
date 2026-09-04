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
from radhydropy.units import CodeUnits
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name(
    "cosmological_dark_matter_correlation_z100.yaml"
)


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    config = load_nested_example_config(config_filename)
    par = config["par"]
    icparams = config["initial_condition"]
    gravity = par["gravity"]
    units = CodeUnits.from_mapping(par["units"]["CodeUnits"])
    if gravity.get("cosmology_type") in ("lambda_cdm", "LambdaCDM", "lcdm"):
        cosmology = LambdaCDM.from_code_units(
            units,
            t_ref=float(gravity["cosmology_t_ref"]),
            a_ref=float(gravity["cosmology_a_ref"]),
            omega_m=float(gravity["cosmology_omega_m"]),
            omega_lambda=float(gravity["cosmology_omega_lambda"]),
            hubble_ref=float(gravity["cosmology_hubble_ref"]),
        )
    else:
        cosmology = EinsteinDeSitter.from_code_units(
            units,
            t_ref=float(gravity["cosmology_t_ref"]),
            a_ref=float(gravity["cosmology_a_ref"]),
        )

    table_filename = Path(par["linear_correlation_table_filename"])
    if not table_filename.is_absolute():
        table_filename = config_filename.parent / table_filename
    correlation_table = et.load_lcdm_correlation_table(table_filename)

    initial = et.build_initial_condition(
        {"par": par, "initial_condition": icparams},
        units,
        cosmology,
        correlation_table=correlation_table,
    )
    output = Path(par["simulation"]["initial_condition_filename"])
    output.parent.mkdir(parents=True, exist_ok=True)
    rio.writehdf5(initial, output)

    length_unit_mpc_h = (
        float(units.length_in_cgs)
        / float((1.0 * unyt.Mpc).to_value("cm"))
        * float(icparams.get("correlation_h", 0.674))
    )
    radius = np.asarray(initial.mesh.coordinate, dtype=float)
    delta, mean_delta = et.density_contrast_profile(
        radius,
        icparams,
        cosmology,
        correlation_table=correlation_table,
        length_unit_mpc_h=length_unit_mpc_h,
    )
    initial_time = float(icparams["initial_cosmic_time"])
    scale_factor = float(cosmology.scale_factor(initial_time))
    peculiar_velocity = np.asarray(initial.fluid.vel_code, dtype=float)
    hubble_velocity = float(cosmology.hubble(initial_time)) * scale_factor * radius

    figure = output.with_name("CosmologicalCorrelationInitialCondition.jpg")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].semilogx(radius, delta, label=r"$\delta(r)$")
    axes[0].semilogx(radius, mean_delta, label=r"$\bar{\delta}(<r)$")
    axes[0].set_xlabel("comoving radius [code length]")
    axes[0].set_ylabel("linear density contrast")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)
    axes[1].semilogx(radius, hubble_velocity, label="quiet Hubble flow")
    axes[1].semilogx(radius, peculiar_velocity, label="peculiar velocity")
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
    print("target enclosed overdensity = %.8g" % float(icparams["initial_overdensity"]))
    print("mean overdensity at outermost cell = %.8g" % float(mean_delta[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)


