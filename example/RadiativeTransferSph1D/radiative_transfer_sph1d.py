# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Spherical long-characteristic radiative-transfer example.

A source at the coordinate origin emits ionizing photons at a constant rate.
Hydrodynamics and hydrogen thermo-chemistry are not advanced; the script only
applies the optional long-characteristic radiative-transfer update and compares
the resulting photon number density with the analytic optically thin spherical
dilution solution.

The example builds the static spherical problem from YAML parameters, applies
the long-characteristic radiative-transfer update once through ``Rsim``, writes
an HDF5 snapshot, reloads that snapshot, and compares the result with the
analytic optically thin spherical dilution solution.
"""  # noqa: CPY001

import argparse
import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
example_root = Path(__file__).resolve().parents[1]
if str(example_root) not in sys.path:
    sys.path.insert(0, str(example_root))

cache_dir = os.path.join(tempfile.gettempdir(), "radhydropy-cache")
mplconfig_dir = os.path.join(tempfile.gettempdir(), "radhydropy-matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
os.environ.setdefault("MPLCONFIGDIR", mplconfig_dir)

import example_utils as eu  # noqa: E402

import radhydropy.io as rio  # noqa: E402
import tools as et  # noqa: E402
from radhydropy.units import CodeUnits  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().with_name("radiative_transfer_sph1d.yaml")


def main(config_filename=DEFAULT_CONFIG):
    Path.cwd().resolve()
    nested = eu.load_nested_example_config(config_filename)

    config = nested
    config["_code_units"] = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    eu.clean_previous_outputs(nested)

    Path(nested["par"]["output"]["directory"]).mkdir(parents=True, exist_ok=True)
    Path(nested["par"]["output"]["directory"]).mkdir(parents=True, exist_ok=True)

    et.write_initial_condition(config)

    mainrun = rio.loadhdf5(
        config,
        nested["par"]["simulation"]["initial_condition_filename"],
    )
    mainrun.SetMesh()
    mainrun.SetFluid()
    mainrun.SetInitFluid()
    if (
        nested["par"].get("radiation", {}).get("radiative_transfer_temporal_scheme", "c2ray")
        == "c2ray"
    ):
        mainrun.EvolveStaticThermochemistry(
            nested["par"]["simulation"]["final_time"],
            nested["par"]["timestep"]["evolution_timestep"],
        )
    rio.write_numbered_hdf5(mainrun, 0)

    output_filename = (
        Path(nested["par"]["output"]["directory"])
        / f"{nested['par']['output']['filename_prefix']}_000.hdf5"
    )
    output_snapshot = et.load_output_state(output_filename, config)
    et.save_plot(
        output_snapshot,
        config,
        str(
            Path(nested["par"]["output"]["directory"])
            / (
                "RadiativeTransferSph1D_C2Ray.jpg"
                if nested["par"]
                .get("radiation", {})
                .get("radiative_transfer_temporal_scheme", "c2ray")
                == "c2ray"
                else "RadiativeTransferSph1D.jpg"
            ),
        ),
    )

    (
        "RadiativeTransferSph1D_C2Ray.jpg"
        if nested["par"].get("radiation", {}).get("radiative_transfer_temporal_scheme", "c2ray")
        == "c2ray"
        else "RadiativeTransferSph1D.jpg"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the spherical radiative-transfer example.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help='YAML file containing nested nested["par"] and initial-condition settings.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
