"""Run short central-boundary A/B/C comparisons in isolated directories.

Case A keeps the current finite inner wall and central kinetic
thermalization.  Case B keeps the wall and disables thermalization.  Case C
disables thermalization and restores the zero-area spherical origin.
"""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import tempfile

import yaml

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "radhydropy-matplotlib")
)

import cosmological_gas_correlation_z100 as experiment


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLE_DIR / "cosmological_gas_correlation_z100.yaml"

CASES = {
    "A": {
        "label": "wall_thermalization",
        "central_kinetic_thermalization": True,
        "inner_wall_radius_comoving": 3.0,
    },
    "B": {
        "label": "wall_no_thermalization",
        "central_kinetic_thermalization": False,
        "inner_wall_radius_comoving": 3.0,
    },
    "C": {
        "label": "origin_no_thermalization",
        "central_kinetic_thermalization": False,
        "inner_wall_radius_comoving": 0.0,
    },
}


def _load_raw_config(filename):
    with Path(filename).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("short A/B/C base configuration must be a mapping")
    return config


def _case_config(base_config, case_name, final_time):
    case = CASES[case_name]
    config = deepcopy(base_config)
    runparams = config["runparams"]
    icparams = config["ICparams"]

    output_dir = EXAMPLE_DIR / (
        "outputs_short_%s_%s" % (case_name, case["label"])
    )
    figure_prefix = "CosmologicalGasCorrelationShort%s" % case_name
    runparams.update({
        "simname": figure_prefix,
        "figure_prefix": figure_prefix,
        "ICfilename": str(output_dir / "InitialCondition.hdf5"),
        "outdir": str(output_dir),
        "savedir": str(output_dir),
        "final_cosmic_time": float(final_time),
        "central_kinetic_thermalization": bool(
            case["central_kinetic_thermalization"]
        ),
        # Keep the shared correlation table resolvable after placing the
        # effective YAML inside the case output directory.
        "linear_correlation_table_filename": str(
            EXAMPLE_DIR / "outputs_correlation" / "lcdm_linear_correlation.h5"
        ),
    })
    icparams["inner_wall_radius_comoving"] = float(
        case["inner_wall_radius_comoving"]
    )
    return config, output_dir


def run_case(base_config, case_name, final_time):
    config, output_dir = _case_config(base_config, case_name, final_time)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_filename = output_dir / "effective_config.yaml"
    with config_filename.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    print(
        "short case %s: thermalization=%s inner_wall=%g comoving kpc output=%s"
        % (
            case_name,
            config["runparams"]["central_kinetic_thermalization"],
            config["ICparams"]["inner_wall_radius_comoving"],
            output_dir,
        ),
        flush=True,
    )
    return experiment.run(
        config_filename,
        final_time_override=float(final_time),
    )


def main(config_filename=DEFAULT_CONFIG, cases=None, final_time=1.0):
    base_config = _load_raw_config(config_filename)
    selected = list(CASES) if cases is None else list(cases)
    outputs = []
    for case_name in selected:
        outputs.append(run_case(base_config, case_name, final_time))
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run isolated short A/B/C cosmological shock comparisons."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--case", action="append", choices=tuple(CASES), dest="cases",
        help="run only the selected case; repeat to select multiple cases",
    )
    parser.add_argument(
        "--final-time", type=float, default=1.0,
        help="final cosmic time in code time units (default: 1.0)",
    )
    args = parser.parse_args()
    main(args.config, args.cases, args.final_time)
