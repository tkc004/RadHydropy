"""Run short central-boundary wall/origin comparisons in isolated directories."""

import argparse
from copy import deepcopy
from collections.abc import Mapping
import os
from pathlib import Path
import tempfile

import yaml
import unyt

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "radhydropy-matplotlib")
)

import cosmological_gas_correlation_z100 as experiment
import example_utils as eu


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLE_DIR / "cosmological_gas_correlation_z100.yaml"

CASES = {
    "B": {
        "label": "wall_no_thermalization",
        "inner_wall_radius_comoving": 3.0,
    },
    "C": {
        "label": "origin_no_thermalization",
        "inner_wall_radius_comoving": 0.0,
    },
}


def _yaml_value(value):
    """Convert the nested loader's unit-aware values back to YAML mappings."""
    if isinstance(value, Mapping):
        return {key: _yaml_value(item) for key, item in value.items()}
    if isinstance(value, unyt.unyt_array):
        numeric_value = value.value.tolist()
        if value.ndim == 0:
            numeric_value = value.value.item()
        return {"value": numeric_value, "unit": str(value.units)}
    if isinstance(value, list):
        return [_yaml_value(item) for item in value]
    return value


def _load_config(filename):
    """Load and validate the complete nested short-run configuration."""
    config = eu.load_nested_example_config(filename)
    if not isinstance(config, dict):
        raise ValueError("short A/B/C base configuration must be a mapping")
    return config


def _case_config(base_config, case_name, final_time):
    case = CASES[case_name]
    config = deepcopy(base_config)

    initial_condition = config["initial_condition"]

    output_dir = EXAMPLE_DIR / (
        "outputs_short_%s_%s" % (case_name, case["label"])
    )
    figure_prefix = "CosmologicalGasCorrelationShort%s" % case_name
    config["par"]["simulation"]["name"] = figure_prefix
    config["par"]["simulation"]["initial_condition_filename"] = str(
        output_dir / "InitialCondition.hdf5"
    )
    config["par"]["simulation"]["final_time"] = float(final_time)
    config["par"]["output"].update({
        "directory": str(output_dir),
        "directory": str(output_dir),
        # Keep the shared correlation table resolvable after placing the
        # effective YAML inside the case output directory.
    })
    config["example"]["linear_correlation_table_filename"] = str(
        EXAMPLE_DIR / "outputs_correlation" / "lcdm_linear_correlation.h5"
    )
    initial_condition["inner_wall_radius_comoving"] = float(
        case["inner_wall_radius_comoving"]
    )
    return config, output_dir


def run_case(base_config, case_name, final_time):
    config, output_dir = _case_config(base_config, case_name, final_time)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_filename = output_dir / "effective_config.yaml"
    with config_filename.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(_yaml_value(config), stream, sort_keys=False)

    print(
        "short case %s: inner_wall=%g comoving kpc output=%s"
        % (
            case_name,
            config["initial_condition"]["inner_wall_radius_comoving"],
            output_dir,
        ),
        flush=True,
    )
    return experiment.run(
        config_filename,
        final_time_override=float(final_time),
    )


def main(config_filename=DEFAULT_CONFIG, cases=None, final_time=1.0):
    base_config = _load_config(config_filename)
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
