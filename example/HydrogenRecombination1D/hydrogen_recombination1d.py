"""Fixed-temperature case-B hydrogen recombination box.

The gas starts fully ionized at ``T = 2e4 K``. Hydrogen cooling/heating terms
and collisional ionization are disabled, leaving pure case-B recombination.
The run stops once the gas is 99 percent neutral and writes a JPG comparing
the ionized fraction against the analytic case-B expectation.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

cache_dir = os.path.join(tempfile.gettempdir(), "radhydropy-cache")
mplconfig_dir = os.path.join(tempfile.gettempdir(), "radhydropy-matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
os.environ.setdefault(
    "MPLCONFIGDIR",
    mplconfig_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import example_utils as eu  # noqa: E402

import radhydropy.io as rio  # noqa: E402
import tools as et  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().with_name("hydrogen_recombination1d.yaml")


def main(config_filename=DEFAULT_CONFIG):
    Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)

    config["initial_condition"]
    exampleparams = config["example"]
    output = config["par"]["output"]
    eu.clean_previous_outputs(config)
    ric = et.build_initial_condition(config)
    ic_filename = config["par"]["simulation"]["initial_condition_filename"]
    ric.write(ic_filename)
    sim = rio.loadhdf5(config, ic_filename)
    et.run_hydrogen_recombination(
        sim,
        exampleparams["target_neutral_fraction"],
        outputtime=0,
    )

    outputfiles = et.output_files(
        output["directory"],
        output["filename_prefix"],
    )
    history = et.load_history_from_outputs(
        outputfiles,
        config,
    )

    figure_filename = Path(output["directory"]) / exampleparams["plot_filename"]
    et.save_history_plot(
        history,
        str(figure_filename),
        config,
        exampleparams["target_neutral_fraction"],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 1D hydrogen recombination example.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="YAML file containing nested par, initial_condition, and example sections.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
