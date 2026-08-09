"""Five-group H/He multifrequency static Strömgren sphere, evolved to 100 Myr."""

import argparse
import sys
from pathlib import Path

source_example = Path(__file__).resolve().parents[1] / "MultiFrequencyRadiativeTransferSph1D"
sys.path.insert(0, str(source_example))

from multifrequency_radiative_transfer_sph1d import main as run_example


DEFAULT_CONFIG = Path(__file__).with_name(
    "multifrequency_radiative_transfer_sph1d_hhe_100myr.yaml"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    run_example(parser.parse_args().config)
