"""Launch the preserved LambdaCDM dark-matter-only workflow."""

from pathlib import Path
import runpy
import sys


SOURCE = Path(__file__).resolve().parent.parent / (
    "CosmologicalVirialShock1D/cosmological_dark_matter_only_lambda_cdm.py"
)


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend([
            "--config",
            str(Path(__file__).with_name(
                "cosmological_dark_matter_correlation_z100_lambda_cdm.yaml"
            )),
        ])
    runpy.run_path(str(SOURCE), run_name="__main__")
