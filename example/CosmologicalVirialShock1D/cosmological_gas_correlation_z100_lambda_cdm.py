"""Launch the preserved LambdaCDM gas-correlation workflow.

The implementation and configuration remain in the historical
``CosmologicalVirialShock1DLambdaCDM`` directory.  This entry point makes the
workflow discoverable from the canonical cosmological virial-shock example
without copying or modifying either original directory.
"""

from pathlib import Path
import runpy
import sys


SOURCE = Path(__file__).resolve().parent.parent / (
    "CosmologicalVirialShock1D/cosmological_gas_correlation_z100.py"
)


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend([
            "--config",
            str(Path(__file__).with_name(
                "cosmological_gas_correlation_z100_lambda_cdm.yaml"
            )),
        ])
    runpy.run_path(str(SOURCE), run_name="__main__")
