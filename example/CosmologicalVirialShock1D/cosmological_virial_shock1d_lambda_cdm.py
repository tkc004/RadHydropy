"""Launch the preserved LambdaCDM virial-shock comparison workflow."""

from pathlib import Path
import runpy


SOURCE = Path(__file__).resolve().parent.parent / (
    "CosmologicalVirialShock1DLambdaCDM/cosmological_virial_shock1d.py"
)


if __name__ == "__main__":
    runpy.run_path(str(SOURCE), run_name="__main__")
