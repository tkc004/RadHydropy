# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Launch the preserved LambdaCDM virial-shock comparison workflow."""

import runpy
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / (
    "CosmologicalVirialShock1DLambdaCDM/cosmological_virial_shock1d.py"
)


if __name__ == "__main__":
    runpy.run_path(str(SOURCE), run_name="__main__")
