# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Contract tests for the direct nested example-configuration API."""  # noqa: CPY001

import sys
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "example"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
from radhydropy.example_config import load_example_config  # noqa: E402


def test_example_config_is_consumed_as_nested_mapping():
    config = load_example_config(
        EXAMPLE_ROOT / "Advection1D" / "advection1d.yaml",
    )

    assert set(config) == {"par", "initial_condition", "example"}
    assert config["par"]["mesh"]["grid_cells"] == 100
    assert config["initial_condition"]["grid_cells"] == 100
    assert "hydrodynamics" in config["par"]
    assert isinstance(config["example"], dict)
