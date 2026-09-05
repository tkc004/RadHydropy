"""Contract tests for the direct nested example-configuration API."""

from pathlib import Path
import sys


EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "example"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
import example_utils


def test_removed_flat_example_helpers_are_absent():
    for removed_name in (
        "load_nested_example_parameters",
        "legacy_example_parameters",
        "legacy_initial_condition_parameters",
    ):
        assert not hasattr(example_utils, removed_name)


def test_example_config_is_consumed_as_nested_mapping():
    config = example_utils.load_nested_example_config(
        EXAMPLE_ROOT / "Advection1D" / "advection1d.yaml"
    )

    assert set(config) == {"par", "initial_condition", "example"}
    assert config["par"]["mesh"]["grid_cells"] == 100
    assert config["initial_condition"]["grid_cells"] == 100
    assert "hydrodynamics" in config["par"]
    assert isinstance(config["example"], dict)
