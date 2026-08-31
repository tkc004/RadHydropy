from pathlib import Path

import example_utils


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "StellarWindBubble1D"


def test_stellar_wind_cie_and_no_metal_configs_select_distinct_physics():
    with_metal = example_utils.load_nested_example_config(
        EXAMPLE_DIR / "stellar_wind_bubble1d.yaml"
    )
    no_metal = example_utils.load_nested_example_config(
        EXAMPLE_DIR / "stellar_wind_bubble1d_no_metal.yaml"
    )

    assert with_metal["par"]["thermochemistry"]["network"] == "cie_cooling"
    assert with_metal["par"]["thermochemistry"]["cie_cooling"] is True
    assert with_metal["par"]["chemistry"]["metallicity"] == 1.0
    assert with_metal["par"]["chemistry"]["hydrogen_mass_fraction"] == 0.7
    assert with_metal["example"]["figure_prefix"].endswith("with_metal")

    assert no_metal["par"]["thermochemistry"]["network"] == "hydrogen"
    assert no_metal["par"]["chemistry"]["hydrogen_mass_fraction"] == 1.0
    assert no_metal["example"]["figure_prefix"].endswith("no_metal")

    assert with_metal["initial_condition"]["grid_cells"] == no_metal["initial_condition"]["grid_cells"] == 1024
    assert with_metal["initial_condition"]["coordinate_system"] == no_metal["initial_condition"]["coordinate_system"] == "spherical"
