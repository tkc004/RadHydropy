from pathlib import Path

from radhydropy.example_config import load_example_parameters


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "StellarWindBubble1D"


def test_stellar_wind_cie_and_no_metal_configs_select_distinct_physics():
    with_metal, with_metal_ic = load_example_parameters(
        EXAMPLE_DIR / "stellar_wind_bubble1d.yaml"
    )
    no_metal, no_metal_ic = load_example_parameters(
        EXAMPLE_DIR / "stellar_wind_bubble1d_no_metal.yaml"
    )

    assert with_metal["thermochemistry_network"] == "cie_cooling"
    assert with_metal["cie_cooling"] is True
    assert with_metal["metallicity"] == 1.0
    assert with_metal["hydrogen_mass_fraction"] == 0.7
    assert with_metal["figure_prefix"].endswith("with_metal")

    assert no_metal["thermochemistry_network"] == "hydrogen"
    assert no_metal["hydrogen_mass_fraction"] == 1.0
    assert no_metal["figure_prefix"].endswith("no_metal")
    assert no_metal.get("hydrogen_chemistry", False) is False

    assert with_metal_ic["nogrid"] == no_metal_ic["nogrid"] == 1024
    assert with_metal_ic["coordsys"] == no_metal_ic["coordsys"] == "spherical"

