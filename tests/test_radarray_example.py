from pathlib import Path

import numpy as np

from example.RadArrayConversion1D.radarray_conversion1d import run


def test_radarray_conversion_example():
    result = run(
        Path(__file__).parents[1]
        / "example"
        / "RadArrayConversion1D"
        / "radarray_conversion1d.yaml"
    )

    np.testing.assert_allclose(result["radius_proper_code"], [0.5, 1.0])
    np.testing.assert_allclose(result["rho_proper_code"], [64.0, 128.0])
    np.testing.assert_allclose(result["temp_proper_code"], [32.0])
    np.testing.assert_allclose(result["pre_proper_code"], [2048.0])
    np.testing.assert_allclose(result["vel_proper_code"], [4.14])
