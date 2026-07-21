import unittest

import numpy as np
import unyt

import radhydropy.hydrogen as rh


class Testing(unittest.TestCase):
    def test_implicit_neutral_fraction_update_uses_fixed_recombination_rate(self):
        rho = np.ones(1) * unyt.mp / unyt.cm**3
        temperature = np.ones(1) * unyt.K
        xHI = np.zeros(1)

        updated = rh.hydrogen_neutral_fraction_implicit_update(
            rho,
            temperature,
            xHI,
            1.0 * unyt.s,
            recombination=True,
            collisional_ionization=False,
            recombination_coefficient=2.0 * unyt.cm**3 / unyt.s,
        )

        np.testing.assert_allclose(updated, [0.5])

    def test_neutral_fraction_rate_uses_fixed_collisional_ionization_rate(self):
        rho = np.ones(1) * unyt.mp / unyt.cm**3
        temperature = np.ones(1) * unyt.K
        xHI = np.ones(1) * 0.5

        rate = rh.hydrogen_neutral_fraction_rate(
            rho,
            temperature,
            xHI,
            recombination=False,
            collisional_ionization=True,
            ionization_coefficient=4.0 * unyt.cm**3 / unyt.s,
        )

        np.testing.assert_allclose(rate.to_value(1.0 / unyt.s), [-1.0])


if __name__ == "__main__":
    unittest.main()
