import unittest

import numpy as np

import radhydropy.radiative_transfer as rrt
from radhydropy.constants import PROTON_MASS_CGS
from radhydropy.thermo_networks.hydrogen_helium import (
    _closure,
    _alpha_heii,
    _alpha_heii_dielectronic,
    _alpha_heiii,
    _beta_hei,
    _beta_heii,
    _gamma_bremsstrahlung,
    _rates,
    ionization_fraction_implicit_update,
)


class HydrogenHeliumNetworkTests(unittest.TestCase):
    def _state(self):
        return {
            "rho_g_cm3": np.array([PROTON_MASS_CGS, 2.0 * PROTON_MASS_CGS]),
            "hydrogen_mass_fraction": 0.7,
            "helium_mass_fraction": 0.28,
            "temperature_K": np.array([1.0e4, 2.0e4]),
            "xHI": np.array([0.8, 0.2]),
            "xHeI": np.array([0.7, 0.1]),
            "xHeIII": np.array([0.0, 0.2]),
            "sigma_gamma_cm2": {
                "HI": np.array([1.0e-18, 2.0e-19]),
                "HeI": np.array([3.0e-18, 4.0e-19]),
                "HeII": np.array([0.0, 5.0e-19]),
            },
            "epsilon_gamma_erg": {
                "HI": np.array([1.0e-12, 2.0e-12]),
                "HeI": np.array([3.0e-12, 4.0e-12]),
                "HeII": np.array([0.0, 5.0e-12]),
            },
        }

    def test_closure_computes_electrons_and_conserves_helium(self):
        state = self._state()
        _closure(state)

        n_h = state["rho_g_cm3"] * 0.7 / PROTON_MASS_CGS
        n_he = state["rho_g_cm3"] * 0.28 / (4.0 * PROTON_MASS_CGS)
        expected_ne = n_h * (1.0 - state["xHI"]) + n_he * (
            state["xHeII"] + 2.0 * state["xHeIII"]
        )

        np.testing.assert_allclose(state["xHeI"] + state["xHeII"] + state["xHeIII"], 1.0)
        np.testing.assert_allclose(state["ne_cm3"], expected_ne)
        self.assertTrue(np.all(state["mu"] > 0.0))

    def test_multigroup_rates_and_heating_are_finite(self):
        state = self._state()
        ngamma = np.array([[1.0e-3, 2.0e-3], [3.0e-3, 4.0e-3]])

        rates = _rates(state, ngamma)
        self.assertEqual(len(rates), 4)
        for value in rates:
            self.assertTrue(np.all(np.isfinite(value)))

        photo_rates = rrt.species_photoionization_rates(ngamma, state["sigma_gamma_cm2"])
        photo_heating = rrt.species_photoionization_heating(
            ngamma,
            state["sigma_gamma_cm2"],
            state["epsilon_gamma_erg"],
        )
        self.assertEqual(photo_rates["HI"].shape, (2,))
        self.assertEqual(photo_heating["HeI"].shape, (2,))

    def test_cited_helium_rate_fits_are_finite_and_positive(self):
        temperature = np.array([1.0e4, 1.0e5])
        for rate in (
            _alpha_heii(temperature),
            _alpha_heii_dielectronic(temperature),
            _alpha_heiii(temperature),
            _beta_hei(temperature),
            _beta_heii(temperature),
            _gamma_bremsstrahlung(temperature),
        ):
            self.assertTrue(np.all(np.isfinite(rate)))
            self.assertTrue(np.all(rate > 0.0))

        # The Hummer & Storey He II fit is intentionally below the old
        # approximate 1.5e-12 cm^3/s value near 10^4 K.
        self.assertLess(float(_alpha_heii(np.array([1.0e4]))[0]), 1.0e-12)

    def test_implicit_update_keeps_hydrogen_and_helium_fractions_bounded(self):
        state = self._state()
        ngamma = np.array([[1.0e-3, 2.0e-3], [3.0e-3, 4.0e-3]])

        ionization_fraction_implicit_update(state, ngamma, 1.0e10)

        self.assertTrue(np.all((state["xHI"] > 0.0) & (state["xHI"] < 1.0)))
        self.assertTrue(np.all((state["xHeI"] > 0.0) & (state["xHeI"] < 1.0)))
        self.assertTrue(np.all(state["xHeIII"] >= 0.0))
        self.assertTrue(np.all(state["xHeI"] + state["xHeII"] + state["xHeIII"] <= 1.0 + 1.0e-12))


if __name__ == "__main__":
    unittest.main()
