import unittest
from types import SimpleNamespace

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
    source_state,
    ionization_fraction_implicit_update,
)
from radhydropy.constants import BOLTZMANN_CONSTANT_CGS
from radhydropy.units import CodeUnits


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

    def test_supercomoving_source_state_is_physical(self):
        code = CodeUnits.from_mapping({
            "InternalUnitSystem": {
                "UnitMass_in_cgs": 1.0,
                "UnitLength_in_cgs": 1.0,
                "UnitVelocity_in_cgs": 1.0,
                "UnitCurrent_in_cgs": 1.0,
                "UnitTemp_in_cgs": 1.0,
            }
        })
        scale_factor = 2.0
        gamma = 5.0 / 3.0
        temperature = 100.0
        mu = 1.0 / (0.75 + 0.25 / 4.0)
        specific_internal = (
            BOLTZMANN_CONSTANT_CGS * temperature
            / ((gamma - 1.0) * mu * PROTON_MASS_CGS)
        )
        velocity_super = 2.0
        fluid = SimpleNamespace(
            rho=np.array([8.0]),
            vel=np.array([velocity_super]),
            temp=np.array([temperature * scale_factor**2]),
            mu=np.array([mu]),
            Mass=np.array([8.0]),
            Energy=np.array([
                8.0 * (specific_internal * scale_factor**2
                        + 0.5 * velocity_super**2)
            ]),
            xHI=np.array([1.0]),
            xHeI=np.array([1.0]),
            xHeII=np.array([0.0]),
            eos=SimpleNamespace(gamma=gamma),
        )
        mesh = SimpleNamespace(
            boundary=np.array([0.0, 1.0]),
            vol=np.array([1.0]),
            coordinate=np.array([0.5]),
        )
        par = SimpleNamespace(
            CodeUnits=code,
            noghost=0,
            nogrid=1,
            gamma=gamma,
            hydrogen_mass_fraction=0.75,
            helium_mass_fraction=0.25,
            radiation_group_sigma_gamma=np.array([1.0e-18]),
            radiation_group_epsilon_gamma=np.array([1.0e-11]),
            supercomoving_coordinates=True,
            fluid_time=0.0,
            cosmology=SimpleNamespace(
                scale_factor_from_supercomoving=lambda _: scale_factor,
            ),
        )

        state = source_state(mesh, fluid, par)

        np.testing.assert_allclose(state["rho_g_cm3"], [1.0])
        np.testing.assert_allclose(state["volume_cm3"], [8.0])
        np.testing.assert_allclose(state["radius_kpc"], [1.0 / 3.08567758e21])
        np.testing.assert_allclose(state["temperature_K"], [temperature])
        np.testing.assert_allclose(state["specific_energy_erg_g"], [specific_internal])
        np.testing.assert_allclose(state["velocity_supercomoving_cm_s"], [velocity_super])
        self.assertEqual(state["source_scale_factor"], scale_factor)
        self.assertEqual(state["source_temperature_factor"], scale_factor**2)


if __name__ == "__main__":
    unittest.main()
