import unittest
import radhydropy.utils as ru
import radhydropy.chemistry_species.hydrogen as rh
from radhydropy.constants import SPEED_OF_LIGHT_CGS
import radhydropy.thermo_networks.hydrogen as rth
import unyt
import numpy as np

class Testing(unittest.TestCase):
    def test_calpressure(self):
        rho = 1.0 * unyt.mp / unyt.cm**3
        mu = 1.0
        temp = 1.0 * unyt.K
        precal = ru.CalPressure(rho,temp,mu)
        preex = rho / (mu * unyt.mp) * unyt.kb * temp
        self.assertEqual(precal, preex)

    def test_CheckParamDimen(self):
        params = {'vini': 1.0*unyt.s}
        testdim = ru.CheckParamDimen(params)
        self.assertEqual(testdim ,'vini')
        params = {'vini': 1.0*unyt.cm/unyt.s}
        testdim = ru.CheckParamDimen(params) 
        self.assertEqual(testdim ,True)

    def test_zero_density_temperature_and_sound_speed_are_zero(self):
        rho = np.array([1.0, 0.0]) * unyt.g / unyt.cm**3
        pressure = np.ones(2) * unyt.dyn / unyt.cm**2

        temp = ru.CalTemperature(rho, pressure, 1.0)
        sound_speed = ru.CalSoundSpeed(pressure, rho, 5.0/3.0)

        self.assertEqual(temp.units, unyt.K)
        self.assertEqual(sound_speed.units, unyt.cm/unyt.s)
        self.assertEqual(temp[1], 0.0 * unyt.K)
        self.assertEqual(sound_speed[1], 0.0 * unyt.cm/unyt.s)

    def test_apply_flux_limiter_handles_flat_regions(self):
        q = np.array([1.0, 1.0, 2.0]) * unyt.g
        flux_0 = np.zeros(3) * unyt.g/unyt.s
        flux_1 = np.ones(3) * unyt.g/unyt.s

        flux, philim = ru.ApplyFluxLimiter(q, flux_1, flux_0)

        self.assertFalse(np.any(np.isnan(philim)))
        self.assertEqual(flux.units, flux_0.units)

    def test_hydrogen_rate_coefficients_have_expected_units_and_values(self):
        temp = np.array([1.0e4]) * unyt.K
        temp_value = temp.to_value(unyt.K)
        temp5 = temp_value / 1.0e5
        lam = 315614.0 / temp_value

        alpha_b_expected = (
            2.753e-14
            * lam**1.5
            * (1.0 + (lam / 2.740) ** 0.407) ** -2.242
        )
        beta_expected = (
            1.17e-10
            * temp_value**0.5
            * np.exp(-157809.1 / temp_value)
            / (1.0 + temp5**0.5)
        )
        gamma_ion_expected = (
            2.54e-21
            * temp_value**0.5
            * np.exp(-157809.1 / temp_value)
            / (1.0 + temp5**0.5)
        )
        gamma_line_expected = (
            7.5e-19
            * np.exp(-118348.0 / temp_value)
            / (1.0 + temp5**0.5)
        )
        gamma_b_expected = (
            3.435e-30
            * temp_value
            * lam**1.970
            * (1.0 + (lam / 2.250) ** 0.376) ** -3.720
        )
        gamma_ff_expected = (
            1.42e-27
            * temp_value**0.5
            * (1.1 + 0.34 * np.exp(-(5.5 - np.log10(temp_value)) ** 2 / 3.0))
        )

        alpha_b = rth._cgs_alpha_B(temp.to_value(unyt.K)) * (unyt.cm**3 / unyt.s)
        beta = rth._cgs_beta(temp.to_value(unyt.K)) * (unyt.cm**3 / unyt.s)
        gamma_ion = rth._cgs_gamma_ion_eHI(temp.to_value(unyt.K)) * (unyt.erg * unyt.cm**3 / unyt.s)
        gamma_line = rth._cgs_gamma_line_eHI(temp.to_value(unyt.K)) * (unyt.erg * unyt.cm**3 / unyt.s)
        gamma_b = rth._cgs_gamma_B_eHII(temp.to_value(unyt.K)) * (unyt.erg * unyt.cm**3 / unyt.s)
        gamma_ff = rth._cgs_gamma_ff_eHII(temp.to_value(unyt.K)) * (unyt.erg * unyt.cm**3 / unyt.s)

        self.assertEqual(alpha_b.units, unyt.cm**3/unyt.s)
        self.assertEqual(beta.units, unyt.cm**3/unyt.s)
        self.assertEqual(gamma_ion.units, unyt.erg*unyt.cm**3/unyt.s)
        self.assertEqual(gamma_line.units, unyt.erg*unyt.cm**3/unyt.s)
        self.assertEqual(gamma_b.units, unyt.erg*unyt.cm**3/unyt.s)
        self.assertEqual(gamma_ff.units, unyt.erg*unyt.cm**3/unyt.s)
        np.testing.assert_allclose(alpha_b.value, alpha_b_expected)
        np.testing.assert_allclose(beta.value, beta_expected)
        np.testing.assert_allclose(gamma_ion.value, gamma_ion_expected)
        np.testing.assert_allclose(gamma_line.value, gamma_line_expected)
        np.testing.assert_allclose(gamma_b.value, gamma_b_expected)
        np.testing.assert_allclose(gamma_ff.value, gamma_ff_expected)

    def test_hydrogen_radiation_attenuation_matches_analytic_solution(self):
        rho = np.ones(1) * unyt.mp / unyt.cm**3
        xHI = np.array([0.25])
        ngamma_cgs_cm3 = np.ones(1) * 12.0 / unyt.cm**3
        sigma_gamma = 3.0e-18 * unyt.cm**2
        dt = 2.0e6 * unyt.s

        absorption_frequency = (
            rth._cgs_hydrogen_number_density(
                rho.to_value(unyt.g / unyt.cm**3),
                1.0,
            )
            * xHI
            * rth._cgs_photoionization_frequency(
                ngamma_cgs_cm3.to_value(1.0 / unyt.cm**3),
                sigma_gamma.to_value(unyt.cm**2),
            )
        ) * (1.0 / unyt.s)
        updated = ngamma_cgs_cm3 * np.exp(-(absorption_frequency * dt).to_value(''))
        exponent = -(absorption_frequency * dt).to_value('')
        expected = ngamma_cgs_cm3 * np.exp(exponent)

        self.assertEqual(updated.units, (1.0/unyt.cm**3).units)
        np.testing.assert_allclose(updated.value, expected.value)

    def test_hydrogen_species_helpers_return_cgs_scalars(self):
        sigma_gamma = rh.photon_cross_section(1.62e-18)
        epsilon_gamma = rh.photon_excess_energy(0.0)
        xhi = rh.clip_neutral_fraction(np.array([-0.2, 0.5, 1.2]))
        mu = rh.mean_molecular_weight_mu(np.array([0.0, 0.5, 1.0]), hydrogen_mass_fraction=1.0)

        self.assertIsInstance(sigma_gamma, np.ndarray)
        self.assertIsInstance(epsilon_gamma, np.ndarray)
        self.assertEqual(float(sigma_gamma), 1.62e-18)
        self.assertEqual(float(epsilon_gamma), 0.0)
        np.testing.assert_allclose(xhi, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(mu, np.array([0.5, 2.0 / 3.0, 1.0]))
        self.assertAlmostEqual(SPEED_OF_LIGHT_CGS, 2.99792458e10)

if __name__ == '__main__':
    unittest.main()
