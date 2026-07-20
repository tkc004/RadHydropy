import unittest
import radhydropy.utils as ru
import radhydropy.hydrogen as rh
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

        alpha_a_expected = (
            1.269e-13
            * lam**1.503
            * (1.0 + (lam / 0.522) ** 0.470) ** -1.923
        )
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
        gamma_a_expected = (
            1.778e-29
            * temp_value
            * lam**1.965
            * (1.0 + (lam / 0.541) ** 0.502) ** -2.697
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

        alpha_a = rh.alpha_A(temp)
        alpha_b = rh.alpha_B(temp)
        beta = rh.beta(temp)
        gamma_ion = rh.gamma_ion_eHI(temp)
        gamma_line = rh.gamma_line_eHI(temp)
        gamma_a = rh.gamma_A_eHII(temp)
        gamma_b = rh.gamma_B_eHII(temp)
        gamma_ff = rh.gamma_ff_eHII(temp)

        self.assertEqual(alpha_a.units, unyt.cm**3/unyt.s)
        self.assertEqual(alpha_b.units, unyt.cm**3/unyt.s)
        self.assertEqual(beta.units, unyt.cm**3/unyt.s)
        self.assertEqual(gamma_ion.units, unyt.erg*unyt.cm**3/unyt.s)
        self.assertEqual(gamma_line.units, unyt.erg*unyt.cm**3/unyt.s)
        self.assertEqual(gamma_a.units, unyt.erg*unyt.cm**3/unyt.s)
        self.assertEqual(gamma_b.units, unyt.erg*unyt.cm**3/unyt.s)
        self.assertEqual(gamma_ff.units, unyt.erg*unyt.cm**3/unyt.s)
        np.testing.assert_allclose(alpha_a.value, alpha_a_expected)
        np.testing.assert_allclose(alpha_b.value, alpha_b_expected)
        np.testing.assert_allclose(beta.value, beta_expected)
        np.testing.assert_allclose(gamma_ion.value, gamma_ion_expected)
        np.testing.assert_allclose(gamma_line.value, gamma_line_expected)
        np.testing.assert_allclose(gamma_a.value, gamma_a_expected)
        np.testing.assert_allclose(gamma_b.value, gamma_b_expected)
        np.testing.assert_allclose(gamma_ff.value, gamma_ff_expected)

    def test_hydrogen_source_terms_return_expected_dimensions(self):
        rho = np.ones(2) * unyt.mp / unyt.cm**3
        temp = np.ones(2) * 1.0e4 * unyt.K
        xHI = np.array([0.5, 0.9])

        thermal_rate, neutral_fraction_rate = rh.hydrogen_source_terms(rho, temp, xHI)

        self.assertEqual(thermal_rate.units, unyt.erg/unyt.cm**3/unyt.s)
        self.assertEqual(neutral_fraction_rate.units, (1.0/unyt.s).units)
        self.assertTrue(np.all(thermal_rate <= 0.0 * thermal_rate.units))

    def test_hydrogen_implicit_neutral_fraction_update_satisfies_backward_euler(self):
        rho = np.ones(2) * unyt.mp / unyt.cm**3
        temp = np.ones(2) * 2.0e4 * unyt.K
        xHI = np.array([0.2, 0.8])
        dt = 1.0e10 * unyt.s

        updated = rh.hydrogen_neutral_fraction_implicit_update(rho, temp, xHI, dt)
        nH = rh.hydrogen_number_density(rho)
        recombination_rate = (nH * rh.alpha_B(temp)).to_value(1.0/unyt.s)
        ionization_rate = (nH * rh.beta(temp)).to_value(1.0/unyt.s)
        residual = updated - xHI - dt.to_value(unyt.s) * (
            recombination_rate * (1.0 - updated)**2
            - ionization_rate * updated * (1.0 - updated)
        )

        np.testing.assert_allclose(residual, np.zeros_like(updated), atol=1.0e-14)
        self.assertTrue(np.all(updated >= 0.0))
        self.assertTrue(np.all(updated <= 1.0))
    
    


if __name__ == '__main__':
    unittest.main()
