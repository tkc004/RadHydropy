import unittest
import radhydropy.utils as ru
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
    
    


if __name__ == '__main__':
    unittest.main()
