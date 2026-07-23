import unittest

import numpy as np
import unyt

from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
from radhydropy.solver import Solver


class Mesh:
    def __init__(self):
        self.boundary = np.linspace(0.0, 8.0, 9) * unyt.cm
        self.vol = np.ones(8) * unyt.cm**3


class Testing(unittest.TestCase):
    def test_isothermal_eos_allows_gamma_one(self):
        eos = EOS('isothermal', gamma=1.0)

        self.assertTrue(eos.is_isothermal)
        self.assertFalse(eos.is_polytropic)

    def test_polytropic_eos_rejects_gamma_one(self):
        with self.assertRaisesRegex(Exception, 'gamma cannot be equal to 1'):
            EOS('polytropic', gamma=1.0)

    def test_isothermal_sound_speed_uses_pressure_over_density(self):
        eos = EOS('isothermal', gamma=1.0)
        rho = np.array([2.0]) * unyt.g / unyt.cm**3
        pressure = np.array([18.0]) * unyt.dyn / unyt.cm**2

        sound_speed = eos.sound_speed(rho, pressure)

        expected = np.sqrt((pressure / rho)).to(unyt.cm / unyt.s)
        self.assertEqual(sound_speed.units, unyt.cm / unyt.s)
        np.testing.assert_allclose(sound_speed.value, expected.value)

    def test_isothermal_set_conserved_keeps_only_kinetic_energy(self):
        fluid = Fluid()
        fluid.eos = EOS('isothermal', gamma=1.0)
        fluid.rho = np.ones(8) * 2.0 * unyt.g / unyt.cm**3
        fluid.vel = np.ones(8) * 3.0 * unyt.cm / unyt.s
        fluid.temp = np.ones(8) * 100.0 * unyt.K
        fluid.mu = np.ones(8)
        fluid.SetPressure()

        Solver().SetConserved(Mesh(), fluid)

        expected = (0.5 * fluid.rho * fluid.vel**2 * Mesh().vol).to(fluid.Energy.units)
        np.testing.assert_allclose(fluid.Energy.value, expected.value)

    def test_isothermal_set_primitive_recovers_pressure_from_temperature(self):
        mesh = Mesh()
        fluid = Fluid()
        fluid.eos = EOS('isothermal', gamma=1.0)
        fluid.rho = np.ones(8) * unyt.g / unyt.cm**3
        fluid.vel = np.zeros(8) * unyt.cm / unyt.s
        fluid.temp = np.ones(8) * 250.0 * unyt.K
        fluid.mu = np.ones(8)
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)

        fluid.Energy[:] = 0.0 * fluid.Energy.units
        Solver().SetPrimitive(mesh, fluid)

        expected_pressure = fluid.eos.pressure(fluid.rho, fluid.temp, fluid.mu)
        np.testing.assert_allclose(fluid.pre.value, expected_pressure.value)


if __name__ == '__main__':
    unittest.main()
