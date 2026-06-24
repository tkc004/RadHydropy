import unittest

import numpy as np
import unyt

from radhydropy.solver import Solver


class Par:
    def __init__(self, boundcond):
        self.boundcond = boundcond
        self.noghost = 2
        self.nogrid = 4
        self.CFL = 0.1
        self.dtmin = 1.0e-8 * unyt.s
        self.dtmax = 1.0 * unyt.s
        self.rho_inflow = 9.0 * unyt.g/unyt.cm**3
        self.vel_inflow = 8.0 * unyt.cm/unyt.s
        self.temp_inflow = 0.0 * unyt.K
        self.mu_inflow = 1.0
        self.rho_outflow = 7.0 * unyt.g/unyt.cm**3
        self.vel_outflow = 6.0 * unyt.cm/unyt.s
        self.temp_outflow = 0.0 * unyt.K
        self.mu_outflow = 1.0


class EOS:
    gamma = 5.0/3.0


class Fluid:
    def __init__(self):
        self.rho = np.arange(8, dtype=float) * unyt.g/unyt.cm**3
        self.vel = np.arange(10, 18, dtype=float) * unyt.cm/unyt.s
        self.pre = np.arange(20, 28, dtype=float) * unyt.dyn/unyt.cm**2
        self.eos = EOS()


class Mesh:
    def __init__(self):
        self.vol = np.ones(3) * unyt.cm**3
        self.xdelta = np.ones(3) * unyt.cm


class Testing(unittest.TestCase):
    def test_open_boundary_fills_all_ghost_cells(self):
        fluid = Fluid()
        Solver().SetBoundary(None, fluid, Par('Open'))

        np.testing.assert_array_equal(fluid.rho[:2].value, [2.0, 2.0])
        np.testing.assert_array_equal(fluid.rho[-2:].value, [5.0, 5.0])
        np.testing.assert_array_equal(fluid.vel[:2].value, [12.0, 12.0])
        np.testing.assert_array_equal(fluid.vel[-2:].value, [15.0, 15.0])

    def test_periodic_boundary_wraps_interior(self):
        fluid = Fluid()
        Solver().SetBoundary(None, fluid, Par('Periodic'))

        np.testing.assert_array_equal(fluid.rho[:2].value, [4.0, 5.0])
        np.testing.assert_array_equal(fluid.rho[-2:].value, [2.0, 3.0])

    def test_reflecting_boundary_reverses_velocity(self):
        fluid = Fluid()
        Solver().SetBoundary(None, fluid, Par('Reflecting'))

        np.testing.assert_array_equal(fluid.rho[:2].value, [3.0, 2.0])
        np.testing.assert_array_equal(fluid.rho[-2:].value, [5.0, 4.0])
        np.testing.assert_array_equal(fluid.vel[:2].value, [-13.0, -12.0])
        np.testing.assert_array_equal(fluid.vel[-2:].value, [-15.0, -14.0])

    def test_set_primitive_handles_zero_mass(self):
        fluid = Fluid()
        fluid.Mass = np.array([1.0, 0.0, 2.0]) * unyt.g
        fluid.Mom = np.array([2.0, 1.0, 0.0]) * unyt.g*unyt.cm/unyt.s
        fluid.Energy = np.array([10.0, 5.0, 1.0]) * unyt.g*unyt.cm**2/unyt.s**2

        Solver().SetPrimitive(Mesh(), fluid)

        self.assertEqual(fluid.vel[1], 0.0 * unyt.cm/unyt.s)
        self.assertFalse(np.any(np.isnan(fluid.rho)))
        self.assertFalse(np.any(np.isnan(fluid.vel)))
        self.assertFalse(np.any(np.isnan(fluid.pre)))


if __name__ == '__main__':
    unittest.main()
