import unittest
from types import SimpleNamespace

import numpy as np
import unyt

import radhydropy.radiative_transfer as rrt


class Testing(unittest.TestCase):
    def test_cartesian_long_characteristic_attenuates_exponentially(self):
        mesh = SimpleNamespace(
            coordsys="cartesian",
            boundary=np.array([0.0, 1.0, 2.0, 3.0]) * unyt.cm,
            vol=np.ones(3) * unyt.cm**3,
            area=np.ones(3) * unyt.cm**2,
        )
        rho = np.ones(3) * unyt.mp / unyt.cm**3
        xHI = np.ones(3)
        boundary_flux = 10.0 / (unyt.cm**2 * unyt.s)

        result = rrt.trace_long_characteristics(
            mesh,
            rho,
            xHI,
            sigma_gamma=1.0 * unyt.cm**2,
            boundary_flux=boundary_flux,
        )

        expected_face_flux = 10.0 * np.exp(-np.arange(4))
        np.testing.assert_allclose(
            result.face_photon_flux.to_value(1.0 / (unyt.cm**2 * unyt.s)),
            expected_face_flux,
        )
        expected_cell_flux = expected_face_flux[:-1] * (1.0 - np.exp(-1.0))
        np.testing.assert_allclose(
            result.cell_photon_flux.to_value(1.0 / (unyt.cm**2 * unyt.s)),
            expected_cell_flux,
        )
        np.testing.assert_allclose(result.optical_depth, np.ones(3))

    def test_spherical_long_characteristic_keeps_photon_rate_and_dilutes_density(self):
        mesh = SimpleNamespace(
            coordsys="spherical",
            boundary=np.array([0.0, 1.0, 2.0]) * unyt.cm,
            vol=np.array([4.0 * np.pi / 3.0, 28.0 * np.pi / 3.0]) * unyt.cm**3,
            area=np.array([0.0, 4.0 * np.pi]) * unyt.cm**2,
        )
        rho = np.ones(2) * unyt.mp / unyt.cm**3
        xHI = np.ones(2)
        source_photon_rate = 12.0 / unyt.s

        result = rrt.trace_long_characteristics(
            mesh,
            rho,
            xHI,
            sigma_gamma=0.0 * unyt.cm**2,
            source_photon_rate=source_photon_rate,
            coordsys="spherical",
        )

        np.testing.assert_allclose(result.face_photon_rate.to_value(1.0 / unyt.s), 12.0)
        expected_density = (
            source_photon_rate
            * np.array([1.0, 1.0])
            * unyt.cm
            / mesh.vol
            / unyt.c.to(unyt.cm / unyt.s)
        ).to_value(1.0 / unyt.cm**3)
        np.testing.assert_allclose(
            result.cell_photon_density.to_value(1.0 / unyt.cm**3),
            expected_density,
        )
        self.assertGreater(result.cell_photon_density[0], result.cell_photon_density[1])

    def test_apply_long_characteristics_to_fluid_populates_ngamma(self):
        mesh = SimpleNamespace(
            coordsys="cartesian",
            boundary=np.array([-1.0, 0.0, 1.0, 2.0]) * unyt.cm,
            vol=np.ones(3) * unyt.cm**3,
            area=np.ones(3) * unyt.cm**2,
        )
        fluid = SimpleNamespace(
            rho=np.ones(3) * unyt.mp / unyt.cm**3,
            xHI=np.ones(3),
        )
        par = SimpleNamespace(
            noghost=1,
            nogrid=1,
            hydrogen_mass_fraction=1.0,
            hydrogen_sigma_gamma=1.0 * unyt.cm**2,
            radiative_transfer=True,
            radiative_transfer_method="long_characteristics",
            radiative_transfer_boundary_flux=10.0 / (unyt.cm**2 * unyt.s),
            radiative_transfer_source_photon_rate=0.0 / unyt.s,
            radiative_transfer_direction=1,
        )

        result = rrt.apply_long_characteristics_to_fluid(mesh, fluid, par)

        self.assertIsNotNone(result)
        self.assertTrue(hasattr(fluid, "ngamma"))
        self.assertGreater(fluid.ngamma[1], 0.0 / unyt.cm**3)
        self.assertEqual(fluid.ngamma[0], 0.0 / unyt.cm**3)


if __name__ == "__main__":
    unittest.main()
