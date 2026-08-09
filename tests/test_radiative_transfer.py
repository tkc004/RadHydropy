import unittest
from types import SimpleNamespace

import numpy as np
import unyt

import radhydropy.radiative_transfer as rrt
from radhydropy.units import CodeUnits, code_unit_scales


class Testing(unittest.TestCase):
    def test_cartesian_long_characteristic_attenuates_exponentially(self):
        mesh = SimpleNamespace(
            coordsys="cartesian",
            boundary=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
            vol=np.ones(3, dtype=float),
            area=np.ones(3, dtype=float),
        )
        rho = np.ones(3, dtype=float) * unyt.mp.to_value(unyt.g)
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
            np.asarray(result.face_photon_flux),
            expected_face_flux,
            rtol=1.0e-5,
        )
        expected_cell_flux = expected_face_flux[:-1] * (1.0 - np.exp(-1.0))
        np.testing.assert_allclose(
            np.asarray(result.cell_photon_flux),
            expected_cell_flux,
            rtol=1.0e-5,
        )
        np.testing.assert_allclose(result.optical_depth, np.ones(3), rtol=1.0e-5)

    def test_one_group_general_path_matches_legacy_hydrogen_path(self):
        mesh = SimpleNamespace(
            coordsys="cartesian",
            boundary=np.array([0.0, 1.0, 2.0], dtype=float),
            vol=np.ones(2, dtype=float),
            area=np.ones(2, dtype=float),
        )
        rho = np.ones(2, dtype=float) * unyt.mp.to_value(unyt.g)
        xHI = np.array([1.0, 0.5])
        legacy = rrt.trace_long_characteristics(
            mesh,
            rho,
            xHI,
            sigma_gamma=1.0 * unyt.cm**2,
            boundary_flux=10.0 / (unyt.cm**2 * unyt.s),
        )
        general = rrt.trace_long_characteristics(
            mesh,
            group_edges_eV=[13.6, 24.6],
            absorber_densities={
                "HI": rho / unyt.mp.to_value(unyt.g) * xHI,
            },
            cross_sections_cm2={"HI": [1.0 * unyt.cm**2]},
            boundary_flux=[10.0 / (unyt.cm**2 * unyt.s)],
        )

        np.testing.assert_allclose(
            general.optical_depth,
            legacy.optical_depth,
        )
        np.testing.assert_allclose(
            general.cell_photon_density,
            legacy.cell_photon_density,
        )
        self.assertEqual(general.cell_photon_density.shape, (2,))

    def test_multigroup_long_characteristics_combines_absorbers(self):
        mesh = SimpleNamespace(
            coordsys="cartesian",
            boundary=np.array([0.0, 1.0, 2.0], dtype=float),
            vol=np.ones(2, dtype=float),
            area=np.ones(2, dtype=float),
        )
        n_h = np.ones(2, dtype=float)
        n_he = np.ones(2, dtype=float) * 2.0
        result = rrt.trace_long_characteristics(
            mesh,
            group_edges_eV=[13.6, 24.6, 54.4],
            absorber_densities={"HI": n_h, "HeI": n_he},
            cross_sections_cm2={
                "HI": np.array([1.0, 2.0]),
                "HeI": np.array([0.0, 3.0]),
            },
            boundary_flux=np.array([10.0, 20.0]),
        )

        np.testing.assert_allclose(
            result.optical_depth,
            np.array([[1.0, 1.0], [8.0, 8.0]]),
        )
        np.testing.assert_allclose(
            result.face_photon_flux[0],
            10.0 * np.exp(-np.array([0.0, 1.0, 2.0])),
        )
        np.testing.assert_allclose(
            result.face_photon_flux[1],
            20.0 * np.exp(-np.array([0.0, 8.0, 16.0])),
        )
        self.assertEqual(result.cell_photon_density.shape, (2, 2))

    def test_spherical_long_characteristic_keeps_photon_rate_and_dilutes_density(self):
        mesh = SimpleNamespace(
            coordsys="spherical",
            boundary=np.array([0.0, 1.0, 2.0], dtype=float),
            vol=np.array([4.0 * np.pi / 3.0, 28.0 * np.pi / 3.0], dtype=float),
            area=np.array([0.0, 4.0 * np.pi], dtype=float),
        )
        rho = np.ones(2, dtype=float) * unyt.mp.to_value(unyt.g)
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

        np.testing.assert_allclose(np.asarray(result.face_photon_rate), 12.0)
        expected_density = (
            np.array([12.0, 12.0], dtype=float)
            / np.asarray(mesh.vol, dtype=float)
            / unyt.c.to_value(unyt.cm / unyt.s)
        )
        np.testing.assert_allclose(np.asarray(result.cell_photon_density), expected_density)
        self.assertGreater(result.cell_photon_density[0], result.cell_photon_density[1])

    def test_trace_photon_density_returns_cgs_number_density(self):
        state = {
            "boundary_cm": np.array([0.0, 1.0], dtype=float),
            "width_cm": np.array([1.0], dtype=float),
            "volume_cm3": np.array([1.0], dtype=float),
            "rho_g_cm3": np.array([unyt.mp.to_value(unyt.g)], dtype=float),
            "xHI": np.array([1.0], dtype=float),
        }
        par = SimpleNamespace(
            noghost=1,
            nogrid=1,
            coordsys="cartesian",
            radiative_transfer=True,
            radiative_transfer_method="long_characteristics",
            hydrogen_mass_fraction=1.0,
            hydrogen_sigma_gamma=1.0 * unyt.cm**2,
            radiative_transfer_boundary_flux=10.0 / (unyt.cm**2 * unyt.s),
            radiative_transfer_source_photon_rate=0.0 / unyt.s,
            radiative_transfer_direction=1,
        )

        ngamma = rrt.trace_photon_density(state, par)

        self.assertIsInstance(ngamma, np.ndarray)
        self.assertEqual(ngamma.shape, (1,))
        self.assertGreater(ngamma[0], 0.0)

    def test_trace_photon_density_converts_code_unit_parameters(self):
        code_units = CodeUnits.from_mapping(
            {
                "name": "test_units",
                "InternalUnitSystem": {
                    "UnitMass_in_cgs": 1.0e2,
                    "UnitLength_in_cgs": 1.0e1,
                    "UnitVelocity_in_cgs": 2.0e0,
                    "UnitCurrent_in_cgs": 1.0,
                    "UnitTemp_in_cgs": 1.0,
                },
            }
        )
        scales = code_unit_scales(code_units)
        state = {
            "boundary_cm": np.array([0.0, 1.0, 2.0], dtype=float),
            "width_cm": np.array([1.0, 1.0], dtype=float),
            "volume_cm3": np.array([4.0 * np.pi / 3.0, 28.0 * np.pi / 3.0], dtype=float),
            "rho_g_cm3": np.ones(2, dtype=float) * unyt.mp.to_value(unyt.g),
            "xHI": np.ones(2, dtype=float),
        }
        par = SimpleNamespace(
            CodeUnits=code_units,
            noghost=1,
            nogrid=2,
            coordsys="spherical",
            radiative_transfer=True,
            radiative_transfer_method="long_characteristics",
            hydrogen_mass_fraction=1.0,
            hydrogen_sigma_gamma=0.5,
            radiative_transfer_boundary_flux=0.0,
            radiative_transfer_source_photon_rate=3.0,
            radiative_transfer_direction=1,
        )

        result = rrt.trace_photon_density(state, par)
        expected = rrt.trace_long_characteristics(
            SimpleNamespace(
                coordsys="spherical",
                boundary=state["boundary_cm"],
                vol=state["volume_cm3"],
            ),
            state["rho_g_cm3"],
            state["xHI"],
            hydrogen_mass_fraction=1.0,
            sigma_gamma=0.5 * scales["area_cm2"] * unyt.cm**2,
            boundary_flux=0.0 * scales["photon_flux_per_cm2_s"] / (
                unyt.cm**2 * unyt.s
            ),
            source_photon_rate=3.0 * scales["photon_rate_per_s"] / unyt.s,
            direction=1,
            coordsys="spherical",
        ).cell_photon_density

        np.testing.assert_allclose(np.asarray(result), np.asarray(expected))


if __name__ == "__main__":
    unittest.main()
