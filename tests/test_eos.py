import unittest
from types import SimpleNamespace

import numpy as np
import unyt

from radhydropy.eos import EOS
from radhydropy.fluid import Fluid
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
)


class Mesh:
    def __init__(self):
        self.geometry_state = MeshGeometryState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            x_proper_code=np.arange(8, dtype=float) + 0.5,
            boundary_proper_code=np.linspace(0.0, 8.0, 9),
            width_proper_code=np.ones(8), area_proper_code=np.ones(8), volume_proper_code=np.ones(8),
        )


CODE_UNITS = CodeUnits.from_mapping(
    {
        "name": "test_units",
        "InternalUnitSystem": {
            "UnitMass_in_cgs": 1.0,
            "UnitLength_in_cgs": 1.0,
            "UnitVelocity_in_cgs": 1.0,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        },
    }
)


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

    def test_isothermal_energy_excludes_pressure_term(self):
        eos = EOS('isothermal', gamma=1.0)
        rho = np.array([2.0, 3.0])
        vel = np.array([4.0, 5.0])
        pressure = np.array([18.0, 27.0])

        np.testing.assert_array_equal(
            eos.thermal_energy_density(pressure), np.zeros(2)
        )
        np.testing.assert_allclose(
            eos.total_energy_density(rho, vel, pressure),
            0.5 * rho * vel**2,
        )

    def test_isothermal_set_conserved_keeps_only_kinetic_energy(self):
        fluid = Fluid()
        fluid.eos = EOS('isothermal', gamma=1.0, code_units=CODE_UNITS)
        fluid.rho_code = np.ones(8) * 2.0
        fluid.vel_code = np.ones(8) * 3.0
        fluid.temp_code = np.ones(8) * 100.0
        fluid.mu = np.ones(8)
        fluid.SetPressure()
        fluid.runtime_state = FluidRuntimeState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            rho_proper_code=fluid.rho_code, vel_proper_code=fluid.vel_code,
            pre_proper_code=fluid.pre_code, temp_proper_code=fluid.temp_code, time_proper_code=0.0,
            mu_dimensionless=fluid.mu,
        )

        Solver().SetConserved(Mesh(), fluid)

        expected = 0.5 * fluid.rho_code * fluid.vel_code**2 * Mesh().geometry_state.volume_proper_code
        np.testing.assert_allclose(np.asarray(fluid.Energy_code), expected)

    def test_isothermal_rejects_dual_energy(self):
        fluid = Fluid()
        fluid.eos = EOS('isothermal', gamma=1.0, code_units=CODE_UNITS)
        mesh = Mesh()
        mesh._par = SimpleNamespace(dual_energy=True)

        with self.assertRaisesRegex(ValueError, 'dual energy is not supported'):
            Solver().SetConserved(mesh, fluid)

    def test_isothermal_set_primitive_recovers_pressure_from_temperature(self):
        mesh = Mesh()
        fluid = Fluid()
        fluid.eos = EOS('isothermal', gamma=1.0, code_units=CODE_UNITS)
        fluid.rho_code = np.ones(8)
        fluid.vel_code = np.zeros(8)
        fluid.temp_code = np.ones(8) * 250.0
        fluid.mu = np.ones(8)
        fluid.SetPressure()
        fluid.runtime_state = FluidRuntimeState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            rho_proper_code=fluid.rho_code, vel_proper_code=fluid.vel_code,
            pre_proper_code=fluid.pre_code, temp_proper_code=fluid.temp_code, time_proper_code=0.0,
            mu_dimensionless=fluid.mu,
        )
        Solver().SetConserved(mesh, fluid)

        fluid.Energy_code[:] = 0.0
        Solver().SetPrimitive(mesh, fluid)

        expected_pressure = fluid.eos.pressure(fluid.rho_code, fluid.temp_code, fluid.mu)
        np.testing.assert_allclose(np.asarray(fluid.pre_code), np.asarray(expected_pressure))

    def test_code_unit_temperature_is_zero_for_zero_density(self):
        eos = EOS('polytropic', gamma=5.0 / 3.0, code_units=CODE_UNITS)
        temperature = eos.temperature(
            np.array([0.0, 2.0]),
            np.array([0.0, 6.0]),
            np.ones(2),
        )
        self.assertTrue(np.isfinite(temperature).all())
        self.assertEqual(temperature[0], 0.0)


if __name__ == '__main__':
    unittest.main()
