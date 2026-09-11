import unittest

import numpy as np

from radhydropy.cosmology_context import CosmologyContext
from radhydropy.field_metadata import _FIELD_DEFINITIONS, field_spec
from radhydropy.radarray import (
    RadArray,
    RadQuantity,
    RepresentationMismatchError,
)
from radhydropy.units import CodeUnits


CODE_UNITS = CodeUnits.from_mapping(
    {
        "InternalUnitSystem": {
            "UnitMass_in_cgs": 4.92e31,
            "UnitLength_in_cgs": 3.08567758e21,
            "UnitVelocity_in_cgs": 1.0e5,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        }
    }
)
CONTEXT = CosmologyContext(
    gamma=5.0 / 3.0,
    scale_factor=0.5,
    hubble_parameter_km_s_Mpc=70.0,
)


def _array(values, field_name):
    return RadArray(
        values,
        code_units=CODE_UNITS,
        field_spec=field_spec(
            field_name,
            CODE_UNITS,
            hubble_parameter_km_s_Mpc=(
                CONTEXT.hubble_parameter_km_s_Mpc
                if field_name == "vel_supercomoving_code"
                else None
            ),
        ),
        cosmology=CONTEXT,
    )


def _quantity(value, field_name):
    return RadQuantity(
        value,
        code_units=CODE_UNITS,
        field_spec=field_spec(
            field_name,
            CODE_UNITS,
            hubble_parameter_km_s_Mpc=(
                CONTEXT.hubble_parameter_km_s_Mpc
                if field_name == "vel_supercomoving_code"
                else None
            ),
        ),
        cosmology=CONTEXT,
    )


class RadArrayTests(unittest.TestCase):
    def test_scalar_quantity_round_trips_all_cosmological_fields(self):
        cases = (
            ("boundary_comoving_code", "boundary_proper_code", 2.0, 1.0),
            ("rho_comoving_code", "rho_proper_code", 8.0, 64.0),
            ("temp_supercomoving_code", "temp_proper_code", 8.0, 32.0),
            ("pre_supercomoving_code", "pre_proper_code", 64.0, 2048.0),
        )
        for source_name, target_name, source_value, target_value in cases:
            with self.subTest(source_name=source_name):
                source = _quantity(source_value, source_name)
                proper = source.to_proper()
                self.assertIsInstance(proper, RadQuantity)
                self.assertEqual(
                    proper.field_spec,
                    field_spec(
                        target_name,
                        CODE_UNITS,
                        cosmology=CONTEXT.cosmology,
                        scale_factor=CONTEXT.scale_factor,
                    ),
                )
                self.assertAlmostEqual(float(proper.value), target_value)
                self.assertAlmostEqual(
                    float(proper.to_comoving().value), source_value
                )

    def test_scalar_velocity_requires_position_and_round_trips(self):
        velocity = _quantity(2.0, "vel_supercomoving_code")
        with self.assertRaises(ValueError):
            velocity.to_proper()

        proper = velocity.to_proper(x_comoving_code=4.0)
        self.assertAlmostEqual(float(proper.value), 4.14)
        self.assertAlmostEqual(
            float(proper.to_comoving(x_comoving_code=4.0).value),
            2.0,
        )

    def test_scalar_mixed_representations_are_rejected(self):
        proper = _quantity(1.0, "rho_proper_code")
        comoving = _quantity(1.0, "rho_comoving_code")

        with self.assertRaises(RepresentationMismatchError):
            proper + comoving

    def test_scalar_quantity_converts_to_cgs(self):
        density = _quantity(2.0, "rho_proper_code")
        self.assertTrue(np.isfinite(float(density.to_cgs().value)))

    def test_every_registered_quantity_constructs_and_converts_to_cgs(self):
        for field_name in _FIELD_DEFINITIONS:
            with self.subTest(field_name=field_name):
                array = _array([1.0, 2.0], field_name)

                self.assertIsInstance(array, RadArray)
                self.assertEqual(array.field_spec, field_spec(
                    field_name,
                    CODE_UNITS,
                    hubble_parameter_km_s_Mpc=(
                        CONTEXT.hubble_parameter_km_s_Mpc
                        if field_name == "vel_supercomoving_code"
                        else None
                    ),
                ))
                self.assertTrue(np.all(np.isfinite(array.to_cgs().value)))

    def test_density_round_trip(self):
        source = _array([8.0, 16.0], "rho_comoving_code")
        proper = source.to_proper()

        np.testing.assert_allclose(proper.value, [64.0, 128.0])
        self.assertEqual(proper.field_spec.representation, "proper")
        np.testing.assert_allclose(proper.to_comoving(), source)

    def test_temperature_and_pressure_conversion_use_gamma(self):
        temperature_supercomoving = _array([8.0], "temp_supercomoving_code")
        pressure_supercomoving = _array([64.0], "pre_supercomoving_code")
        temperature = temperature_supercomoving.to_proper()
        pressure = pressure_supercomoving.to_proper()

        np.testing.assert_allclose(temperature.value, [32.0])
        np.testing.assert_allclose(pressure.value, [2048.0])
        np.testing.assert_allclose(
            temperature.to_comoving().value,
            temperature_supercomoving.value,
        )
        np.testing.assert_allclose(
            pressure.to_comoving().value,
            pressure_supercomoving.value,
        )

    def test_velocity_conversion_requires_position(self):
        velocity = _array([2.0], "vel_supercomoving_code")
        with self.assertRaises(ValueError):
            velocity.to_proper()

        proper = velocity.to_proper(x_comoving_code=np.array([4.0]))
        np.testing.assert_allclose(proper.value, [4.14])
        np.testing.assert_allclose(
            proper.to_comoving(x_comoving_code=np.array([4.0])), velocity
        )

    def test_mixed_representations_are_rejected(self):
        proper = _array([1.0], "rho_proper_code")
        comoving = _array([1.0], "rho_comoving_code")

        with self.assertRaises(RepresentationMismatchError):
            proper + comoving

    def test_same_representation_arithmetic_preserves_metadata(self):
        left = _array([2.0], "rho_proper_code")
        right = _array([3.0], "rho_proper_code")

        result = left + right
        self.assertIsInstance(result, RadArray)
        self.assertEqual(result.field_spec, left.field_spec)
        np.testing.assert_allclose(result, [5.0])

    def test_multiplication_creates_derived_metadata(self):
        left = _array([2.0], "rho_proper_code")
        right = _array([3.0], "rho_proper_code")

        result = left * right
        self.assertIsInstance(result, RadArray)
        self.assertEqual(result.field_spec.representation, "proper")
        self.assertEqual(result.field_spec.dimensions, (2, -6, 0, 0, 0))
        np.testing.assert_allclose(result.value, [6.0])
