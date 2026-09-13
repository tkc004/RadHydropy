import unittest

from radhydropy.field_metadata import (
    FIELD_DIMENSION_BASIS_NAME,
    FieldSpec,
    _FIELD_DEFINITIONS,
    hubble_parameter_code,
    field_spec,
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


class FieldSpecTests(unittest.TestCase):
    def test_field_spec_registry_uses_code_units(self):
        spec = field_spec("rho_proper_code", CODE_UNITS)

        self.assertEqual(spec.quantity, "mass_density")
        self.assertEqual(spec.dimensions, (1, -3, 0, 0, 0))
        self.assertEqual(spec.representation, "proper")
        self.assertAlmostEqual(
            spec.code_unit_cgs,
            4.92e31 / (3.08567758e21 ** 3),
        )

    def test_field_spec_registry_supports_cosmological_snapshot_state(self):
        spec = field_spec(
            "rho_comoving_code",
            CODE_UNITS,
            cosmology="lambda_cdm",
            scale_factor=0.5,
            scale_factor_power=3.0,
            conversion_factor=8.0,
        )

        self.assertEqual(spec.representation, "comoving")
        self.assertEqual(spec.coordinate_frame, "comoving")
        self.assertEqual(spec.scale_factor, 0.5)
        self.assertEqual(spec.conversion_factor, 8.0)

    def test_field_spec_registry_rejects_unknown_fields(self):
        with self.assertRaises(ValueError):
            field_spec("unknown_code_field", CODE_UNITS)

    def test_field_spec_registry_derives_angular_momentum_scales(self):
        spec = field_spec("AngularMomentum_code", CODE_UNITS)

        self.assertAlmostEqual(
            spec.code_unit_cgs,
            4.92e31 * 3.08567758e21 * 1.0e5,
        )

    def test_field_spec_registry_covers_every_canonical_field(self):
        expected = {
            "boundary": ("radius", (0, 1, 0, 0, 0), "proper"),
            "boundary_proper_code": ("radius", (0, 1, 0, 0, 0), "proper"),
            "boundary_comoving_code": ("radius", (0, 1, 0, 0, 0), "comoving"),
            "radius_proper_code": ("radius", (0, 1, 0, 0, 0), "proper"),
            "radius_comoving_code": ("radius", (0, 1, 0, 0, 0), "comoving"),
            "rho_proper_code": ("mass_density", (1, -3, 0, 0, 0), "proper"),
            "rho_comoving_code": ("mass_density", (1, -3, 0, 0, 0), "comoving"),
            "vel_proper_code": ("velocity", (0, 0, 1, 0, 0), "proper"),
            "vel_supercomoving_code": ("velocity", (0, 0, 1, 0, 0), "supercomoving"),
            "temp_proper_code": ("temperature", (0, 0, 0, 0, 1), "proper"),
            "temp_supercomoving_code": (
                "temperature", (0, 0, 0, 0, 1), "supercomoving"
            ),
            "pre_proper_code": ("pressure", (1, -3, 2, 0, 0), "proper"),
            "pre_supercomoving_code": (
                "pressure", (1, -3, 2, 0, 0), "supercomoving"
            ),
            "Mass_code": ("mass", (1, 0, 0, 0, 0), "physical"),
            "dark_matter_mass_code": ("mass", (1, 0, 0, 0, 0), "physical"),
            "Energy_code": ("energy", (1, 0, 2, 0, 0), "physical"),
            "InternalEnergy_code": ("energy", (1, 0, 2, 0, 0), "physical"),
            "GravitationalPotentialEnergy_code": (
                "energy", (1, 0, 2, 0, 0), "physical"
            ),
            "AngularMomentum_code": (
                "angular_momentum", (1, 1, 1, 0, 0), "physical"
            ),
            "specific_angular_momentum_code": (
                "specific_angular_momentum", (0, 1, 1, 0, 0), "physical"
            ),
            "specific_angular_momentum_proper_code": (
                "specific_angular_momentum", (0, 1, 1, 0, 0), "proper"
            ),
            "specific_angular_momentum_supercomoving_code": (
                "specific_angular_momentum", (0, 1, 1, 0, 0), "supercomoving"
            ),
            "ngamma_code": ("number_density", (0, -3, 0, 0, 0), "physical"),
            "ngamma_proper_code": ("number_density", (0, -3, 0, 0, 0), "proper"),
            "ngamma_comoving_code": ("number_density", (0, -3, 0, 0, 0), "comoving"),
        }

        self.assertEqual(set(expected), set(_FIELD_DEFINITIONS))
        for field_name, (quantity, dimensions, representation) in expected.items():
            with self.subTest(field_name=field_name):
                spec = field_spec(
                    field_name,
                    CODE_UNITS,
                    hubble_parameter_km_s_Mpc=10.0
                    if field_name == "vel_supercomoving_code"
                    else None,
                )
                self.assertEqual(spec.quantity, quantity)
                self.assertEqual(spec.dimensions, dimensions)
                self.assertEqual(spec.representation, representation)
                self.assertEqual(
                    spec.coordinate_frame,
                    "comoving"
                    if representation in {"supercomoving", "comoving"}
                    else "physical",
                )
                self.assertGreater(spec.code_unit_cgs, 0.0)

    def test_field_spec_registry_uses_field_specific_conversion_relations(self):
        expected_relations = {
            "boundary_comoving_code": "physical = a * stored",
            "rho_comoving_code": "physical = stored / a**3",
            "vel_supercomoving_code": "physical = H*a*x + stored/a",
            "temp_supercomoving_code": "physical = stored / a**(3*(gamma - 1))",
            "pre_supercomoving_code": "physical = stored / a**(3*gamma)",
        }
        for field_name, expected_relation in expected_relations.items():
            with self.subTest(field_name=field_name):
                spec = field_spec(
                    field_name,
                    CODE_UNITS,
                    hubble_parameter_km_s_Mpc=10.0
                    if field_name == "vel_supercomoving_code"
                    else None,
                )
                self.assertEqual(spec.physical_relation, expected_relation)

    def test_supercomoving_velocity_requires_and_stores_hubble_parameter(self):
        with self.assertRaises(ValueError):
            field_spec("vel_supercomoving_code", CODE_UNITS)

        spec = field_spec(
            "vel_supercomoving_code",
            CODE_UNITS,
            hubble_parameter_km_s_Mpc=25.0,
        )
        self.assertEqual(spec.hubble_parameter_km_s_Mpc, 25.0)
        self.assertEqual(spec.to_metadata()["hubble_parameter_km_s_Mpc"], 25.0)

    def test_hubble_parameter_code_is_derived_from_code_units(self):
        self.assertAlmostEqual(
            hubble_parameter_code(CODE_UNITS, 70.0),
            70.0 / 1000.0,
        )

    def test_field_spec_uses_the_five_code_unit_bases(self):
        spec = FieldSpec(
            quantity="mass_density",
            dimensions=(1, -3, 0, 0, 0),
            representation="proper",
            coordinate_frame="physical",
            code_unit_cgs=1.675e-33,
            physical_relation="physical = stored",
        )

        self.assertEqual(spec.dimension_basis, FIELD_DIMENSION_BASIS_NAME)
        self.assertEqual(spec.dimensions, (1, -3, 0, 0, 0))
        self.assertIsInstance(spec.dimensions, tuple)
        self.assertEqual(spec.to_metadata()["code_unit_cgs"], 1.675e-33)
        self.assertEqual(spec.storage_unit, "code")
        self.assertEqual(spec.to_metadata()["storage_unit"], "code")

    def test_field_spec_supports_cgs_thermochemistry_storage(self):
        spec = FieldSpec(
            quantity="number_density",
            dimensions=(0, -3, 0, 0, 0),
            representation="physical",
            coordinate_frame="physical",
            code_unit_cgs=1.0e-24,
            physical_relation="physical = stored",
            storage_unit="cgs",
        )

        self.assertEqual(spec.storage_unit, "cgs")
        self.assertEqual(spec.to_metadata()["storage_unit"], "cgs")
        self.assertEqual(FieldSpec.from_metadata(spec.to_metadata()), spec)
        self.assertEqual(spec.scale_factor, 1.0)
        self.assertEqual(spec.scale_factor_power, 0.0)
        self.assertEqual(spec.conversion_factor, 1.0)

    def test_field_spec_is_immutable_and_round_trips_metadata(self):
        spec = FieldSpec(
            quantity="energy",
            dimensions=(1, 0, 2, 0, 0),
            representation="physical",
            coordinate_frame="physical",
            code_unit_cgs=4.92e41,
            physical_relation="physical = stored",
        )

        with self.assertRaises((AttributeError, TypeError)):
            spec.quantity = "mass"
        self.assertEqual(FieldSpec.from_metadata(spec.to_metadata()), spec)

    def test_field_spec_records_cosmological_conversion_metadata(self):
        spec = FieldSpec(
            quantity="mass_density",
            dimensions=(1, -3, 0, 0, 0),
            representation="comoving",
            coordinate_frame="comoving",
            code_unit_cgs=1.675e-33,
            physical_relation="physical = stored / a**3",
            cosmology="lambda_cdm",
            scale_factor=0.5,
            scale_factor_power=3.0,
            conversion_factor=8.0,
        )

        self.assertEqual(spec.scale_factor, 0.5)
        self.assertEqual(spec.scale_factor_power, 3.0)
        self.assertEqual(spec.conversion_factor, 8.0)
        self.assertIsNone(spec.hubble_parameter_km_s_Mpc)
        self.assertEqual(FieldSpec.from_metadata(spec.to_metadata()), spec)

    def test_proper_field_keeps_conversion_state_for_comoving_conversion(self):
        spec = FieldSpec(
            quantity="mass_density",
            dimensions=(1, -3, 0, 0, 0),
            representation="proper",
            coordinate_frame="physical",
            code_unit_cgs=1.675e-33,
            physical_relation="comoving = stored * a**3",
            cosmology="lambda_cdm",
            scale_factor=0.5,
            scale_factor_power=3.0,
            conversion_factor=0.125,
        )

        self.assertEqual(spec.scale_factor, 0.5)
        self.assertEqual(spec.scale_factor_power, 3.0)
        self.assertEqual(spec.conversion_factor, 0.125)

    def test_field_spec_rejects_the_old_four_component_basis(self):
        with self.assertRaises(ValueError):
            FieldSpec(
                quantity="mass_density",
                dimensions=(1, -3, 0, 0),
                representation="proper",
                coordinate_frame="physical",
                code_unit_cgs=1.0,
                physical_relation="physical = stored",
            )

    def test_field_spec_rejects_nonpositive_conversion_scale(self):
        with self.assertRaises(ValueError):
            FieldSpec(
                quantity="velocity",
                dimensions=(0, 0, 1, 0, 0),
                representation="proper",
                coordinate_frame="physical",
                code_unit_cgs=0.0,
                physical_relation="physical = stored",
            )

    def test_field_spec_rejects_invalid_metadata_values(self):
        common = {
            "quantity": "mass_density",
            "dimensions": (1, -3, 0, 0, 0),
            "representation": "proper",
            "coordinate_frame": "physical",
            "code_unit_cgs": 1.0,
            "physical_relation": "physical = stored",
        }
        invalid_values = (
            ({"dimension_basis": "mass,length,time,current,temperature"}, ValueError),
            ({"dimensions": (1.5, -3, 0, 0, 0)}, ValueError),
            ({"scale_factor": 0.0}, ValueError),
            ({"conversion_factor": -1.0}, ValueError),
            ({"cosmology": ""}, ValueError),
            ({"storage_unit": "si"}, ValueError),
        )
        for overrides, exception in invalid_values:
            with self.subTest(overrides=overrides):
                with self.assertRaises(exception):
                    FieldSpec(**{**common, **overrides})

    def test_field_spec_rejects_incomplete_metadata(self):
        with self.assertRaises(ValueError):
            FieldSpec.from_metadata({"quantity": "density"})

    def test_field_spec_registry_requires_code_units(self):
        with self.assertRaises(TypeError):
            field_spec("rho_proper_code", object())

    def test_field_spec_registry_propagates_all_cosmology_metadata(self):
        spec = field_spec(
            "vel_proper_code",
            CODE_UNITS,
            cosmology="einstein_de_sitter",
            scale_factor=0.25,
            scale_factor_power=1.0,
            conversion_factor=4.0,
        )

        self.assertEqual(
            {
                "cosmology": spec.cosmology,
                "scale_factor": spec.scale_factor,
                "scale_factor_power": spec.scale_factor_power,
                "conversion_factor": spec.conversion_factor,
            },
            {
                "cosmology": "einstein_de_sitter",
                "scale_factor": 0.25,
                "scale_factor_power": 1.0,
                "conversion_factor": 4.0,
            },
        )
