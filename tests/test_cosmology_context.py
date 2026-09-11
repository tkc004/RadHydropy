import unittest
from dataclasses import FrozenInstanceError

from radhydropy.cosmology_context import CosmologyContext


class CosmologyContextTests(unittest.TestCase):
    def test_gamma_is_required(self):
        with self.assertRaises(TypeError):
            CosmologyContext()

    def test_context_stores_snapshot_state(self):
        context = CosmologyContext(
            gamma=5.0 / 3.0,
            cosmology="lambda_cdm",
            scale_factor=0.5,
            hubble_parameter_km_s_Mpc=70.0,
        )

        self.assertEqual(context.gamma, 5.0 / 3.0)
        self.assertEqual(context.cosmology, "lambda_cdm")
        self.assertEqual(context.scale_factor, 0.5)
        self.assertEqual(context.hubble_parameter_km_s_Mpc, 70.0)

    def test_context_is_immutable(self):
        context = CosmologyContext(gamma=5.0 / 3.0)

        with self.assertRaises(FrozenInstanceError):
            context.gamma = 1.4

    def test_context_rejects_invalid_values(self):
        invalid_contexts = (
            {"gamma": 1.0},
            {"gamma": float("nan")},
            {"gamma": 5.0 / 3.0, "scale_factor": 0.0},
            {
                "gamma": 5.0 / 3.0,
                "hubble_parameter_km_s_Mpc": -1.0,
            },
        )

        for values in invalid_contexts:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    CosmologyContext(**values)
