import unittest

import radhydropy.chemistry as chem


class Testing(unittest.TestCase):
    def test_available_chemistry_modules_include_hhe(self):
        keys = chem.available_chemistry_modules()
        self.assertIn("HHe", keys)
        self.assertIn("H", keys)

    def test_get_chemistry_module_returns_selected_bundle(self):
        selection = chem.get_chemistry_module(key="HHe")

        self.assertEqual(selection.key, "HHe")
        self.assertEqual(selection.species, ("hydrogen", "helium"))

    def test_get_species_modules_returns_species_helpers(self):
        modules = chem.get_species_modules("HHeM")

        self.assertEqual(
            tuple(module.__name__.rsplit(".", 1)[-1] for module in modules),
            ("hydrogen", "helium", "metal"),
        )

    def test_unknown_chemistry_key_raises_clear_error(self):
        with self.assertRaises(ValueError):
            chem.get_chemistry_module(key="unknown")


if __name__ == "__main__":
    unittest.main()
