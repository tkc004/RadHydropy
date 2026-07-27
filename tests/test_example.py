import unittest
from pathlib import Path

import unyt

from radhydropy.example_config import load_example_parameters


class Testing(unittest.TestCase):
    def test_inflow1d_uses_explicit_output_time_schedule(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'Inflow1D'
            / 'Inflow1d.yaml'
        )
        runparams, _ = load_example_parameters(config_filename)

        self.assertNotIn('outdeltatime', runparams)
        self.assertIn('outputtimefilename', runparams)
        self.assertTrue(runparams['outputtimefilename'].endswith('output_times.txt'))

        outputtimepath = Path(runparams['outputtimefilename'])
        self.assertTrue(outputtimepath.exists())

        with outputtimepath.open() as handle:
            lines = [line.strip() for line in handle if line.strip()]

        self.assertEqual(lines[0], 's')
        self.assertEqual(len(lines) - 1, 20)
        self.assertEqual(float(lines[1]), 0.1)
        self.assertEqual(float(lines[-1]), 2.0)


if __name__ == '__main__':
    unittest.main()
