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

    def test_radiative_transfer_sph1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'RadiativeTransferSph1D'
            / 'radiative_transfer_sph1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(runparams['coordsys'], 'spherical')
        self.assertEqual(icparams['number_of_cells'], 256)
        self.assertEqual(icparams['boxsize'].to_value(unyt.pc), 1.0)
        self.assertEqual(icparams['source_photon_rate'].to_value(1.0 / unyt.s), 1.0e49)


if __name__ == '__main__':
    unittest.main()
