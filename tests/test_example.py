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

    def test_hydrogen_photoionization1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HydrogenPhotoionization1D'
            / 'hydrogen_photoionization1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['target_neutral_fraction'], 0.01)
        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(icparams['nogrid'], 16)
        self.assertEqual(icparams['boxsize'].to_value(unyt.kpc), 1.0)

    def test_hydrogen_photoheating1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HydrogenPhotoheating1D'
            / 'hydrogen_photoheating1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(runparams['source_switch_time'].to_value(unyt.yr), 5.0e7)
        self.assertAlmostEqual(
            runparams['thermal_equilibrium_timescale'].to_value(unyt.yr),
            1.99526231496888e9,
        )
        self.assertEqual(icparams['nHini'].to_value(1.0 / unyt.cm**3), 1.0)

    def test_hydrostatic_equilibrium1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HydrostaticEquilibrium1D'
            / 'hydrostatic_equilibrium1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(runparams['EOStype'], 'isothermal')
        self.assertEqual(runparams['boundcond'], 'Open')
        self.assertEqual(icparams['nogrid'], 256)
        self.assertEqual(icparams['boxsize'].to_value(unyt.pc), 10.0)
        self.assertAlmostEqual(
            icparams['gravity_strength'].to_value(unyt.cm / unyt.s**2),
            1.0e-7,
        )

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

    def test_static_stromgren_sphere1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'StaticStromgrenSphere1D'
            / 'static_stromgren_sphere1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(icparams['number_of_cells'], 256)
        self.assertEqual(icparams['boxsize'].to_value(unyt.kpc), 20.0)
        self.assertEqual(
            icparams['source_photon_rate'].to_value(1.0 / unyt.s),
            5.0e48,
        )
        self.assertAlmostEqual(
            icparams['chemistry_timestep_cfl'],
            0.1,
        )

    def test_static_stromgren_sphere_photoheating1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'StaticStromgrenSpherePhotoheating1D'
            / 'static_stromgren_sphere_photoheating1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(icparams['number_of_cells'], 1024)
        self.assertEqual(icparams['boxsize'].to_value(unyt.kpc), 20.0)
        self.assertEqual(
            icparams['source_photon_rate'].to_value(1.0 / unyt.s),
            5.0e48,
        )
        self.assertEqual(icparams['evolution_timestep'].to_value(unyt.Myr), 1.0)
        self.assertTrue(
            icparams['temperature_reference_filename'].endswith(
                'TTT1Dthin_Stromgren100Myr.txt'
            )
        )

    def test_dynamic_stromgren_sphere_photoheating1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'DynamicStromgrenSpherePhotoheating1D'
            / 'dynamic_stromgren_sphere_photoheating1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(runparams['coordsys'], 'spherical')
        self.assertTrue(Path(runparams['outputtimefilename']).exists())
        self.assertEqual(icparams['number_of_cells'], 1024)
        self.assertEqual(icparams['boxsize'].to_value(unyt.kpc), 20.0)
        self.assertEqual(
            runparams['radiative_transfer_source_photon_rate'].to_value(1.0 / unyt.s),
            5.0e48,
        )
        self.assertTrue(
            icparams['density_reference_filename'].endswith(
                'Stromgren3D_rhd_n_r_zeusmp_t200.csv'
            )
        )
        self.assertEqual(
            runparams['hydrogen_source_dtmin'].to_value(unyt.Myr),
            1.0e-3,
        )

    def test_early_hii_region_expansion1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HIIRegionExpansion1D'
            / 'early_hii_region_expansion1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(icparams['number_of_cells'], 512)
        self.assertEqual(icparams['boxsize'].to_value(unyt.pc), 2.0)
        self.assertEqual(icparams['final_time'].to_value(unyt.Myr), 0.14)
        self.assertTrue(Path(runparams['outputtimefilename']).exists())
        self.assertEqual(len(icparams['output_snapshots']), 8)
        self.assertEqual(
            icparams['output_snapshots'][1]['label'],
            '0p005',
        )

    def test_late_hii_region_expansion1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HIIRegionExpansion1D'
            / 'late_hii_region_expansion1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(icparams['number_of_cells'], 512)
        self.assertEqual(icparams['boxsize'].to_value(unyt.pc), 7.0)
        self.assertEqual(icparams['final_time'].to_value(unyt.Myr), 3.0)
        self.assertTrue(Path(runparams['outputtimefilename']).exists())
        self.assertTrue(icparams['show_stagnation_radius'])
        self.assertEqual(
            icparams['output_snapshots'][-1]['label'],
            '3p00',
        )


if __name__ == '__main__':
    unittest.main()
