import unittest
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import tempfile
from unittest import mock

import h5py
import numpy as np
import unyt

from radhydropy.example_config import load_example_parameters
import radhydropy.io as rio
from radhydropy.rsim import Rsim
import radhydropy.radiative_transfer as rrt
import radhydropy.thermo_chemistry as rtc

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / 'example'
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
import example_utils as example_utils
HII_EXAMPLE_ROOT = EXAMPLE_ROOT / 'HIIRegionExpansion1D'
if str(HII_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(HII_EXAMPLE_ROOT))

HII_TOOLS_PATH = HII_EXAMPLE_ROOT / 'tools.py'
HII_TOOLS_SPEC = importlib.util.spec_from_file_location(
    'hii_region_expansion1d_tools_for_tests',
    HII_TOOLS_PATH,
)
hii_tools = importlib.util.module_from_spec(HII_TOOLS_SPEC)
assert HII_TOOLS_SPEC.loader is not None
HII_TOOLS_SPEC.loader.exec_module(hii_tools)

LATE_HII_PATH = HII_EXAMPLE_ROOT / 'late_hii_region_expansion1d.py'
LATE_HII_SPEC = importlib.util.spec_from_file_location(
    'late_hii_region_expansion1d_for_tests',
    LATE_HII_PATH,
)
late_hii = importlib.util.module_from_spec(LATE_HII_SPEC)
assert LATE_HII_SPEC.loader is not None
LATE_HII_SPEC.loader.exec_module(late_hii)

STELLAR_WIND_TOOLS_PATH = (
    Path(__file__).resolve().parents[1]
    / 'example'
    / 'StellarWindBubble1D'
    / 'tools.py'
)
WEAVER_ANALYTIC_PATH = (
    Path(__file__).resolve().parents[1]
    / 'example'
    / 'StellarWindBubble1D'
    / 'weaver_analytic.py'
)
WEAVER_ANALYTIC_SPEC = importlib.util.spec_from_file_location(
    'weaver_analytic',
    WEAVER_ANALYTIC_PATH,
)
weaver_analytic = importlib.util.module_from_spec(WEAVER_ANALYTIC_SPEC)
assert WEAVER_ANALYTIC_SPEC.loader is not None
sys.modules['weaver_analytic'] = weaver_analytic
WEAVER_ANALYTIC_SPEC.loader.exec_module(weaver_analytic)
STELLAR_WIND_TOOLS_SPEC = importlib.util.spec_from_file_location(
    'stellar_wind_bubble1d_tools_for_tests',
    STELLAR_WIND_TOOLS_PATH,
)
stellar_wind_tools = importlib.util.module_from_spec(STELLAR_WIND_TOOLS_SPEC)
assert STELLAR_WIND_TOOLS_SPEC.loader is not None
STELLAR_WIND_TOOLS_SPEC.loader.exec_module(stellar_wind_tools)

STATIC_STROMGREN_PHOTONHEATING_TOOLS_PATH = (
    Path(__file__).resolve().parents[1]
    / 'example'
    / 'StaticStromgrenSpherePhotoheating1D'
    / 'tools.py'
)
STATIC_STROMGREN_PHOTONHEATING_TOOLS_SPEC = importlib.util.spec_from_file_location(
    'static_stromgren_sphere_photoheating1d_tools_for_tests',
    STATIC_STROMGREN_PHOTONHEATING_TOOLS_PATH,
)
static_stromgren_photoheating_tools = importlib.util.module_from_spec(
    STATIC_STROMGREN_PHOTONHEATING_TOOLS_SPEC
)
assert STATIC_STROMGREN_PHOTONHEATING_TOOLS_SPEC.loader is not None
STATIC_STROMGREN_PHOTONHEATING_TOOLS_SPEC.loader.exec_module(
    static_stromgren_photoheating_tools
)

STATIC_STROMGREN_TOOLS_PATH = (
    Path(__file__).resolve().parents[1]
    / 'example'
    / 'StaticStromgrenSphere1D'
    / 'tools.py'
)
STATIC_STROMGREN_TOOLS_SPEC = importlib.util.spec_from_file_location(
    'static_stromgren_sphere1d_tools_for_tests',
    STATIC_STROMGREN_TOOLS_PATH,
)
static_stromgren_tools = importlib.util.module_from_spec(STATIC_STROMGREN_TOOLS_SPEC)
assert STATIC_STROMGREN_TOOLS_SPEC.loader is not None
STATIC_STROMGREN_TOOLS_SPEC.loader.exec_module(static_stromgren_tools)

HYDROGEN_PHOTOIONIZATION_EXAMPLE_ROOT = EXAMPLE_ROOT / 'HydrogenPhotoionization1D'
if str(HYDROGEN_PHOTOIONIZATION_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(HYDROGEN_PHOTOIONIZATION_EXAMPLE_ROOT))

HYDROGEN_PHOTOIONIZATION_ANALYTIC_PATH = (
    HYDROGEN_PHOTOIONIZATION_EXAMPLE_ROOT / 'hydrogen_photoionization_analytic.py'
)
HYDROGEN_PHOTOIONIZATION_ANALYTIC_SPEC = importlib.util.spec_from_file_location(
    'hydrogen_photoionization_analytic_for_tests',
    HYDROGEN_PHOTOIONIZATION_ANALYTIC_PATH,
)
hydrogen_photoionization_analytic = importlib.util.module_from_spec(
    HYDROGEN_PHOTOIONIZATION_ANALYTIC_SPEC
)
assert HYDROGEN_PHOTOIONIZATION_ANALYTIC_SPEC.loader is not None
HYDROGEN_PHOTOIONIZATION_ANALYTIC_SPEC.loader.exec_module(
    hydrogen_photoionization_analytic
)

ADVECTION_SPH_EXAMPLE_ROOT = EXAMPLE_ROOT / 'AdvectionSph1D'
if str(ADVECTION_SPH_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(ADVECTION_SPH_EXAMPLE_ROOT))

ADVECTION_SPH_ANALYTIC_PATH = (
    ADVECTION_SPH_EXAMPLE_ROOT / 'advection_sph_analytic.py'
)
ADVECTION_SPH_ANALYTIC_SPEC = importlib.util.spec_from_file_location(
    'advection_sph_analytic_for_tests',
    ADVECTION_SPH_ANALYTIC_PATH,
)
advection_sph_analytic = importlib.util.module_from_spec(
    ADVECTION_SPH_ANALYTIC_SPEC
)
assert ADVECTION_SPH_ANALYTIC_SPEC.loader is not None
ADVECTION_SPH_ANALYTIC_SPEC.loader.exec_module(advection_sph_analytic)


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

    def test_hydrogen_photoionization1d_analytic_neutral_fraction_uses_units(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HydrogenPhotoionization1D'
            / 'hydrogen_photoionization1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        neutral_fraction = hydrogen_photoionization_analytic.neutral_fraction(
            0.0,
            icparams['xHIini'],
            icparams['tempini'],
            icparams['nHini'],
            icparams['ngammaini'],
            runparams['hydrogen_sigma_gamma'],
        )

        self.assertAlmostEqual(float(neutral_fraction), 1.0, places=12)

    def test_advection_sph1d_analytic_uses_spherical_dilution(self):
        radius = np.array([0.5, 1.5, 3.0], dtype=float)
        density = advection_sph_analytic.top_hat_density_profile(
            radius,
            time=1.0,
            velocity=1.0,
            boxsize=4.0,
            density_high=10.0,
            density_low_factor=0.1,
            left_fraction=0.25,
            right_fraction=0.75,
        )

        expected = np.array([
            0.0,
            10.0 * 0.1 * (0.5 / 1.5) ** 2,
            10.0 * (2.0 / 3.0) ** 2,
        ])
        np.testing.assert_allclose(density, expected, rtol=1e-12, atol=1e-12)

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
        self.assertTrue(
            runparams['outputtimefilename'].endswith(
                'hydrostatic_equilibrium1d_output_times.txt'
            )
        )
        self.assertEqual(runparams['timesim'].to_value(unyt.Myr), 0.0001)
        self.assertEqual(runparams['EOStype'], 'isothermal')
        self.assertEqual(runparams['boundcond'], 'Reflecting')
        self.assertEqual(icparams['nogrid'], 256)
        self.assertEqual(icparams['boxsize'].to_value(unyt.pc), 10.0)
        self.assertAlmostEqual(
            icparams['gravity_strength'].to_value(unyt.cm / unyt.s**2),
            1.0e-7,
        )

    def test_spherical_point_mass_hydrostatic_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HydrostaticEquilibriumSphericalPointMass1D'
            / 'hydrostatic_equilibrium_spherical_point_mass1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(runparams['coordsys'], 'spherical')
        self.assertEqual(runparams['boundcond'], 'Reflecting')
        self.assertTrue(
            runparams['outputtimefilename'].endswith(
                'hydrostatic_equilibrium_spherical_point_mass1d_output_times.txt'
            )
        )
        self.assertAlmostEqual(
            runparams['timesim'].to_value(unyt.Myr),
            3.168808781402895e-22,
        )
        self.assertEqual(icparams['nogrid'], 256)
        self.assertEqual(icparams['coordsys'], 'spherical')
        self.assertEqual(icparams['rmin'].to_value(unyt.pc), 2.0)
        self.assertEqual(icparams['rmax'].to_value(unyt.pc), 20.0)
        self.assertEqual(icparams['point_mass'].to_value(unyt.g), 1.0e38)

    def test_spherical_ballistic_infall_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'BallisticInfallSphericalPointMass1D'
            / 'ballistic_infall_spherical_point_mass1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(runparams['coordsys'], 'spherical')
        self.assertEqual(runparams['boundcond'], 'Reflecting')
        self.assertTrue(
            runparams['outputtimefilename'].endswith(
                'ballistic_infall_spherical_point_mass1d_output_times.txt'
            )
        )
        self.assertEqual(runparams['timesim'].to_value(unyt.Myr), 0.0001)
        self.assertEqual(icparams['nogrid'], 256)
        self.assertEqual(icparams['coordsys'], 'spherical')
        self.assertEqual(icparams['rmin'].to_value(unyt.pc), 2.0)
        self.assertEqual(icparams['rmax'].to_value(unyt.pc), 20.0)
        self.assertEqual(runparams['EOStype'], 'polytropic')
        self.assertEqual(runparams['gamma'], 1.4)
        self.assertEqual(icparams['tempini'].to_value(unyt.K), 1.0)
        self.assertEqual(icparams['point_mass'].to_value(unyt.g), 1.0e38)

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
        self.assertEqual(runparams['boundcond'], 'OpenSph')
        self.assertEqual(runparams['noghost'], 2)
        self.assertEqual(runparams['hydrogen_chemistry'], False)
        self.assertEqual(runparams['radiative_transfer'], True)
        self.assertEqual(
            runparams['radiative_transfer_source_photon_rate'].to_value(1.0 / unyt.s),
            1.0e49,
        )
        self.assertIn('CodeUnits', runparams)
        self.assertEqual(icparams['number_of_cells'], 256)
        self.assertEqual(icparams['boxsize'].to_value(unyt.pc), 1.0)

    def test_radiative_transfer_sph1d_c2ray_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'RadiativeTransferSph1D'
            / 'radiative_transfer_sph1d_c2ray.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['radiative_transfer_temporal_scheme'], 'c2ray')
        self.assertTrue(runparams['hydrogen_chemistry'])
        self.assertEqual(runparams['outfileprefix'], 'Output_C2Ray')
        self.assertEqual(
            runparams['radiative_transfer_c2ray_nonconvergence'],
            'raise',
        )
        self.assertEqual(icparams['number_of_cells'], 256)

    def test_multifrequency_radiative_transfer_sph1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'MultiFrequencyRadiativeTransferSph1D'
            / 'multifrequency_radiative_transfer_sph1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['radiative_transfer'], True)
        self.assertFalse(runparams['hydrogen_initial_collisional_equilibrium'])
        self.assertEqual(runparams['hydrogen_xHI_initial'], 0.9988)
        self.assertEqual(
            runparams['radiation_spectrum_total_photon_rate'].to_value(1.0 / unyt.s),
            5.0e48,
        )
        spectrum_filename = (
            config_filename.parent / runparams['radiation_spectrum_filename']
        )
        with h5py.File(spectrum_filename, 'r') as spectrum:
            group = spectrum['RadiationSpectrum']
            self.assertEqual(group.attrs['number_of_radiation_groups'], 5)
            self.assertEqual(group.attrs['number_of_group_edges'], 6)
            self.assertEqual(group.attrs['stellar_spectrum_type'], 1)
            self.assertEqual(
                group.attrs['stellar_spectrum_blackbody_temperature_K'],
                1.0e5,
            )
            self.assertEqual(
                list(group['group_edges_eV']),
                [13.6, 24.6, 35.5, 54.4, 75.0, 50000.0],
            )
            self.assertEqual(len(group['ionizing_photon_energy_erg']), 5)
        self.assertEqual(icparams['number_of_cells'], 512)

    def test_multifrequency_radiative_transfer_sph1d_c2ray_uses_distinct_outputs(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'MultiFrequencyRadiativeTransferSph1D'
            / 'multifrequency_radiative_transfer_sph1d_c2ray.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['radiative_transfer_temporal_scheme'], 'c2ray')
        self.assertEqual(runparams['outfileprefix'], 'Output_C2Ray')
        self.assertEqual(
            Path(runparams['ICfilename']).name,
            'InitialCondition_C2Ray.hdf5',
        )
        self.assertEqual(
            runparams['figure_filename'],
            'MultiFrequencyRadiativeTransferSph1D_C2Ray.jpg',
        )
        self.assertEqual(icparams['number_of_cells'], 512)

    def test_stellar_wind_bubble1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'StellarWindBubble1D'
            / 'stellar_wind_bubble1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(runparams['coordsys'], 'spherical')
        self.assertEqual(runparams['boundcond'], 'OutflowSph')
        self.assertEqual(runparams['EOStype'], 'polytropic')
        self.assertTrue(
            runparams['outputtimefilename'].endswith(
                'stellar_wind_bubble1d_output_times.txt'
            )
        )
        self.assertEqual(
            runparams['shell_edge_density_threshold_factor'],
            1.0,
        )
        self.assertEqual(runparams['order'], 1)
        self.assertEqual(icparams['nogrid'], 1024)
        self.assertEqual(icparams['boxsize'].to_value(unyt.pc), 25.0)
        self.assertEqual(icparams['rinj'].to_value(unyt.pc), 0.05)
        self.assertEqual(icparams['rhoini'].to_value(unyt.g / unyt.cm**3), 1.0e-24)
        self.assertEqual(runparams['vel_outflow'].to_value(unyt.km / unyt.s), 1000.0)
        self.assertEqual(runparams['rho_outflow'].to_value(unyt.g / unyt.cm**3), 1.0e-22)
        self.assertEqual(icparams['time'].to_value(unyt.Myr), 0.0)
        self.assertEqual(runparams['timesim'].to_value(unyt.Myr), 0.1)

    def test_stellar_wind_shell_edge_radius_uses_inner_shell(self):
        boundary = unyt.unyt_array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            unyt.pc,
        )
        density = unyt.unyt_array(
            [2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            unyt.g / unyt.cm**3,
        ) * 1.0e-24
        rout = SimpleNamespace(
            mesh=SimpleNamespace(boundary=boundary),
            fluid=SimpleNamespace(rho=density),
        )

        radius = stellar_wind_tools.shell_inner_edge_radius(
            rout,
            unyt.unyt_quantity(1.0e-24, unyt.g / unyt.cm**3),
            threshold_factor=1.0,
        )

        self.assertIsNotNone(radius)
        self.assertAlmostEqual(radius.to_value(unyt.pc), 5.0, places=2)

    def test_stellar_wind_shell_diagnostics_include_velocity_and_pressure(self):
        def make_snapshot(time_myr, shell_start_index):
            boundary = unyt.unyt_array(
                [0.1 * idx for idx in range(51)],
                unyt.pc,
            )
            rho = [0.5e-24] * shell_start_index + [2.0e-24] * (
                len(boundary) - 1 - shell_start_index
            )
            temp = [2.0e6] * shell_start_index + [1.0e4] * (
                len(boundary) - 1 - shell_start_index
            )
            return SimpleNamespace(
                par=SimpleNamespace(time=unyt.unyt_quantity(time_myr, unyt.Myr)),
                mesh=SimpleNamespace(boundary=boundary),
                fluid=SimpleNamespace(
                    rho=unyt.unyt_array(rho, unyt.g / unyt.cm**3),
                    temp=unyt.unyt_array(temp, unyt.K),
                    mu=unyt.unyt_array([0.62] * (len(boundary) - 1)),
                ),
            )

        snapshots = [make_snapshot(1.0, 25), make_snapshot(2.0, 26)]
        runparams = {
            'shell_edge_density_threshold_factor': 1.0,
            'rho_outflow': unyt.unyt_quantity(1.0e-22, unyt.g / unyt.cm**3),
            'vel_outflow': unyt.unyt_quantity(1000.0, unyt.km / unyt.s),
        }
        icparams = {
            'rhoini': unyt.unyt_quantity(1.0e-24, unyt.g / unyt.cm**3),
            'rinj': unyt.unyt_quantity(0.05, unyt.pc),
        }

        diagnostics = stellar_wind_tools.collect_shell_diagnostics(
            snapshots,
            icparams,
            runparams,
        )

        self.assertIsNotNone(diagnostics)
        self.assertEqual(diagnostics['times'].to_value(unyt.Myr).tolist(), [1.0, 2.0])
        self.assertAlmostEqual(
            diagnostics['radii'][0].to_value(unyt.pc),
            2.4833333333333334,
            places=12,
        )
        self.assertAlmostEqual(
            diagnostics['radii'][1].to_value(unyt.pc),
            2.5833333333333335,
            places=12,
        )
        expected_velocity = (0.1 * unyt.pc / unyt.Myr).to_value(unyt.km / unyt.s)
        self.assertAlmostEqual(
            diagnostics['velocities'][0].to_value(unyt.km / unyt.s),
            expected_velocity,
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics['velocities'][1].to_value(unyt.km / unyt.s),
            expected_velocity,
            places=6,
        )
        expected_pressure = (
            0.5e-24 * unyt.g / unyt.cm**3
            / (0.62 * unyt.mp)
            * unyt.kb
            * 2.0e6 * unyt.K
        ).to_value(unyt.dyn / unyt.cm**2)
        self.assertAlmostEqual(
            diagnostics['pressures'][0].to_value(unyt.dyn / unyt.cm**2),
            expected_pressure,
            places=12,
        )
        self.assertAlmostEqual(
            diagnostics['pressures'][1].to_value(unyt.dyn / unyt.cm**2),
            expected_pressure,
            places=12,
        )

    def test_static_stromgren_sphere1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'StaticStromgrenSphere1D'
            / 'static_stromgren_sphere1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(runparams['coordsys'], 'spherical')
        self.assertEqual(runparams['boundcond'], 'OpenSph')
        self.assertEqual(runparams['noghost'], 2)
        self.assertEqual(runparams['EOStype'], 'polytropic')
        self.assertEqual(runparams['gamma'], 1.6666666666666667)
        self.assertEqual(runparams['timesim'].to_value(unyt.Myr), 500.0)
        self.assertEqual(runparams['final_time'].to_value(unyt.Myr), 500.0)
        self.assertEqual(runparams['chemistry_timestep'].to_value(unyt.Myr), 5.0)
        self.assertEqual(
            runparams['radiative_transfer_source_photon_rate'].to_value(1.0 / unyt.s),
            5.0e48,
        )
        self.assertEqual(
            runparams['hydrogen_alpha_B'].to_value(unyt.cm**3 / unyt.s),
            2.59e-13,
        )
        self.assertEqual(
            runparams['hydrogen_sigma_gamma'].to_value(unyt.cm**2),
            8.13e-18,
        )
        self.assertEqual(
            runparams['hydrogen_source_dtmin'].to_value(unyt.Myr),
            1.0e-3,
        )
        self.assertEqual(runparams['plot_radius_max'].to_value(unyt.kpc), 7.5)
        self.assertEqual(icparams['number_of_cells'], 256)
        self.assertEqual(icparams['boxsize'].to_value(unyt.kpc), 20.0)
        self.assertEqual(icparams['hydrogen_number_density'].to_value(1.0 / unyt.cm**3), 1.0e-3)

    def test_static_stromgren_c2ray_comparison_uses_256_cells_and_requested_steps(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'StaticStromgrenC2RayComparison'
            / 'static_stromgren_c2ray_comparison.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(icparams['number_of_cells'], 256)
        self.assertEqual(runparams['comparison_c2ray_steps'], 100)
        self.assertEqual(
            runparams['comparison_instantaneous_steps'],
            [100, 1000, 10000, 100000],
        )
        self.assertEqual(runparams['radiative_transfer_temporal_scheme'], 'instantaneous')

    def test_static_stromgren_sphere_photoheating1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'StaticStromgrenSpherePhotoheating1D'
            / 'static_stromgren_sphere_photoheating1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(runparams['coordsys'], 'spherical')
        self.assertEqual(runparams['boundcond'], 'OpenSph')
        self.assertEqual(runparams['noghost'], 2)
        self.assertEqual(runparams['EOStype'], 'polytropic')
        self.assertEqual(runparams['gamma'], 1.6666666666666667)
        self.assertEqual(runparams['timesim'].to_value(unyt.Myr), 500.0)
        self.assertEqual(runparams['final_time'].to_value(unyt.Myr), 500.0)
        self.assertEqual(runparams['evolution_timestep'].to_value(unyt.Myr), 1.0)
        self.assertEqual(runparams['reference_time'].to_value(unyt.Myr), 100.0)
        self.assertEqual(runparams['source_photon_rate'].to_value(1.0 / unyt.s), 5.0e48)
        self.assertIsNone(runparams['alpha_B_coefficient'])
        self.assertIsNone(runparams['hydrogen_beta'])
        self.assertTrue(runparams['hydrogen_collisional_ionization'])
        self.assertEqual(
            runparams['temperature_reference_filename'].endswith(
                'TTT1Dthin_Stromgren100Myr.txt'
            ),
            True,
        )
        self.assertEqual(
            runparams['neutral_fraction_reference_filename'].endswith(
                'xTT1Dthin_Stromgren100Myr.txt'
            ),
            True,
        )
        self.assertEqual(icparams['number_of_cells'], 1024)
        self.assertEqual(icparams['boxsize'].to_value(unyt.kpc), 20.0)
        self.assertEqual(icparams['hydrogen_number_density'].to_value(1.0 / unyt.cm**3), 1.0e-3)
        self.assertEqual(icparams['initial_temperature'].to_value(unyt.K), 100.0)
        self.assertEqual(icparams['time'].to_value(unyt.Myr), 0.0)
        self.assertEqual(icparams['analytic_inner_radius'].to_value(unyt.kpc), 0.1)

    def test_static_stromgren_sphere_photoheating1d_c2ray_uses_distinct_outputs(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'StaticStromgrenSpherePhotoheating1D'
            / 'static_stromgren_sphere_photoheating1d_c2ray.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['radiative_transfer_temporal_scheme'], 'c2ray')
        self.assertEqual(runparams['ICfilename'].split('/')[-1], 'InitialCondition_C2Ray.hdf5')
        self.assertEqual(runparams['outfileprefix'], 'Output_C2Ray')
        self.assertEqual(runparams['radiative_transfer_c2ray_nonconvergence'], 'raise')
        self.assertEqual(icparams['number_of_cells'], 1024)

    def test_static_stromgren_sphere_photoheating1d_static_evolution_runs(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'StaticStromgrenSpherePhotoheating1D'
            / 'static_stromgren_sphere_photoheating1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)
        config = {**runparams, **icparams}

        par, mesh, fluid, solver = static_stromgren_photoheating_tools.build_static_problem(
            config
        )
        sim = Rsim.FromComponents(par, mesh, fluid, solver)

        state = {
            'xHI': np.array([1.0, 0.0, 1.0], dtype=float),
            'nH_cm3': np.array([1.0, 1.0, 1.0], dtype=float),
            'volume_cm3': np.array([1.0, 1.0, 1.0], dtype=float),
            'temperature_K': np.array([100.0, 200.0, 150.0], dtype=float),
            'radius_kpc': np.array([0.1, 0.2, 0.3], dtype=float),
        }
        refresh_calls = []

        with mock.patch.object(rtc, 'source_state', return_value=state), \
            mock.patch.object(rrt, 'trace_photon_density', return_value=np.array([0.0, 0.0, 0.0])), \
            mock.patch.object(rtc, 'get_timestep', return_value=(1.0, None)), \
            mock.patch.object(sim, '_advance_source_thermochemistry_state', return_value=0.0), \
            mock.patch.object(sim, '_finish_static_thermochemistry', return_value=None), \
            mock.patch.object(sim, '_refresh_static_photon_density', side_effect=lambda *args: refresh_calls.append(args[-1]) or (None, 0)):
            history = sim.EvolveStaticThermochemistry(
                1.0 * unyt.s,
                1.0 * unyt.s,
                include_thermal_history=True,
                reference_time=None,
            )

        self.assertEqual(refresh_calls, [1])
        self.assertEqual(history['evolution_steps'], 1)
        self.assertIn('mean_ionized_temp_K', history)

    def test_static_reference_snapshot_stores_myr_not_seconds(self):
        sim = Rsim.__new__(Rsim)
        state = {
            'radius_kpc': np.array([1.0]),
            'xHI': np.array([0.5]),
            'temperature_K': np.array([1.0e4]),
        }
        snapshot = sim._snapshot_static_state(
            state,
            2.0 * (1.0 * unyt.Myr).to_value(unyt.s),
        )

        self.assertAlmostEqual(snapshot['time_Myr'], 2.0)

    def test_static_stromgren_sphere_photon_budget_uses_physical_units(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'StaticStromgrenSphere1D'
            / 'static_stromgren_sphere1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)
        config = {**runparams, **icparams}

        par, mesh, fluid, solver = static_stromgren_tools.build_static_problem(config)
        sim = Rsim.FromComponents(par, mesh, fluid, solver)
        sim.ConvertParametersToCodeUnits()

        state = {
            'xHI': np.array([1.0], dtype=float),
            'nH_cm3': np.array([1.0], dtype=float),
            'volume_cm3': np.array([1.0], dtype=float),
            'temperature_K': np.array([1.0], dtype=float),
            'radius_kpc': np.array([1.0], dtype=float),
        }

        with mock.patch.object(rtc, 'source_state', return_value=state), \
            mock.patch.object(rrt, 'trace_photon_density', return_value=np.array([0.0])), \
            mock.patch.object(rtc, 'get_timestep', return_value=(1.0, None)), \
            mock.patch.object(sim, '_advance_source_thermochemistry_state', return_value=0.0), \
            mock.patch.object(sim, '_finish_static_thermochemistry', return_value=None), \
            mock.patch.object(sim, '_refresh_static_photon_density', return_value=(None, 0)):
            history = sim.EvolveStaticThermochemistry(
                1.0 * unyt.s,
                1.0 * unyt.s,
                include_thermal_history=False,
                reference_time=None,
            )

        self.assertAlmostEqual(history['injected_photons'][-1], 5.0e48)
        self.assertLess(history['time_Myr'][-1], 1.0e-12)

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
            0.0,
        )

    def test_dynamic_stromgren_sphere_photoheating1d_c2ray_uses_distinct_outputs(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'DynamicStromgrenSpherePhotoheating1D'
            / 'dynamic_stromgren_sphere_photoheating1d_c2ray.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['radiative_transfer_temporal_scheme'], 'c2ray')
        self.assertEqual(runparams['outfileprefix'], 'Output_C2Ray')
        self.assertEqual(runparams['ICfilename'].split('/')[-1], 'InitialCondition_C2Ray.hdf5')
        self.assertEqual(runparams['radiative_transfer_c2ray_nonconvergence'], 'raise')
        self.assertEqual(icparams['number_of_cells'], 1024)

    def test_dynamic_stromgren_sphere_high_density_uses_requested_parameters(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'DynamicStromgrenSpherePhotoheating20pc1D'
            / 'dynamic_stromgren_sphere_photoheating20pc1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['timesim'].to_value(unyt.Myr), 1.0)
        self.assertEqual(
            runparams['radiative_transfer_source_photon_rate'].to_value(1.0 / unyt.s),
            1.0e49,
        )
        self.assertEqual(icparams['boxsize'].to_value(unyt.pc), 20.0)
        self.assertEqual(icparams['number_of_cells'], 512)
        self.assertEqual(
            icparams['hydrogen_number_density'].to_value(1.0 / unyt.cm**3),
            100.0,
        )

    def test_powerlaw_hii_w1_c2ray_uses_distinct_outputs(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'PowerLawHIIRegion1D'
            / 'power_law_hii_region_radhydropy_c2ray.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['radiative_transfer_temporal_scheme'], 'c2ray')
        self.assertEqual(runparams['outfileprefix'], 'Output_PowerLawHIIRegion1D_C2Ray')
        self.assertEqual(
            runparams['ICfilename'].split('/')[-1],
            'InitialCondition_PowerLawHIIRegion1D_C2Ray.hdf5',
        )
        self.assertEqual(runparams['front_plot_filename'], 'PowerLawHIIRegion1D_C2Ray_RadHydroVsAnalytic.jpg')
        self.assertEqual(icparams['density_power_law_exponent'], 1.0)
        self.assertEqual(icparams['number_of_cells'], 256)

    def test_powerlaw_hii_w1p4_c2ray_uses_distinct_outputs(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'PowerLawHIIRegion1D'
            / 'power_law_hii_region_w1p4_c2ray.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['radiative_transfer_temporal_scheme'], 'c2ray')
        self.assertEqual(runparams['outfileprefix'], 'Output_PowerLawHIIRegion1D_w1p4_C2Ray')
        self.assertEqual(icparams['density_power_law_exponent'], 1.4)
        self.assertEqual(icparams['number_of_cells'], 1024)

    def test_powerlaw_hii_w1p5_c2ray_uses_distinct_outputs(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'PowerLawHIIRegion1D'
            / 'power_law_hii_region_w1p5_c2ray.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['radiative_transfer_temporal_scheme'], 'c2ray')
        self.assertEqual(runparams['outfileprefix'], 'Output_PowerLawHIIRegion1D_w1p5_C2Ray')
        self.assertEqual(icparams['density_power_law_exponent'], 1.5)
        self.assertEqual(icparams['number_of_cells'], 1024)

    def test_dynamic_stromgren_sphere_stellar_wind_uses_requested_wind(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'DynamicStromgrenSpherePhotoheating20pcStellarWind1D'
            / 'dynamic_stromgren_sphere_photoheating20pc_stellar_wind1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(icparams['number_of_cells'], 512)
        self.assertEqual(icparams['boxsize'].to_value(unyt.pc), 20.0)
        self.assertEqual(
            runparams['radiative_transfer_source_photon_rate'].to_value(1.0 / unyt.s),
            1.0e49,
        )
        self.assertEqual(
            runparams['wind_mass_loss_rate'].to_value(unyt.Msun / unyt.yr),
            1.0e-6,
        )
        self.assertEqual(
            runparams['wind_velocity'].to_value(unyt.km / unyt.s),
            1000.0,
        )

    def test_dynamic_stromgren_front_uses_first_xhi_half_crossing(self):
        tools_path = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'DynamicStromgrenSpherePhotoheating1D'
            / 'tools.py'
        )
        spec = importlib.util.spec_from_file_location(
            'dynamic_stromgren_tools_front_test',
            tools_path,
        )
        tools = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(tools)

        par = SimpleNamespace(noghost=0, nogrid=5, CodeUnits=None)
        mesh = SimpleNamespace(
            coordinate=np.arange(1.0, 6.0) * unyt.kpc,
        )
        fluid = SimpleNamespace(
            xHI=np.array([0.1, 0.2, 0.8, 0.2, 0.9]),
        )

        front = tools.ionization_front_position(mesh, fluid, par)

        self.assertAlmostEqual(front, 2.5)

    def test_radial_profile_csv_uses_cell_centers_and_skips_ghost_cells(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_filename = Path(tmpdir) / 'Output_000.hdf5'
            csv_filename = Path(tmpdir) / 'radial_profile.csv'
            with h5py.File(hdf5_filename, 'w') as hdf5:
                header = hdf5.create_group('Header')
                header.attrs['noghost'] = 1
                data = hdf5.create_group('Data')
                boundary = data.create_dataset(
                    'Boundary',
                    data=np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]) * 1.0e18,
                )
                boundary.attrs['units'] = 'cm'
                velocity = data.create_dataset(
                    'Velocity',
                    data=np.array([0.0, 1.0, 2.0, 3.0, 4.0]) * 1.0e5,
                )
                velocity.attrs['units'] = 'cm/s'
                density = data.create_dataset(
                    'Density',
                    data=np.arange(5.0) * (1.0 * unyt.mp).to_value(unyt.g),
                )
                density.attrs['units'] = 'g/cm**3'
                temperature = data.create_dataset(
                    'Temperature',
                    data=np.arange(5.0) * 100.0,
                )
                temperature.attrs['units'] = 'K'

            written = example_utils.write_radial_profile_csv(
                hdf5_filename,
                csv_filename,
            )

            lines = written.read_text().splitlines()
            self.assertEqual(
                lines[0],
                'RADIUS_PC,VELOCITY_KMS,DENSITY_CM3,TEMP_K',
            )
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[1].split(',')[0], '0.16203896')
            self.assertEqual(lines[1].split(',')[1], '1')
            self.assertEqual(lines[1].split(',')[2], '1')
            self.assertEqual(lines[1].split(',')[3], '100')

    def test_early_hii_region_expansion1d_uses_yaml_config(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HIIRegionExpansion1D'
            / 'early_hii_region_expansion1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['outfileprefix'], 'Output')
        self.assertEqual(runparams['noghost'], 2)
        self.assertEqual(runparams['CFL'], 0.5)
        self.assertEqual(runparams['order'], 1)
        self.assertEqual(runparams['timesim'].to_value(unyt.Myr), 0.14)
        self.assertEqual(icparams['number_of_cells'], 2048)

    def test_early_hii_region_expansion1d_c2ray_uses_distinct_outputs(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HIIRegionExpansion1D'
            / 'early_hii_region_expansion1d_c2ray.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['radiative_transfer_temporal_scheme'], 'c2ray')
        self.assertEqual(runparams['outfileprefix'], 'Output_C2Ray')
        self.assertEqual(runparams['ICfilename'].split('/')[-1], 'InitialCondition_C2Ray.hdf5')
        self.assertEqual(runparams['radiative_transfer_c2ray_nonconvergence'], 'warn')
        self.assertEqual(icparams['number_of_cells'], 2048)
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
        self.assertEqual(runparams['noghost'], 2)
        self.assertEqual(runparams['CFL'], 0.5)
        self.assertEqual(runparams['order'], 1)
        self.assertEqual(runparams['timesim'].to_value(unyt.Myr), 3.0)
        self.assertTrue(
            runparams['ICfilename'].endswith('InitialCondition_lateHII.hdf5')
        )
        self.assertEqual(icparams['number_of_cells'], 512)

    def test_late_hii_region_expansion1d_c2ray_uses_distinct_outputs(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HIIRegionExpansion1D'
            / 'late_hii_region_expansion1d_c2ray.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)

        self.assertEqual(runparams['radiative_transfer_temporal_scheme'], 'c2ray')
        self.assertEqual(runparams['outfileprefix'], 'Output_lateHII_C2Ray')
        self.assertEqual(
            runparams['ICfilename'].split('/')[-1],
            'InitialCondition_lateHII_C2Ray.hdf5',
        )
        self.assertEqual(runparams['radiative_transfer_c2ray_nonconvergence'], 'warn')
        self.assertEqual(icparams['number_of_cells'], 512)
        self.assertEqual(icparams['boxsize'].to_value(unyt.pc), 7.0)
        self.assertEqual(icparams['final_time'].to_value(unyt.Myr), 3.0)
        self.assertEqual(runparams['hydrogen_source_CFL'], 10000.0)
        self.assertTrue(Path(runparams['outputtimefilename']).exists())
        self.assertIn('CodeUnits', runparams)
        self.assertIsNotNone(runparams['CodeUnits'])
        self.assertTrue(icparams['show_stagnation_radius'])
        self.assertEqual(
            icparams['output_snapshots'][-1]['label'],
            '3p00',
        )

    def test_late_hii_initial_condition_file_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            icfilename = Path(tmpdir) / 'InitialCondition_lateHII.hdf5'
            icfilename.write_text('stale')
            runparams = {'ICfilename': str(icfilename)}
            sim = SimpleNamespace(name='dummy')

            with mock.patch.object(rio, 'writehdf5') as write_mock:
                late_hii.write_initial_condition(sim, runparams)

            self.assertFalse(icfilename.exists())
            write_mock.assert_called_once_with(sim, str(icfilename))

    def test_late_hii_region_snapshot_reload_recomputes_geometry(self):
        config_filename = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HIIRegionExpansion1D'
            / 'late_hii_region_expansion1d.yaml'
        )
        runparams, icparams = load_example_parameters(config_filename)
        config = {**runparams, **icparams}

        par, mesh, fluid, _ = hii_tools.build_problem(config)
        modified_boundary = np.asarray(mesh.boundary, dtype=float).copy() * 1.25
        mesh.boundary = modified_boundary * unyt.cm

        with tempfile.TemporaryDirectory() as tmpdir:
            outputfilename = Path(tmpdir) / 'Output_000.hdf5'
            rio.writehdf5(SimpleNamespace(par=par, mesh=mesh, fluid=fluid), outputfilename)

            out_par, out_mesh, out_fluid = hii_tools.load_output_state(outputfilename, config)

        interior = slice(out_par.noghost, out_par.noghost + out_par.nogrid)
        expected_coordinate = 0.5 * (modified_boundary[1:] + modified_boundary[:-1])
        vol_denom = modified_boundary[1:] ** 3 - modified_boundary[:-1] ** 3
        nonzero_vol_denom = vol_denom != 0.0
        expected_coordinate[nonzero_vol_denom] = 0.75 * (
            modified_boundary[1:][nonzero_vol_denom] ** 4
            - modified_boundary[:-1][nonzero_vol_denom] ** 4
        ) / vol_denom[nonzero_vol_denom]
        np.testing.assert_allclose(
            np.asarray(out_mesh.coordinate[interior], dtype=float),
            np.asarray(expected_coordinate[out_par.noghost : out_par.noghost + out_par.nogrid], dtype=float),
        )
        self.assertEqual(
            float(np.asarray(out_fluid.time, dtype=float)),
            0.0,
        )


if __name__ == '__main__':
    unittest.main()
