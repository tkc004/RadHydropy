import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from tests.parameter_fixtures import parameter_namespace
from unittest import mock
import importlib.util
import sys
import os

import numpy as np
import unyt
import yaml

from radhydropy.rsim import Rsim
import radhydropy.io as rio


class Testing(unittest.TestCase):
    def test_load_output_time_list_reads_unit_from_first_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'output_times.txt'
            path.write_text(
                '\n'.join(
                    [
                        'yr',
                        '1.0e4',
                        '2.5e4',
                        '# comment lines are ignored',
                        '3.0e4',
                    ]
                )
            )

            output_times = rio.load_output_time_list(path)

            self.assertEqual(output_times.units, unyt.yr)
            self.assertEqual(
                output_times.to_value(unyt.yr).tolist(),
                [1.0e4, 2.5e4, 3.0e4],
            )

    def test_run_with_output_times_emits_requested_outputs_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'output_times.txt'
            path.write_text('\n'.join(['s', '3.0', '1.0']))

            fluid = SimpleNamespace(
                time=0.0 * unyt.s,
                SetTemperature=lambda: None,
            )
            par = parameter_namespace(
                outputtimefilename=str(path),
                timesim=3.0 * unyt.s,
                outdir=str(tmpdir),
                outfileprefix='Output',
            )
            sim = Rsim.FromComponents(par, SimpleNamespace(), fluid)

            writes = []

            def fake_write(sim, index):
                writes.append((index, fluid.time.copy()))

            def fake_step(dt=None, mode=None, **kwargs):
                fluid.time += dt
                return {'dt': dt, 'hydro_steps': 1, 'source_steps': 1}

            def fake_get_step_time(dt=None, final_time=None):
                if dt is not None:
                    return dt
                if final_time is not None:
                    return final_time - fluid.time
                return 1.0 * unyt.s

            with mock.patch.object(rio, 'write_numbered_hdf5', side_effect=fake_write):
                sim.GetStepTime = fake_get_step_time
                rio.run_with_output_times(
                    sim,
                    mode='sources',
                    step_backend=fake_step,
                )

            self.assertEqual([index for index, _ in writes], [0, 1, 2])
            self.assertEqual(
                [time.to_value(unyt.s) for _, time in writes],
                [0.0, 1.0, 3.0],
            )
            self.assertEqual(fluid.time, 3.0 * unyt.s)

    def test_run_honors_stop_condition_in_source_only_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fluid = SimpleNamespace(
                time=0.0 * unyt.s,
                SetTemperature=lambda: None,
            )
            par = parameter_namespace(
                timesim=5.0 * unyt.s,
                outdir=str(tmpdir),
                outfileprefix='Output',
                outdeltatime=1.0 * unyt.s,
            )
            sim = Rsim.FromComponents(par, SimpleNamespace(), fluid)

            writes = []
            step_modes = []

            def fake_write(sim, index):
                writes.append((index, fluid.time.copy()))

            def fake_step(dt=None, mode=None, **kwargs):
                step_modes.append(mode)
                fluid.time += dt
                return {'dt': dt, 'hydro_steps': 0, 'source_steps': 1}

            def fake_get_step_time(dt=None, final_time=None):
                if dt is not None:
                    return dt
                if final_time is not None:
                    return 1.0 * unyt.s
                return 1.0 * unyt.s

            with mock.patch.object(rio, 'write_numbered_hdf5', side_effect=fake_write):
                sim.GetStepTime = fake_get_step_time
                sim.Step = fake_step
                rio.run_with_output_times(
                    sim,
                    mode='sources',
                    stop_condition=lambda runner: runner.fluid.time >= 2.5 * unyt.s,
                )

            self.assertEqual(step_modes, ['sources', 'sources', 'sources'])
            self.assertEqual([index for index, _ in writes], [0, 1])
            self.assertEqual(
                [time.to_value(unyt.s) for _, time in writes],
                [0.0, 3.0],
            )
            self.assertEqual(fluid.time, 3.0 * unyt.s)

    def test_run_writes_used_parameters_in_current_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                par = parameter_namespace(
                    timesim=0.0 * unyt.s,
                    outdir=str(tmpdir),
                    outfileprefix='Output',
                    outdeltatime=1.0 * unyt.s,
                    simname='test_run',
                )
                fluid = SimpleNamespace(
                    time=0.0 * unyt.s,
                    SetTemperature=lambda: None,
                )
                sim = Rsim.FromComponents(par, SimpleNamespace(), fluid)
                sim.Step = lambda **kwargs: {'dt': 0.0 * unyt.s, 'hydro_steps': 0, 'source_steps': 0}
                sim.Evolve = lambda **kwargs: None

                with mock.patch.object(rio, 'write_numbered_hdf5', lambda *args, **kwargs: None):
                    sim.Run()

                used_parameters = Path(tmpdir) / 'used_parameters.yaml'
                self.assertTrue(used_parameters.exists())
                payload = yaml.safe_load(used_parameters.read_text())
                self.assertIn('runparams', payload)
                self.assertIn('ICparams', payload)
                self.assertEqual(payload['runparams']['simname'], 'test_run')
                self.assertEqual(payload['runparams']['timesim']['value'], 0.0)
                self.assertEqual(payload['runparams']['timesim']['unit'], 's')
                self.assertIsNone(payload['ICparams'])
            finally:
                os.chdir(cwd)

    def test_parameter_tree_converts_numpy_scalars(self):
        self.assertEqual(rio.parameter_tree(np.int64(256)), 256)
        self.assertEqual(rio.parameter_tree(np.bool_(True)), True)

    def test_hydrostatic_example_plots_interior_cells_in_cgs(self):
        example_dir = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HydrostaticEquilibrium1D'
        )
        tools_path = example_dir / 'tools.py'
        spec = importlib.util.spec_from_file_location(
            'hydrostatic_equilibrium_tools_test',
            tools_path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(example_dir))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)

        captured_plots = []

        def fake_plot(x, y, **kwargs):
            captured_plots.append((x, y, kwargs))

        def fake_readhdf5(par, mesh, fluid, outfilename):
            mesh.boundary = np.linspace(0.0, 7.0, 8) * unyt.pc
            fluid.rho = np.linspace(1.0, 7.0, 7) * (unyt.g / unyt.cm**3)
            fluid.vel = np.linspace(-3.0, 3.0, 7) * (unyt.cm / unyt.s)

        with mock.patch.object(module.rio, 'readhdf5', fake_readhdf5), \
            mock.patch.object(module.plt, 'plot', side_effect=fake_plot), \
            mock.patch.object(module.plt, 'subplot', return_value=None), \
            mock.patch.object(module.plt, 'ylabel', return_value=None):
            module.ReadandPlot(
                'unused.hdf5',
                {
                    'nogrid': 5,
                    'coordsys': 'cartesian',
                    'boxsize': 1.0 * unyt.pc,
                    'time': 0.0 * unyt.s,
                    'rho_ref': 1.0 * (unyt.g / unyt.cm**3),
                    'tempini': 1.0 * unyt.K,
                    'muini': 1.0,
                    'gravity_strength': 1.0 * (unyt.cm / unyt.s**2),
                },
                {'noghost': 2},
            )

        self.assertEqual(len(captured_plots), 4)

        density_x, density_y, _ = captured_plots[0]
        analytic_x, analytic_y, _ = captured_plots[1]
        velocity_x, velocity_y, _ = captured_plots[2]
        zero_x, zero_y, _ = captured_plots[3]

        self.assertEqual(density_x.units, unyt.cm)
        self.assertEqual(analytic_x.units, unyt.cm)
        self.assertEqual(velocity_x.units, unyt.cm)
        self.assertEqual(zero_x.units, unyt.cm)
        self.assertEqual(density_x.shape[0], 3)
        self.assertEqual(analytic_x.shape[0], 3)
        self.assertEqual(velocity_x.shape[0], 3)
        self.assertEqual(zero_x.shape[0], 3)
        self.assertEqual(density_y.units, unyt.g / unyt.cm**3)
        self.assertEqual(analytic_y.units, unyt.g / unyt.cm**3)
        self.assertEqual(velocity_y.units, unyt.cm / unyt.s)
        self.assertEqual(zero_y.units, unyt.cm / unyt.s)

    def test_hydrogen_recombination_helper_uses_source_only_wrapper(self):
        example_dir = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HydrogenRecombination1D'
        )
        tools_path = example_dir / 'tools.py'
        spec = importlib.util.spec_from_file_location(
            'hydrogen_recombination_tools_test',
            tools_path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(example_dir))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)

        sim = SimpleNamespace(
            par = parameter_namespace(noghost=2, nogrid=3),
            fluid=SimpleNamespace(
                xHI=np.array([0.0, 0.0, 0.8, 0.9, 1.0]),
            ),
        )

        captured = {}

        def fake_runall(**kwargs):
            captured.update(kwargs)
            self.assertEqual(kwargs['mode'], 'sources')
            self.assertEqual(kwargs['outputtime'], 0)
            self.assertTrue(kwargs['stop_condition'](sim))
            return 'wrapped'

        sim.RunAll = fake_runall

        result = module.run_hydrogen_recombination(sim, 0.7)

        self.assertEqual(result, 'wrapped')
        self.assertEqual(captured['mode'], 'sources')
        self.assertEqual(captured['outputtime'], 0)
        self.assertTrue(captured['stop_condition'](sim))


if __name__ == '__main__':
    unittest.main()
