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
from radhydropy.params import Par
import radhydropy.io as rio
import radhydropy.output as output
from radhydropy.runtime_fields import MeshGeometryState


CODE_UNITS = {
    'name': 'test_units',
    'InternalUnitSystem': {
        'UnitMass_in_cgs': 1.0,
        'UnitLength_in_cgs': 1.0,
        'UnitVelocity_in_cgs': 1.0,
        'UnitCurrent_in_cgs': 1.0,
        'UnitTemp_in_cgs': 1.0,
    },
}


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
                time_proper_code=0.0 * unyt.s,
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
                writes.append((index, fluid.time_proper_code.copy()))

            def fake_step(dt=None, mode=None, **kwargs):
                fluid.time_proper_code += dt
                return {'dt': dt, 'hydro_steps': 1, 'source_steps': 1}

            def fake_get_step_time(dt=None, final_time=None):
                if dt is not None:
                    return dt
                if final_time is not None:
                    return final_time - fluid.time_proper_code
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
            self.assertEqual(fluid.time_proper_code, 3.0 * unyt.s)

    def test_run_with_output_times_notifies_snapshot_callback_with_written_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'output_times.txt'
            path.write_text('\n'.join(['s', '1.0']))
            fluid = SimpleNamespace(
                time_proper_code=0.0 * unyt.s,
                SetTemperature=lambda: None,
            )
            par = parameter_namespace(
                outputtimefilename=str(path),
                timesim=1.0 * unyt.s,
                outdir=str(tmpdir),
                outfileprefix='Output',
            )
            sim = Rsim.FromComponents(par, SimpleNamespace(), fluid)
            callbacks = []

            def fake_write(current_sim, index):
                filename = Path(tmpdir) / ('written_%d.hdf5' % index)
                filename.touch()
                return str(filename)

            def fake_step(dt=None, mode=None, **kwargs):
                fluid.time_proper_code += dt
                return {'dt': dt, 'hydro_steps': 1, 'source_steps': 0}

            sim.GetStepTime = lambda dt=None, final_time=None: (
                final_time - fluid.time_proper_code
            )
            with mock.patch.object(rio, 'write_numbered_hdf5', side_effect=fake_write):
                rio.run_with_output_times(
                    sim,
                    mode='sources',
                    step_backend=fake_step,
                    snapshot_callback=lambda current_sim, filename, index: callbacks.append(
                        (filename, index, Path(filename).exists())
                    ),
                )

            self.assertEqual([index for _, index, _ in callbacks], [0, 1])
            self.assertTrue(all(exists for _, _, exists in callbacks))

    def test_run_with_output_times_orders_pre_step_history_and_snapshot_callbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'output_times.txt'
            path.write_text('\n'.join(['s', '1.0']))
            fluid = SimpleNamespace(
                time_proper_code=0.0 * unyt.s,
                SetTemperature=lambda: None,
            )
            par = parameter_namespace(
                outputtimefilename=str(path),
                timesim=1.0 * unyt.s,
                outdir=str(tmpdir),
                outfileprefix='Output',
            )
            sim = Rsim.FromComponents(par, SimpleNamespace(), fluid)
            events = []

            def fake_write(current_sim, index):
                filename = Path(tmpdir) / ('ordered_%d.hdf5' % index)
                filename.touch()
                events.append('write_%d' % index)
                return str(filename)

            def fake_get_step_time(dt=None, final_time=None):
                events.append('get_step_time')
                return final_time - fluid.time_proper_code

            def fake_step(dt=None, mode=None, **kwargs):
                events.append('step')
                fluid.time_proper_code += dt
                return {'dt': dt, 'hydro_steps': 1, 'source_steps': 0}

            sim.GetStepTime = fake_get_step_time
            with mock.patch.object(rio, 'write_numbered_hdf5', side_effect=fake_write):
                rio.run_with_output_times(
                    sim,
                    mode='sources',
                    step_backend=fake_step,
                    before_step_callback=lambda current_sim: events.append('before_step'),
                    history_callback=lambda current_sim: events.append('history'),
                    snapshot_callback=lambda current_sim, filename, index: events.append(
                        'snapshot_%d' % index
                    ),
                )

            self.assertEqual(
                events,
                ['before_step', 'write_0', 'snapshot_0', 'history',
                 'before_step', 'get_step_time', 'step', 'history',
                 'write_1', 'snapshot_1'],
            )

    def test_fixed_cadence_snapshot_callback_runs_after_initial_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            par = parameter_namespace(
                timesim=0.0 * unyt.s,
                outdir=str(tmpdir),
                outfileprefix='Output',
                outdeltatime=1.0 * unyt.s,
            )
            fluid = SimpleNamespace(
                time_proper_code=0.0 * unyt.s,
                SetTemperature=lambda: None,
            )
            sim = Rsim.FromComponents(par, SimpleNamespace(), fluid)
            events = []
            sim.WriteUsedParameters = lambda: None
            sim.Evolve = lambda **kwargs: events.append('evolve')

            def fake_write(current_sim, index):
                filename = Path(tmpdir) / ('fixed_%d.hdf5' % index)
                filename.touch()
                events.append('write_%d' % index)
                return str(filename)

            with mock.patch.object(rio, 'write_numbered_hdf5', side_effect=fake_write):
                sim.Run(
                    snapshot_callback=lambda current_sim, filename, index: events.append(
                        'snapshot_%d_exists_%s' % (index, Path(filename).exists())
                    ),
                )

            self.assertEqual(
                events,
                ['write_0', 'snapshot_0_exists_True', 'evolve'],
            )

    def test_fixed_cadence_output_callback_notifies_after_serialization(self):
        par = parameter_namespace(
            timesim=1.0 * unyt.s,
            outdir='unused',
            outfileprefix='Output',
            outdeltatime=1.0 * unyt.s,
        )
        fluid = SimpleNamespace(
            time_proper_code=1.0 * unyt.s,
            SetTemperature=lambda: None,
        )
        sim = Rsim.FromComponents(par, SimpleNamespace(), fluid)
        events = []
        output_state = {'outtime': 1.0 * unyt.s, 'outindex': 1}

        def fake_write(current_sim, index):
            events.append('write')
            return 'actual_filename.hdf5'

        with mock.patch.object(output, 'write_numbered_hdf5', side_effect=fake_write):
            callback = output.hdf5_output_callback(
                sim,
                snapshot_callback=lambda current_sim, filename, index: events.append(
                    ('snapshot', filename, index)
                ),
                output_state=output_state,
            )
            callback(sim, {'dt': 0.1 * unyt.s})

        self.assertEqual(events, ['write', ('snapshot', 'actual_filename.hdf5', 1)])

    def test_run_honors_stop_condition_in_source_only_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fluid = SimpleNamespace(
                time_proper_code=0.0 * unyt.s,
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
                writes.append((index, fluid.time_proper_code.copy()))

            def fake_step(dt=None, mode=None, **kwargs):
                step_modes.append(mode)
                fluid.time_proper_code += dt
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
                stop_condition=lambda runner: runner.fluid.time_proper_code >= 2.5 * unyt.s,
                )

            self.assertEqual(step_modes, ['sources', 'sources', 'sources'])
            self.assertEqual([index for index, _ in writes], [0, 1])
            self.assertEqual(
                [time.to_value(unyt.s) for _, time in writes],
                [0.0, 3.0],
            )
            self.assertEqual(fluid.time_proper_code, 3.0 * unyt.s)

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
                    time_proper_code=0.0 * unyt.s,
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
                self.assertIn('par', payload)
                self.assertIn('initial_condition', payload)
                self.assertEqual(payload['par']['simname'], 'test_run')
                self.assertEqual(payload['par']['timesim']['value'], 0.0)
                self.assertEqual(payload['par']['timesim']['unit'], 's')
                self.assertIsNone(payload['initial_condition'])
            finally:
                os.chdir(cwd)

    def test_used_parameters_preserves_nested_runtime_groups(self):
        par = Par({
            'simulation': {'name': 'nested-test'},
            'units': {'CodeUnits': CODE_UNITS},
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'used_parameters.yaml'
            rio.write_used_parameters(path, par)
            payload = yaml.safe_load(path.read_text())

        self.assertEqual(
            list(payload['par']),
            [
                'simulation', 'mesh', 'hydrodynamics', 'boundary', 'timestep',
                'output', 'diagnostics', 'units', 'thermochemistry',
                'chemistry', 'gravity', 'dark_matter', 'radiation',
            ],
        )
        self.assertEqual(payload['par']['simulation']['name'], 'nested-test')
        self.assertNotIn('simname', payload['par'])
        self.assertEqual(set(payload), {'par', 'initial_condition', 'example'})

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
            mesh.geometry_state = MeshGeometryState(
                boundary_proper_code=np.linspace(0.0, 7.0, 8),
            )
            mesh.boundary_radarray = np.linspace(0.0, 7.0, 8) * unyt.cm
            fluid.rho_proper_code = np.linspace(1.0, 7.0, 7)
            fluid.vel_proper_code = np.linspace(-3.0, 3.0, 7)
            fluid.rho_radarray = fluid.rho_proper_code * (unyt.g / unyt.cm**3)
            fluid.vel_radarray = fluid.vel_proper_code * (unyt.cm / unyt.s)

        with mock.patch.object(module.rio, 'readhdf5', fake_readhdf5), \
            mock.patch.object(module.plt, 'plot', side_effect=fake_plot), \
            mock.patch.object(module.plt, 'subplot', return_value=None), \
            mock.patch.object(module.plt, 'ylabel', return_value=None):
            module.plot_snapshot(
                'unused.hdf5',
                {
                    'par': {
                        'mesh': {'grid_cells': 3, 'ghost_cells': 2},
                        'units': {'CodeUnits': CODE_UNITS},
                    },
                    'initial_condition': {
                        'grid_cells': 3,
                        'coordinate_system': 'cartesian',
                        'box_size_proper': 1.0 * unyt.cm,
                        'time_proper': 0.0 * unyt.s,
                        'rho_reference_proper': 1.0 * (unyt.g / unyt.cm**3),
                        'temperature_proper': 1.0 * unyt.K,
                        'mean_molecular_weight': 1.0,
                        'gravity_strength': 1.0 * (unyt.cm / unyt.s**2),
                    },
                    'example': {},
                },
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
