import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import os

import numpy as np
import unyt

from radhydropy.rsim import Rsim


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

            sim = Rsim.__new__(Rsim)
            sim.par = SimpleNamespace(
                outputtimefilename=str(path),
                timesim=1.0e5 * unyt.yr,
            )

            output_times = sim._load_output_time_list()

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
            par = SimpleNamespace(
                outputtimefilename=str(path),
                timesim=5.0 * unyt.s,
                outdir=str(tmpdir),
                outfileprefix='Output',
            )
            sim = Rsim.FromComponents(par, SimpleNamespace(), fluid)

            writes = []

            def fake_write(index):
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

            sim._write_numbered_hdf5 = fake_write
            sim.Step = fake_step
            sim.GetStepTime = fake_get_step_time

            sim._run_with_output_times()

            self.assertEqual([index for index, _ in writes], [0, 1, 2])
            self.assertEqual(
                [time.to_value(unyt.s) for _, time in writes],
                [0.0, 1.0, 3.0],
            )
            self.assertEqual(fluid.time, 5.0 * unyt.s)

    def test_run_honors_stop_condition_in_source_only_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fluid = SimpleNamespace(
                time=0.0 * unyt.s,
                SetTemperature=lambda: None,
            )
            par = SimpleNamespace(
                timesim=5.0 * unyt.s,
                outdir=str(tmpdir),
                outfileprefix='Output',
                outdeltatime=1.0 * unyt.s,
            )
            sim = Rsim.FromComponents(par, SimpleNamespace(), fluid)

            writes = []
            step_modes = []

            def fake_write(index):
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

            sim._write_numbered_hdf5 = fake_write
            sim.Step = fake_step
            sim.GetStepTime = fake_get_step_time

            sim.Run(
                mode='sources',
                stop_condition=lambda runner: runner.fluid.time >= 2.5 * unyt.s,
            )

            self.assertEqual(step_modes, ['sources', 'sources', 'sources'])
            self.assertEqual([index for index, _ in writes], [0, 1, 2])
            self.assertEqual(
                [time.to_value(unyt.s) for _, time in writes],
                [0.0, 2.0, 3.0],
            )
            self.assertEqual(fluid.time, 3.0 * unyt.s)

    def test_run_writes_used_parameters_in_current_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                fluid = SimpleNamespace(
                    time=0.0 * unyt.s,
                    SetTemperature=lambda: None,
                )
                par = SimpleNamespace(
                    timesim=0.0 * unyt.s,
                    outdir=str(tmpdir),
                    outfileprefix='Output',
                    outdeltatime=1.0 * unyt.s,
                    simname='test_run',
                )
                sim = Rsim.FromComponents(par, SimpleNamespace(), fluid)
                sim._write_numbered_hdf5 = lambda index: None
                sim.Evolve = lambda **kwargs: None

                sim.Run()

                used_parameters = Path(tmpdir) / 'used_parameters.txt'
                self.assertTrue(used_parameters.exists())
                text = used_parameters.read_text()
                self.assertIn('timesim:', text)
                self.assertIn('simname:', text)
                self.assertIn('test_run', text)
            finally:
                os.chdir(cwd)

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
            par=SimpleNamespace(noghost=2, nogrid=3),
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
