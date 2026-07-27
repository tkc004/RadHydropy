import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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


if __name__ == '__main__':
    unittest.main()
