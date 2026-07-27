import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import unyt
import yaml

import radhydropy.io as rio


class Testing(unittest.TestCase):
    def test_hdf5_roundtrip_handles_scalar_header_quantities(self):
        par = SimpleNamespace(
            coordsys='cartesian',
            nogrid=3,
            time=0.0 * unyt.s,
            boxsize=3.0 * unyt.cm,
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        fluid = SimpleNamespace(
            rho=np.ones(3) * unyt.g / unyt.cm**3,
            vel=np.zeros(3) * unyt.cm / unyt.s,
            temp=np.ones(3) * unyt.K,
            mu=np.ones(3),
        )
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = SimpleNamespace(coordsys='cartesian')
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(loaded_par.time, par.time)
        self.assertEqual(loaded_par.boxsize, par.boxsize)
        self.assertEqual(loaded_fluid.time, par.time)

    def test_hdf5_roundtrip_preserves_neutral_fraction_when_present(self):
        par = SimpleNamespace(
            coordsys='cartesian',
            nogrid=3,
            time=np.array([0.0]) * unyt.s,
            boxsize=np.array([3.0]) * unyt.cm,
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        fluid = SimpleNamespace(
            rho=np.ones(3) * unyt.g/unyt.cm**3,
            vel=np.zeros(3) * unyt.cm/unyt.s,
            temp=np.ones(3) * unyt.K,
            mu=np.ones(3),
            xHI=np.array([1.0, 0.5, 0.0]),
        )
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = SimpleNamespace(coordsys='cartesian')
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(loaded_fluid.time, loaded_par.time)
        np.testing.assert_array_equal(loaded_fluid.xHI, fluid.xHI)

    def test_hdf5_roundtrip_preserves_photon_number_density_when_present(self):
        par = SimpleNamespace(
            coordsys='cartesian',
            nogrid=3,
            time=np.array([0.0]) * unyt.s,
            boxsize=np.array([3.0]) * unyt.cm,
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        fluid = SimpleNamespace(
            rho=np.ones(3) * unyt.g/unyt.cm**3,
            vel=np.zeros(3) * unyt.cm/unyt.s,
            temp=np.ones(3) * unyt.K,
            mu=np.ones(3),
            ngamma=np.array([0.0, 1.0, 2.0]) / unyt.cm**3,
        )
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = SimpleNamespace(coordsys='cartesian')
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(loaded_fluid.time, loaded_par.time)
        self.assertEqual(loaded_fluid.ngamma.units, fluid.ngamma.units)
        np.testing.assert_array_equal(loaded_fluid.ngamma.value, fluid.ngamma.value)

    def test_writehdf5_appends_icparams_to_used_parameters_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                Path(tmpdir).mkdir(parents=True, exist_ok=True)
                os.chdir(tmpdir)
                Path('used_parameters.yaml').write_text(
                    yaml.safe_dump(
                        {
                            'runparams': {
                                'simname': 'preexisting',
                                'timesim': {'value': 1.0, 'unit': 's'},
                            },
                            'ICparams': {},
                        }
                    )
                )

                par = SimpleNamespace(
                    coordsys='cartesian',
                    nogrid=3,
                    time=0.0 * unyt.s,
                    boxsize=3.0 * unyt.cm,
                )
                mesh = SimpleNamespace(
                    boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
                )
                fluid = SimpleNamespace(
                    rho=np.ones(3) * unyt.g / unyt.cm**3,
                    vel=np.zeros(3) * unyt.cm / unyt.s,
                    temp=np.ones(3) * unyt.K,
                    mu=np.ones(3),
                )
                sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)

                rio.writehdf5(sim, 'InitialCondition.hdf5')

                payload = yaml.safe_load(Path('used_parameters.yaml').read_text())
                self.assertEqual(payload['runparams']['simname'], 'preexisting')
                self.assertEqual(payload['runparams']['timesim']['value'], 1.0)
                self.assertEqual(payload['ICparams']['coordsys'], 'cartesian')
                self.assertEqual(payload['ICparams']['nogrid'], 3)
                self.assertEqual(payload['ICparams']['boxsize']['value'], 3.0)
                self.assertEqual(payload['ICparams']['boxsize']['unit'], 'cm')
            finally:
                os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
