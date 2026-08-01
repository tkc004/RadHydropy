import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import unyt
import yaml
import h5py

import radhydropy.io as rio
from radhydropy.units import CodeUnits


CODE_UNITS = CodeUnits.from_mapping(
    {
        'name': 'test_units',
        'InternalUnitSystem': {
            'UnitMass_in_cgs': 1.0,
            'UnitLength_in_cgs': 1.0,
            'UnitVelocity_in_cgs': 1.0,
            'UnitCurrent_in_cgs': 1.0,
            'UnitTemp_in_cgs': 1.0,
        },
    }
)


class Testing(unittest.TestCase):
    @staticmethod
    def _scalar_value(value):
        return float(np.ravel(np.asarray(value))[0])

    def test_hdf5_roundtrip_handles_scalar_header_quantities(self):
        par = SimpleNamespace(
            coordsys='cartesian',
            nogrid=3,
            time=0.0 * unyt.s,
            boxsize=3.0 * unyt.cm,
            CodeUnits=CODE_UNITS,
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
        loaded_par = SimpleNamespace(coordsys='cartesian', CodeUnits=CODE_UNITS)
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(self._scalar_value(loaded_par.time), 0.0)
        self.assertEqual(self._scalar_value(loaded_par.boxsize), 3.0)
        self.assertEqual(self._scalar_value(loaded_fluid.time), 0.0)

    def test_writehdf5_writes_all_par_values_into_header_attributes(self):
        par = SimpleNamespace(
            coordsys='cartesian',
            nogrid=3,
            time=1.5 * unyt.s,
            boxsize=3.0 * unyt.cm,
            CodeUnits=CODE_UNITS,
            custom_scalar=7,
            custom_text='hello',
            custom_nested={'alpha': 1, 'beta': [2, 3]},
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

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            with h5py.File(output.name, 'r') as handle:
                header = handle['Header']
                self.assertEqual(header.attrs['coordsys'], 'cartesian')
                self.assertEqual(header.attrs['nogrid'], 3)
                self.assertEqual(header.attrs['custom_scalar'], 7)
                self.assertEqual(header.attrs['custom_text'], 'hello')
                self.assertEqual(
                    yaml.safe_load(header.attrs['custom_nested']),
                    {'alpha': 1, 'beta': [2, 3]},
                )
                self.assertEqual(
                    yaml.safe_load(header['Time'].attrs['units']),
                    's',
                )
                self.assertEqual(
                    np.asarray(header['Time'][()]).item(),
                    1.5,
                )
                self.assertEqual(
                    yaml.safe_load(header['BoxSize'].attrs['units']),
                    'cm',
                )
                self.assertEqual(
                    np.asarray(header['BoxSize'][()]).item(),
                    3.0,
                )

    def test_readhdf5_restores_header_attributes_and_code_units(self):
        par = SimpleNamespace(
            coordsys='cartesian',
            nogrid=3,
            time=1.5 * unyt.s,
            boxsize=3.0 * unyt.cm,
            CodeUnits=CODE_UNITS,
            custom_scalar=7,
            custom_nested={'alpha': 1, 'beta': [2, 3]},
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
        loaded_par = SimpleNamespace()
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertTrue(hasattr(loaded_par, 'CodeUnits'))
        self.assertEqual(loaded_par.CodeUnits.name, CODE_UNITS.name)
        self.assertEqual(loaded_par.coordsys, 'cartesian')
        self.assertEqual(loaded_par.nogrid, 3)
        self.assertEqual(loaded_par.custom_scalar, 7)
        self.assertEqual(loaded_par.custom_nested, {'alpha': 1, 'beta': [2, 3]})
        self.assertEqual(self._scalar_value(loaded_par.time), 1.5)
        self.assertEqual(self._scalar_value(loaded_par.boxsize), 3.0)

    def test_writehdf5_does_not_mutate_par_time(self):
        par = SimpleNamespace(
            coordsys='cartesian',
            nogrid=3,
            time=1.5 * unyt.s,
            boxsize=3.0 * unyt.cm,
            CodeUnits=CODE_UNITS,
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        fluid = SimpleNamespace(
            rho=np.ones(3) * unyt.g / unyt.cm**3,
            vel=np.zeros(3) * unyt.cm / unyt.s,
            temp=np.ones(3) * unyt.K,
            mu=np.ones(3),
            time=2.5 * unyt.s,
        )
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)

        self.assertEqual(par.time, 1.5 * unyt.s)

    def test_hdf5_roundtrip_preserves_neutral_fraction_when_present(self):
        par = SimpleNamespace(
            coordsys='cartesian',
            nogrid=3,
            time=np.array([0.0]) * unyt.s,
            boxsize=np.array([3.0]) * unyt.cm,
            CodeUnits=CODE_UNITS,
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
        loaded_par = SimpleNamespace(coordsys='cartesian', CodeUnits=CODE_UNITS)
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(self._scalar_value(loaded_fluid.time), self._scalar_value(loaded_par.time))
        np.testing.assert_array_equal(loaded_fluid.xHI, fluid.xHI)

    def test_hdf5_roundtrip_preserves_photon_number_density_when_present(self):
        par = SimpleNamespace(
            coordsys='cartesian',
            nogrid=3,
            time=np.array([0.0]) * unyt.s,
            boxsize=np.array([3.0]) * unyt.cm,
            CodeUnits=CODE_UNITS,
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
        loaded_par = SimpleNamespace(coordsys='cartesian', CodeUnits=CODE_UNITS)
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(self._scalar_value(loaded_fluid.time), self._scalar_value(loaded_par.time))
        self.assertFalse(hasattr(loaded_fluid.ngamma, "units"))
        np.testing.assert_array_equal(np.asarray(loaded_fluid.ngamma), fluid.ngamma.value)

    def test_readhdf5_errors_on_additional_datasets_with_units_when_not_preserving(self):
        par = SimpleNamespace(
            coordsys='cartesian',
            nogrid=3,
            time=np.array([0.0]) * unyt.s,
            boxsize=np.array([3.0]) * unyt.cm,
            CodeUnits=CODE_UNITS,
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
        loaded_par = SimpleNamespace(coordsys='cartesian', CodeUnits=CODE_UNITS)
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            with h5py.File(output.name, "a") as handle:
                extra = handle["Data"].create_dataset("InternalEnergy", data=np.array([1.0, 2.0, 3.0]))
                extra.attrs["units"] = "erg"

            with self.assertRaises(ValueError):
                rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

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

    def test_writehdf5_recovers_from_malformed_used_parameters_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                Path('used_parameters.yaml').write_text('runparams: [unclosed\n')

                par = SimpleNamespace(
                    coordsys='cartesian',
                    nogrid=3,
                    time=0.0 * unyt.s,
                    boxsize=3.0 * unyt.cm,
                    CodeUnits=CODE_UNITS,
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
                self.assertIn('runparams', payload)
                self.assertIn('ICparams', payload)
                self.assertEqual(payload['ICparams']['coordsys'], 'cartesian')
                self.assertEqual(payload['ICparams']['nogrid'], 3)
            finally:
                os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
