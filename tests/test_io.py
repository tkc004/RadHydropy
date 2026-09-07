import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from tests.parameter_fixtures import parameter_namespace

import numpy as np
import unyt
import yaml
import h5py

import radhydropy.io as rio
from radhydropy.units import CodeUnits
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
)


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


def _attach_proper_runtime_state(mesh, fluid):
    """Attach the numeric proper-code state used by the serializer tests."""
    boundary_proper_code = np.asarray(mesh.boundary.to_value(unyt.cm), dtype=float)
    width_proper_code = np.diff(boundary_proper_code)
    mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        coordinate=0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1]),
        boundary=boundary_proper_code,
        width=width_proper_code,
        area=np.ones_like(width_proper_code),
        volume=width_proper_code,
    )
    density_proper_code = np.asarray(
        fluid.rho_code.to_value(unyt.g / unyt.cm**3), dtype=float
    )
    velocity_proper_code = np.asarray(
        fluid.vel_code.to_value(unyt.cm / unyt.s), dtype=float
    )
    temperature_proper_code = np.asarray(
        fluid.temp_code.to_value(unyt.K), dtype=float
    )
    fluid.rho_proper_code = density_proper_code
    fluid.vel_proper_code = velocity_proper_code
    fluid.temp_proper_code = temperature_proper_code
    fluid.pre_proper_code = np.ones_like(density_proper_code)
    fluid.time_proper_code = 0.0
    fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        density=density_proper_code,
        velocity=velocity_proper_code,
        pressure=fluid.pre_proper_code,
        temperature=temperature_proper_code,
        time=fluid.time_proper_code,
        mu=getattr(fluid, "mu", None),
        xHI=getattr(fluid, "xHI", None),
    )
    return mesh, fluid


class Testing(unittest.TestCase):
    @staticmethod
    def _scalar_value(value):
        return float(np.ravel(np.asarray(value))[0])

    def test_hdf5_roundtrip_handles_scalar_header_quantities(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=0.0 * unyt.s,
            boxsize=3.0 * unyt.cm,
            CodeUnits=CODE_UNITS,
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        fluid = SimpleNamespace(
            rho_code=np.ones(3) * unyt.g / unyt.cm**3,
            vel_code=np.zeros(3) * unyt.cm / unyt.s,
            temp_code=np.ones(3) * unyt.K,
            mu=np.ones(3),
        )
        mesh, fluid = _attach_proper_runtime_state(mesh, fluid)
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = parameter_namespace(coordsys='cartesian', CodeUnits=CODE_UNITS)
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(self._scalar_value(loaded_par.time_proper_code), 0.0)
        self.assertEqual(self._scalar_value(loaded_par.boxsize), 3.0)
        self.assertEqual(self._scalar_value(loaded_fluid.time_proper_code), 0.0)

    def test_hdf5_uses_canonical_code_state_dataset_names(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=2,
            time_code=0.0 * unyt.s,
            boxsize=2.0 * unyt.cm,
            CodeUnits=CODE_UNITS,
        )
        mesh = SimpleNamespace(boundary=np.array([0.0, 1.0, 2.0]) * unyt.cm)
        fluid = SimpleNamespace(
            rho_code=np.ones(2) * unyt.g / unyt.cm**3,
            vel_code=np.zeros(2) * unyt.cm / unyt.s,
            temp_code=np.ones(2) * unyt.K,
            mu=np.ones(2),
            xHI=np.ones(2),
            Mass_code=np.ones(2) * unyt.g,
            Energy_code=np.ones(2) * unyt.erg,
            ngamma_code=np.ones(2) / unyt.cm**3,
        )
        mesh, fluid = _attach_proper_runtime_state(mesh, fluid)
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = parameter_namespace(coordsys='cartesian', CodeUnits=CODE_UNITS)
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            with h5py.File(output.name, 'r') as handle:
                data_names = set(handle['Data'].keys())
                assert {
                    'boundary_proper_code', 'rho_proper_code', 'vel_proper_code', 'temp_proper_code',
                    'Mass_code', 'Energy_code', 'ngamma_code', 'mu', 'xHI',
                }.issubset(data_names)
                assert not {'Density', 'Velocity', 'Temperature', 'Mass', 'Energy'}.intersection(data_names)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        np.testing.assert_allclose(loaded_fluid.rho_proper_code, fluid.rho_code.value)
        np.testing.assert_allclose(loaded_fluid.ngamma_code, fluid.ngamma_code.value)
        np.testing.assert_allclose(loaded_mesh.boundary_proper_code, mesh.boundary.to_value(unyt.cm))

    def test_hdf5_roundtrip_preserves_gas_angular_momentum_fields(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=0.0 * unyt.s,
            boxsize=3.0 * unyt.cm,
            CodeUnits=CODE_UNITS,
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        specific = np.array([1.0, 2.0, 3.0]) * unyt.cm**2 / unyt.s
        angular = np.array([4.0, 5.0, 6.0]) * unyt.g * unyt.cm**2 / unyt.s
        fluid = SimpleNamespace(
            rho_code=np.ones(3) * unyt.g / unyt.cm**3,
            vel_code=np.zeros(3) * unyt.cm / unyt.s,
            temp_code=np.ones(3) * unyt.K,
            mu=np.ones(3),
            specific_angular_momentum_code=specific,
            AngularMomentum_code=angular,
        )
        mesh, fluid = _attach_proper_runtime_state(mesh, fluid)
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = parameter_namespace(coordsys='cartesian', CodeUnits=CODE_UNITS)
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        np.testing.assert_allclose(
            loaded_fluid.specific_angular_momentum_code, np.asarray(specific)
        )
        np.testing.assert_allclose(loaded_fluid.AngularMomentum_code, np.asarray(angular))

    def test_writehdf5_writes_all_par_values_into_header_attributes(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=1.5 * unyt.s,
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
            rho_code=np.ones(3) * unyt.g / unyt.cm**3,
            vel_code=np.zeros(3) * unyt.cm / unyt.s,
            temp_code=np.ones(3) * unyt.K,
            mu=np.ones(3),
        )
        mesh, fluid = _attach_proper_runtime_state(mesh, fluid)
        fluid.time_proper_code = 1.5
        fluid.runtime_state.time_proper_code = fluid.time_proper_code
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
                    yaml.safe_load(header['time_proper_code'].attrs['units']),
                    's',
                )
                self.assertEqual(
                    np.asarray(header['time_proper_code'][()]).item(),
                    1.5,
                )
                self.assertEqual(
                    yaml.safe_load(header['box_size_proper_code'].attrs['units']),
                    'cm',
                )
                self.assertEqual(
                    np.asarray(header['box_size_proper_code'][()]).item(),
                    3.0,
                )

    def test_readhdf5_restores_header_attributes_and_code_units(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=1.5 * unyt.s,
            boxsize=3.0 * unyt.cm,
            CodeUnits=CODE_UNITS,
            custom_scalar=7,
            custom_nested={'alpha': 1, 'beta': [2, 3]},
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        fluid = SimpleNamespace(
            rho_code=np.ones(3) * unyt.g / unyt.cm**3,
            vel_code=np.zeros(3) * unyt.cm / unyt.s,
            temp_code=np.ones(3) * unyt.K,
            mu=np.ones(3),
        )
        mesh, fluid = _attach_proper_runtime_state(mesh, fluid)
        fluid.time_proper_code = 1.5
        fluid.runtime_state.time_proper_code = fluid.time_proper_code
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = parameter_namespace()
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
        self.assertEqual(self._scalar_value(loaded_par.time_proper_code), 1.5)
        self.assertEqual(self._scalar_value(loaded_par.boxsize), 3.0)

    def test_writehdf5_does_not_mutate_par_time(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=1.5 * unyt.s,
            boxsize=3.0 * unyt.cm,
            CodeUnits=CODE_UNITS,
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        fluid = SimpleNamespace(
            rho_code=np.ones(3) * unyt.g / unyt.cm**3,
            vel_code=np.zeros(3) * unyt.cm / unyt.s,
            temp_code=np.ones(3) * unyt.K,
            mu=np.ones(3),
            time_code=2.5 * unyt.s,
        )
        mesh, fluid = _attach_proper_runtime_state(mesh, fluid)
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)

        self.assertEqual(par.time_code, 1.5 * unyt.s)

    def test_hdf5_roundtrip_preserves_neutral_fraction_when_present(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=np.array([0.0]) * unyt.s,
            boxsize=np.array([3.0]) * unyt.cm,
            CodeUnits=CODE_UNITS,
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        fluid = SimpleNamespace(
            rho_code=np.ones(3) * unyt.g/unyt.cm**3,
            vel_code=np.zeros(3) * unyt.cm/unyt.s,
            temp_code=np.ones(3) * unyt.K,
            mu=np.ones(3),
            xHI=np.array([1.0, 0.5, 0.0]),
        )
        mesh, fluid = _attach_proper_runtime_state(mesh, fluid)
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = parameter_namespace(coordsys='cartesian', CodeUnits=CODE_UNITS)
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(self._scalar_value(loaded_fluid.time_proper_code), self._scalar_value(loaded_par.time_proper_code))
        np.testing.assert_array_equal(loaded_fluid.xHI, fluid.xHI)

    def test_hdf5_roundtrip_preserves_photon_number_density_when_present(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=np.array([0.0]) * unyt.s,
            boxsize=np.array([3.0]) * unyt.cm,
            CodeUnits=CODE_UNITS,
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        fluid = SimpleNamespace(
            rho_code=np.ones(3) * unyt.g/unyt.cm**3,
            vel_code=np.zeros(3) * unyt.cm/unyt.s,
            temp_code=np.ones(3) * unyt.K,
            mu=np.ones(3),
            ngamma_code=np.array([0.0, 1.0, 2.0]) / unyt.cm**3,
        )
        mesh, fluid = _attach_proper_runtime_state(mesh, fluid)
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = parameter_namespace(coordsys='cartesian', CodeUnits=CODE_UNITS)
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(self._scalar_value(loaded_fluid.time_proper_code), self._scalar_value(loaded_par.time_proper_code))
        self.assertFalse(hasattr(loaded_fluid.ngamma_code, "units"))
        np.testing.assert_array_equal(np.asarray(loaded_fluid.ngamma_code), fluid.ngamma_code.value)

    def test_hdf5_roundtrip_preserves_internal_energy_when_present(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=np.array([0.0]) * unyt.s,
            boxsize=np.array([3.0]) * unyt.cm,
            CodeUnits=CODE_UNITS,
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        fluid = SimpleNamespace(
            rho_code=np.ones(3) * unyt.g / unyt.cm**3,
            vel_code=np.zeros(3) * unyt.cm / unyt.s,
            temp_code=np.ones(3) * unyt.K,
            mu=np.ones(3),
            InternalEnergy_code=np.array([1.0, 2.0, 3.0]) * unyt.erg,
        )
        mesh, fluid = _attach_proper_runtime_state(mesh, fluid)
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = parameter_namespace(coordsys='cartesian', CodeUnits=CODE_UNITS)
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertTrue(hasattr(loaded_fluid, "InternalEnergy_code"))
        self.assertFalse(hasattr(loaded_fluid.InternalEnergy_code, "units"))
        np.testing.assert_array_equal(
            np.asarray(loaded_fluid.InternalEnergy_code),
            np.asarray(fluid.InternalEnergy_code.value),
        )

    def test_readhdf5_errors_when_header_missing_code_units(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=np.array([0.0]) * unyt.s,
            boxsize=np.array([3.0]) * unyt.cm,
            CodeUnits=CODE_UNITS,
        )
        mesh = SimpleNamespace(
            boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
        )
        fluid = SimpleNamespace(
            rho_code=np.ones(3) * unyt.g / unyt.cm**3,
            vel_code=np.zeros(3) * unyt.cm / unyt.s,
            temp_code=np.ones(3) * unyt.K,
            mu=np.ones(3),
        )
        mesh, fluid = _attach_proper_runtime_state(mesh, fluid)
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = parameter_namespace(coordsys='cartesian')
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            with h5py.File(output.name, "a") as handle:
                del handle["Header"].attrs["CodeUnits"]

            with self.assertRaises(ValueError):
                rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

    def test_writehdf5_appends_initial_condition_to_used_parameters_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                Path(tmpdir).mkdir(parents=True, exist_ok=True)
                os.chdir(tmpdir)
                Path('used_parameters.yaml').write_text(
                    yaml.safe_dump(
                        {
                            'par': {
                                'simname': 'preexisting',
                                'timesim': {'value': 1.0, 'unit': 's'},
                            },
                            'initial_condition': {},
                        }
                    )
                )

                par = parameter_namespace(
                    coordsys='cartesian',
                    nogrid=3,
                    time_code=0.0 * unyt.s,
                    boxsize=3.0 * unyt.cm,
                    CodeUnits=CODE_UNITS,
                )
                mesh = SimpleNamespace(
                    boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
                )
                fluid = SimpleNamespace(
                    rho_code=np.ones(3) * unyt.g / unyt.cm**3,
                    vel_code=np.zeros(3) * unyt.cm / unyt.s,
                    temp_code=np.ones(3) * unyt.K,
                    mu=np.ones(3),
                )
                mesh, fluid = _attach_proper_runtime_state(mesh, fluid)
                sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)

                rio.writehdf5(sim, 'InitialCondition.hdf5')

                payload = yaml.safe_load(Path('used_parameters.yaml').read_text())
                self.assertEqual(payload['par']['simname'], 'preexisting')
                self.assertEqual(payload['par']['timesim']['value'], 1.0)
                self.assertEqual(payload['initial_condition']['coordsys'], 'cartesian')
                self.assertEqual(payload['initial_condition']['nogrid'], 3)
                self.assertEqual(payload['initial_condition']['boxsize']['value'], 3.0)
                self.assertEqual(payload['initial_condition']['boxsize']['unit'], 'cm')
            finally:
                os.chdir(cwd)

    def test_writehdf5_recovers_from_malformed_used_parameters_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                Path('used_parameters.yaml').write_text('par: [unclosed\n')

                par = parameter_namespace(
                    coordsys='cartesian',
                    nogrid=3,
                    time_code=0.0 * unyt.s,
                    boxsize=3.0 * unyt.cm,
                    CodeUnits=CODE_UNITS,
                )
                mesh = SimpleNamespace(
                    boundary=np.linspace(0.0, 3.0, 4) * unyt.cm,
                )
                fluid = SimpleNamespace(
                    rho_code=np.ones(3) * unyt.g / unyt.cm**3,
                    vel_code=np.zeros(3) * unyt.cm / unyt.s,
                    temp_code=np.ones(3) * unyt.K,
                    mu=np.ones(3),
                )
                mesh, fluid = _attach_proper_runtime_state(mesh, fluid)
                sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)

                rio.writehdf5(sim, 'InitialCondition.hdf5')

                payload = yaml.safe_load(Path('used_parameters.yaml').read_text())
                self.assertIn('par', payload)
                self.assertIn('initial_condition', payload)
                self.assertEqual(payload['initial_condition']['coordsys'], 'cartesian')
                self.assertEqual(payload['initial_condition']['nogrid'], 3)
            finally:
                os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
