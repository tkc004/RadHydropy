import tempfile
import unittest
import os
import hashlib
from pathlib import Path
from types import SimpleNamespace
from tests.parameter_fixtures import parameter_namespace

import numpy as np
import unyt
import yaml
import h5py

import radhydropy.io as rio
from radhydropy.units import CodeUnits
from radhydropy.radarray import RadArray
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

NONTRIVIAL_CODE_UNITS = CodeUnits.from_mapping(
    {
        'name': 'nontrivial_test_units',
        'InternalUnitSystem': {
            'UnitMass_in_cgs': 3.0,
            'UnitLength_in_cgs': 10.0,
            'UnitVelocity_in_cgs': 2.0,
            'UnitCurrent_in_cgs': 1.0,
            'UnitTemp_in_cgs': 4.0,
        },
    }
)


def _attach_proper_runtime_state(mesh, fluid):
    """Attach the numeric proper-code state used by the serializer tests."""
    boundary_proper_code = np.asarray(mesh.boundary.to_value(unyt.cm), dtype=float)
    width_proper_code = np.diff(boundary_proper_code)
    mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1]),
        boundary_proper_code=boundary_proper_code,
        width_proper_code=width_proper_code,
        area_proper_code=np.ones_like(width_proper_code),
        volume_proper_code=width_proper_code,
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
        rho_proper_code=density_proper_code,
        vel_proper_code=velocity_proper_code,
        pre_proper_code=fluid.pre_proper_code,
        temp_proper_code=temperature_proper_code,
        time_proper_code=fluid.time_proper_code,
        mu_dimensionless=getattr(fluid, "mu", None),
        xHI_dimensionless=getattr(fluid, "xHI", None),
    )
    return mesh, fluid


class Testing(unittest.TestCase):
    @staticmethod
    def _scalar_value(value):
        return float(np.ravel(np.asarray(value))[0])

    @staticmethod
    def _validation_snapshot():
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=2,
            time_proper_code=0.0,
            box_size_proper=2.0,
            CodeUnits=CODE_UNITS,
        )
        mesh = SimpleNamespace()
        mesh.geometry_state = MeshGeometryState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            x_proper_code=np.array([0.5, 1.5]),
            boundary_proper_code=np.array([0.0, 1.0, 2.0]),
            width_proper_code=np.ones(2),
            area_proper_code=np.ones(2),
            volume_proper_code=np.ones(2),
        )
        fluid = SimpleNamespace(
            rho_proper_code=np.ones(2),
            vel_proper_code=np.zeros(2),
            temp_proper_code=np.ones(2),
            pre_proper_code=np.ones(2),
            mu=np.ones(2),
        )
        fluid.runtime_state = FluidRuntimeState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            rho_proper_code=fluid.rho_proper_code,
            vel_proper_code=fluid.vel_proper_code,
            pre_proper_code=fluid.pre_proper_code,
            temp_proper_code=fluid.temp_proper_code,
            time_proper_code=0.0,
            mu_dimensionless=fluid.mu,
        )
        return SimpleNamespace(par=par, mesh=mesh, fluid=fluid)

    def test_loadhdf5_validates_nested_configuration(self):
        with self.assertRaisesRegex(TypeError, "nested configuration mapping"):
            rio.loadhdf5(None, "unused.hdf5")
        with self.assertRaisesRegex(ValueError, "contain a 'par' mapping"):
            rio.loadhdf5({}, "unused.hdf5")
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            rio.loadhdf5({"par": []}, "unused.hdf5")

    def test_hdf5_roundtrip_handles_scalar_header_quantities(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=0.0 * unyt.s,
            box_size_proper=3.0 * unyt.cm,
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
        self.assertEqual(self._scalar_value(loaded_par.box_size_proper), 3.0)
        self.assertEqual(self._scalar_value(loaded_fluid.time_proper_code), 0.0)
        self.assertAlmostEqual(loaded_par.cosmology_context.gamma, 5.0 / 3.0)
        self.assertEqual(loaded_par.cosmology_context.scale_factor, 1.0)
        self.assertEqual(
            loaded_par.cosmology_context.hubble_parameter_km_s_Mpc,
            0.0,
        )
        self.assertIsInstance(loaded_mesh.boundary_radarray, RadArray)
        self.assertIsInstance(loaded_fluid.rho_radarray, RadArray)
        self.assertIsInstance(loaded_fluid.vel_radarray, RadArray)
        self.assertEqual(loaded_mesh.boundary_radarray.field_spec.representation, "proper")
        self.assertEqual(loaded_fluid.rho_radarray.field_spec.quantity, "mass_density")
        np.testing.assert_allclose(loaded_fluid.rho_radarray.value, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(loaded_mesh.boundary_radarray.value, [0.0, 1.0, 2.0, 3.0])

    def test_hdf5_uses_canonical_code_state_dataset_names(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=2,
            time_code=0.0 * unyt.s,
            box_size_proper=2.0 * unyt.cm,
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

    def test_hdf5_canonical_fields_are_stored_as_code_values(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=2,
            time_proper_code=0.0,
            box_size_proper=20.0,
            CodeUnits=NONTRIVIAL_CODE_UNITS,
        )
        mesh = SimpleNamespace()
        mesh.geometry_state = MeshGeometryState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            x_proper_code=np.array([5.0, 15.0]),
            boundary_proper_code=np.array([0.0, 10.0, 20.0]),
            width_proper_code=np.array([10.0, 10.0]),
            area_proper_code=np.ones(2),
            volume_proper_code=np.array([10.0, 10.0]),
        )
        fluid = SimpleNamespace(
            rho_proper_code=np.array([2.0, 3.0]),
            vel_proper_code=np.array([4.0, 5.0]),
            temp_proper_code=np.array([6.0, 7.0]),
            pre_proper_code=np.array([8.0, 9.0]),
            mu=np.ones(2),
        )
        fluid.runtime_state = FluidRuntimeState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            rho_proper_code=fluid.rho_proper_code,
            vel_proper_code=fluid.vel_proper_code,
            pre_proper_code=fluid.pre_proper_code,
            temp_proper_code=fluid.temp_proper_code,
            time_proper_code=0.0,
            mu_dimensionless=fluid.mu,
        )
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = parameter_namespace(
            coordsys='cartesian', CodeUnits=NONTRIVIAL_CODE_UNITS
        )
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            with h5py.File(output.name, 'r') as handle:
                data = handle['Data']
                np.testing.assert_allclose(data['rho_proper_code'][()], [2.0, 3.0])
                np.testing.assert_allclose(data['vel_proper_code'][()], [4.0, 5.0])
                self.assertEqual(data['rho_proper_code'].attrs['storage_unit'], 'code')
                self.assertEqual(data['rho_proper_code'].attrs['quantity'], 'mass_density')
                np.testing.assert_array_equal(
                    data['rho_proper_code'].attrs['dimensions'],
                    (1, -3, 0, 0, 0),
                )
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        np.testing.assert_allclose(loaded_fluid.rho_proper_code, [2.0, 3.0])
        np.testing.assert_allclose(loaded_fluid.vel_proper_code, [4.0, 5.0])

    def test_hdf5_roundtrip_preserves_gas_angular_momentum_fields(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=0.0 * unyt.s,
            box_size_proper=3.0 * unyt.cm,
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
        self.assertIsInstance(
            loaded_fluid.specific_angular_momentum_radarray, RadArray
        )
        self.assertIsInstance(loaded_fluid.AngularMomentum_radarray, RadArray)
        self.assertFalse(hasattr(loaded_fluid, "specific_angular_momentum_code_radarray"))
        self.assertFalse(hasattr(loaded_fluid, "AngularMomentum_code_radarray"))

    def test_writehdf5_writes_all_par_values_into_header_attributes(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=1.5 * unyt.s,
            box_size_proper=3.0 * unyt.cm,
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

    def test_writehdf5_roundtrips_configuration_provenance(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=0.0 * unyt.s,
            box_size_proper=3.0 * unyt.cm,
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
        provenance = {
            'source_config_yaml': 'par:\n  simulation:\n    name: test\n',
            'effective_config': {
                'par': {
                    'simulation': {'final_time': 2.0 * unyt.s},
                },
                'initial_condition': {'rho_proper': 1.0 * unyt.g / unyt.cm**3},
                'example': {'name': 'io-test'},
            },
            'schema_version': 1,
            'source_config_filename': 'test.yaml',
            'git_commit': 'deadbeef',
            'git_dirty': False,
        }

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name, provenance=provenance)
            with h5py.File(output.name, 'r') as handle:
                stored = handle['Header']['Provenance']
                source_yaml = stored['source_config_yaml'][()].decode()
                effective_yaml = stored['effective_config_yaml'][()].decode()
                self.assertEqual(source_yaml, provenance['source_config_yaml'])
                self.assertIn('rho_proper:', effective_yaml)
                self.assertEqual(stored.attrs['schema_version'], 1)
                self.assertEqual(stored.attrs['source_config_filename'], 'test.yaml')
                self.assertEqual(stored.attrs['git_commit'], 'deadbeef')
                self.assertFalse(stored.attrs['git_dirty'])
                self.assertEqual(
                    stored.attrs['source_config_sha256'],
                    hashlib.sha256(source_yaml.encode()).hexdigest(),
                )
            rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(
            loaded_par.provenance['source_config_yaml'],
            provenance['source_config_yaml'],
        )
        self.assertEqual(loaded_par.provenance['schema_version'], 1)

        sim.par.provenance = loaded_par.provenance
        with tempfile.NamedTemporaryFile(suffix='.hdf5') as snapshot:
            rio.writehdf5(sim, snapshot.name)
            with h5py.File(snapshot.name, 'r') as handle:
                self.assertIn('Provenance', handle['Header'])
                self.assertEqual(
                    handle['Header']['Provenance'].attrs['effective_config_sha256'],
                    loaded_par.provenance['effective_config_sha256'],
                )

    def test_readhdf5_restores_header_attributes_and_code_units(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=1.5 * unyt.s,
            box_size_proper=3.0 * unyt.cm,
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
        self.assertEqual(self._scalar_value(loaded_par.box_size_proper), 3.0)

    def test_writehdf5_does_not_mutate_par_time(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=3,
            time_code=1.5 * unyt.s,
            box_size_proper=3.0 * unyt.cm,
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
            box_size_proper=np.array([3.0]) * unyt.cm,
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
            box_size_proper=np.array([3.0]) * unyt.cm,
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
            box_size_proper=np.array([3.0]) * unyt.cm,
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
            box_size_proper=np.array([3.0]) * unyt.cm,
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

    def test_readhdf5_rejects_code_unit_mismatch_before_mutation(self):
        par = parameter_namespace(
            coordsys='cartesian',
            nogrid=2,
            time_proper_code=0.0,
            box_size_proper=2.0,
            CodeUnits=CODE_UNITS,
        )
        mesh = SimpleNamespace()
        mesh.geometry_state = MeshGeometryState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            x_proper_code=np.array([0.5, 1.5]),
            boundary_proper_code=np.array([0.0, 1.0, 2.0]),
            width_proper_code=np.ones(2),
            area_proper_code=np.ones(2),
            volume_proper_code=np.ones(2),
        )
        fluid = SimpleNamespace(
            rho_proper_code=np.ones(2),
            vel_proper_code=np.zeros(2),
            temp_proper_code=np.ones(2),
            pre_proper_code=np.ones(2),
            mu=np.ones(2),
        )
        fluid.runtime_state = FluidRuntimeState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            rho_proper_code=fluid.rho_proper_code,
            vel_proper_code=fluid.vel_proper_code,
            pre_proper_code=fluid.pre_proper_code,
            temp_proper_code=fluid.temp_proper_code,
            time_proper_code=0.0,
            mu_dimensionless=fluid.mu,
        )
        sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
        loaded_par = parameter_namespace(
            coordsys='cartesian', CodeUnits=NONTRIVIAL_CODE_UNITS
        )
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            with self.assertRaises(rio.SnapshotConfigurationError):
                rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(
            loaded_par.CodeUnits.name, NONTRIVIAL_CODE_UNITS.name
        )

    def test_readhdf5_rejects_coordinate_system_mismatch_before_mutation(self):
        sim = self._validation_snapshot()
        loaded_par = parameter_namespace(
            coordsys='spherical', nogrid=2, CodeUnits=CODE_UNITS
        )
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            with self.assertRaisesRegex(
                rio.SnapshotConfigurationError, 'coordinate system'
            ):
                rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(loaded_par.simulation.coordinate_system, 'spherical')

    def test_readhdf5_rejects_grid_size_mismatch_before_mutation(self):
        sim = self._validation_snapshot()
        loaded_par = parameter_namespace(
            coordsys='cartesian', nogrid=3, CodeUnits=CODE_UNITS
        )
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            with self.assertRaisesRegex(
                rio.SnapshotConfigurationError, 'grid size'
            ):
                rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(loaded_par.mesh.grid_cells, 3)

    def test_readhdf5_rejects_representation_mismatch_before_mutation(self):
        sim = self._validation_snapshot()
        loaded_par = parameter_namespace(
            coordsys='cartesian', nogrid=2, CodeUnits=CODE_UNITS
        )
        loaded_par.coordinate_frame = 'physical'
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            with h5py.File(output.name, 'a') as handle:
                handle['Header'].attrs['CoordinateFrame'] = 'comoving'
            with self.assertRaisesRegex(
                rio.SnapshotConfigurationError, 'coordinate_frame'
            ):
                rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertEqual(loaded_par.coordinate_frame, 'physical')

    def test_readhdf5_rejects_cosmological_snapshot_for_noncosmological_runtime(self):
        sim = self._validation_snapshot()
        loaded_par = parameter_namespace(
            coordsys='cartesian', nogrid=2, CodeUnits=CODE_UNITS
        )
        loaded_par.cosmological_expansion = False
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()

        with tempfile.NamedTemporaryFile(suffix='.hdf5') as output:
            rio.writehdf5(sim, output.name)
            with h5py.File(output.name, 'a') as handle:
                handle['Header'].attrs['CosmologyType'] = 'lambda_cdm'
            with self.assertRaisesRegex(
                rio.SnapshotConfigurationError, 'cosmological'
            ):
                rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, output.name)

        self.assertFalse(loaded_par.cosmological_expansion)

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
                    box_size_proper=3.0 * unyt.cm,
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
                self.assertEqual(payload['initial_condition']['box_size_proper']['value'], 3.0)
                self.assertEqual(payload['initial_condition']['box_size_proper']['unit'], 'cm')
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
                    box_size_proper=3.0 * unyt.cm,
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
