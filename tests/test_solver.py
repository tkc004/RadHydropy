import unittest
from types import SimpleNamespace
from tests.parameter_fixtures import parameter_namespace
from unittest.mock import patch

import numpy as np
import unyt

from radhydropy.fluid import Fluid as RealFluid
from radhydropy.eos import EOS as RHEOS
from radhydropy.arrays import as_named_array
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
import radhydropy.thermo_chemistry as rtc
from radhydropy.units import CodeUnits
from radhydropy.thermo_networks.hydrogen import _fast_sync_state_to_fluid


class Par:
    def __init__(self, boundcond):
        self.boundcond = boundcond
        self.noghost = 2
        self.nogrid = 4
        self.CFL = 0.1
        self.dtmin = 1.0e-8 * unyt.s
        self.dtmax = 1.0 * unyt.s
        self.rho_inflow = 9.0 * unyt.g/unyt.cm**3
        self.vel_inflow = 8.0 * unyt.cm/unyt.s
        self.temp_inflow = 0.0 * unyt.K
        self.mu_inflow = 1.0
        self.rho_outflow = 7.0 * unyt.g/unyt.cm**3
        self.vel_outflow = 6.0 * unyt.cm/unyt.s
        self.temp_outflow = 0.0 * unyt.K
        self.mu_outflow = 1.0
        self.mesh = SimpleNamespace(ghost_cells=2, grid_cells=4)
        self.hydrodynamics = SimpleNamespace(
            CFL=0.1, gamma=5.0 / 3.0, riemann_solver='Rusanov', order=0,
        )
        self.boundary = SimpleNamespace(
            condition=boundcond,
            inflow_density=self.rho_inflow,
            inflow_velocity=self.vel_inflow,
            inflow_temperature=self.temp_inflow,
            inflow_mu=self.mu_inflow,
            outflow_density=self.rho_outflow,
            outflow_velocity=self.vel_outflow,
            outflow_temperature=self.temp_outflow,
            outflow_mu=self.mu_outflow,
        )
        self.radiation = SimpleNamespace(
            radiative_transfer=False,
            method='long_characteristics',
            direction=1,
            boundary_flux=0.0,
            source_photon_rate=0.0,
        )
        self.timestep = SimpleNamespace(dtmin=self.dtmin, dtmax=self.dtmax)
        self.units = SimpleNamespace(CodeUnits=None)


class EOS:
    gamma = 5.0/3.0
    EOStype = 'polytropic'

    def pressure(self, rho, temp, mu):
        return rho / (mu * unyt.mp) * unyt.kb * temp

    def temperature(self, rho, pressure, mu):
        pressure_over_rho = np.zeros_like(np.asarray(rho.value, dtype=float)) * unyt.K
        nonzero = rho != 0.0 * rho.units
        pressure_over_rho[nonzero] = (
            pressure[nonzero] / rho[nonzero] * (mu * unyt.mp) / unyt.kb
        ).to(unyt.K)
        return pressure_over_rho

    def thermal_energy_density(self, pressure):
        return pressure / (self.gamma - 1.0)

    def sound_speed(self, rho, pressure, temp=None, mu=None):
        pressure_over_rho = np.zeros_like(np.asarray(rho.value, dtype=float)) * (
            unyt.cm**2 / unyt.s**2
        )
        nonzero = rho != 0.0 * rho.units
        pressure_over_rho[nonzero] = (pressure[nonzero] / rho[nonzero]).to(
            unyt.cm**2 / unyt.s**2
        )
        soundspeed = np.sqrt(self.gamma * pressure_over_rho).to(unyt.cm / unyt.s)
        soundspeed[np.isnan(soundspeed)] = 0.0 * unyt.cm / unyt.s
        return soundspeed

    def total_energy_density(self, rho, vel, pressure):
        return 0.5 * rho * vel**2 + self.thermal_energy_density(pressure)

    def pressure_from_conserved(self, rho, vel, energy_density, temp=None, mu=None):
        return (energy_density - 0.5 * rho * vel**2) * (self.gamma - 1.0)

    def fluxes(self, rho, vel, pressure):
        Fmass = rho * vel
        qmass = rho
        Fmom = rho * vel * vel
        Fmom[np.logical_or(vel == 0.0, np.isnan(vel))] = 0.0 * rho[0] * vel[0]**2
        Fmom += pressure
        qmom = rho * vel
        FEn = vel * (self.gamma * pressure / (self.gamma - 1.0) + 0.5 * rho * vel**2)
        qEn = pressure / (self.gamma - 1.0) + rho * vel**2 * 0.5
        return Fmass, qmass, Fmom, qmom, FEn, qEn


class Fluid:
    def __init__(self):
        self.rho = np.arange(8, dtype=float) * unyt.g/unyt.cm**3
        self.vel = np.arange(10, 18, dtype=float) * unyt.cm/unyt.s
        self.pre = np.arange(20, 28, dtype=float) * unyt.dyn/unyt.cm**2
        self.eos = EOS()


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


def make_code_mesh(n=8):
    mesh = SimpleNamespace()
    mesh.boundary = np.arange(n + 1, dtype=float)
    mesh.vol = np.ones(n, dtype=float)
    mesh.xdelta = np.ones(n, dtype=float)
    mesh.area = np.ones(n, dtype=float)
    mesh.coordsys = 'cartesian'
    mesh.coordinate = np.arange(n, dtype=float) + 0.5
    return mesh


def make_code_fluid(n=8):
    fluid = RealFluid()
    fluid.eos = RHEOS('polytropic', gamma=5.0 / 3.0, code_units=CODE_UNITS)
    fluid.rho = np.ones(n, dtype=float)
    fluid.vel = np.zeros(n, dtype=float)
    fluid.temp = np.ones(n, dtype=float) * 1.0e4
    fluid.mu = np.ones(n, dtype=float)
    return fluid


def make_code_par(boundcond='Periodic'):
    par = Par(boundcond)
    par.CodeUnits = CODE_UNITS
    par.dtmin = 1.0e-8
    par.dtmax = 1.0
    par.hydrogen_chemistry = True
    par.hydrogen_mass_fraction = 1.0
    par.hydrogen_source_CFL = 0.1
    par.hydrogen_update_mu = False
    par.hydrogen_thermal_coupling = True
    par.hydrogen_recombination = True
    par.hydrogen_collisional_ionization = True
    par.hydrogen_radiation_field = False
    par.hydrogen_radiation_evolution = False
    par.hydrogen_sigma_gamma = 1.0e-18
    par.hydrogen_epsilon_gamma = 0.0
    par.radiative_transfer = False
    par.radiative_transfer_method = 'long_characteristics'
    par.radiative_transfer_boundary_flux = 0.0
    par.radiative_transfer_source_photon_rate = 0.0
    par.radiative_transfer_direction = 1
    par.mesh.grid_cells = par.nogrid
    par.mesh.ghost_cells = par.noghost
    par.hydrodynamics.CFL = par.CFL
    par.hydrodynamics.gamma = par.gamma if hasattr(par, 'gamma') else 5.0 / 3.0
    par.boundary.condition = par.boundcond
    par.timestep.dtmin = par.dtmin
    par.timestep.dtmax = par.dtmax
    par.units.CodeUnits = par.CodeUnits
    return par


class Mesh:
    def __init__(self, boundary=None):
        if boundary is None:
            boundary = np.linspace(0.0, 8.0, 9)
        self.boundary = boundary * unyt.cm
        self.vol = np.ones(len(boundary)-1) * unyt.cm**3
        self.xdelta = np.ones(len(boundary)-1) * unyt.cm
        self.area = np.arange(len(boundary)-1, dtype=float) * unyt.cm**2
        self.coordsys = 'cartesian'


class Testing(unittest.TestCase):
    def test_hllc_uniform_state_returns_physical_flux(self):
        rho = np.array([1.0, 2.0])
        velocity = np.array([0.75, -0.25])
        pressure = np.array([1.0, 0.5])
        flux, valid = Solver._hllc_flux(
            rho, velocity, pressure, rho, velocity, pressure, gamma=1.4
        )
        expected = np.stack((
            rho * velocity,
            rho * velocity**2 + pressure,
            velocity * (1.4 * pressure / 0.4 + 0.5 * rho * velocity**2),
        ))
        np.testing.assert_array_equal(valid, [True, True])
        np.testing.assert_allclose(flux, expected, rtol=1.0e-13, atol=1.0e-13)

    def test_hllc_marks_vacuum_state_for_rusanov_fallback(self):
        _, valid = Solver._hllc_flux(
            np.array([0.0]), np.array([0.0]), np.array([0.0]),
            np.array([1.0]), np.array([1.0]), np.array([1.0]), gamma=1.4
        )
        np.testing.assert_array_equal(valid, [False])

    def test_callreadhdf5_requires_code_units(self):
        sim = Rsim.__new__(Rsim)
        sim.par = parameter_namespace(ICfilename='dummy.hdf5')
        sim.mesh = SimpleNamespace()
        sim.fluid = SimpleNamespace()

        with self.assertRaisesRegex(ValueError, "configured code units"):
            sim.Callreadhdf5()

    @patch('radhydropy.rsim.rio.readhdf5')
    def test_callreadhdf5_rebuilds_eos_from_restored_header(self, readhdf5):
        restored_units = CodeUnits.from_mapping(
            {
                'name': 'restored_units',
                'InternalUnitSystem': {
                    'UnitMass_in_cgs': 2.0,
                    'UnitLength_in_cgs': 3.0,
                    'UnitVelocity_in_cgs': 4.0,
                    'UnitCurrent_in_cgs': 1.0,
                    'UnitTemp_in_cgs': 5.0,
                },
            }
        )
        sim = Rsim.__new__(Rsim)
        sim.par = parameter_namespace(
            ICfilename='dummy.hdf5',
            CodeUnits=CODE_UNITS,
            EOStype='polytropic',
            gamma=1.4,
            time=0.0,
        )
        sim.mesh = SimpleNamespace()
        sim.fluid = RealFluid()
        original_eos = RHEOS('polytropic', gamma=1.4, code_units=CODE_UNITS)
        sim.fluid.eos = original_eos
        sim.checkparams = lambda: None

        def restore_header(par, mesh, fluid, filename):
            par.EOStype = 'polytropic'
            par.gamma = 5.0 / 3.0
            par.CodeUnits = restored_units
            par.time = 7.0
            par.hydrodynamics.eos_type = 'polytropic'
            par.hydrodynamics.gamma = 5.0 / 3.0
            par.units.CodeUnits = restored_units
            par.simulation.current_time = 7.0
            par.mesh.grid_cells = 1

        readhdf5.side_effect = restore_header
        sim.Callreadhdf5()

        self.assertIsNot(sim.fluid.eos, original_eos)
        self.assertEqual(sim.fluid.eos.EOStype, 'polytropic')
        self.assertEqual(sim.fluid.eos.gamma, 5.0 / 3.0)
        self.assertIs(sim.fluid.eos.CodeUnits, restored_units)
        self.assertEqual(sim.fluid.time, 7.0)

    def test_open_boundary_fills_all_ghost_cells(self):
        fluid = Fluid()
        Solver().SetBoundary(None, fluid, Par('Open'))

        np.testing.assert_array_equal(fluid.rho[:2].value, [2.0, 2.0])
        np.testing.assert_array_equal(fluid.rho[-2:].value, [5.0, 5.0])
        np.testing.assert_array_equal(fluid.vel[:2].value, [12.0, 12.0])
        np.testing.assert_array_equal(fluid.vel[-2:].value, [15.0, 15.0])

    def test_optional_gas_angular_momentum_initializes_and_reconstructs(self):
        par = make_code_par('Periodic')
        par.gas_angular_momentum = True
        par.gas_specific_angular_momentum = 0.25
        mesh = make_code_mesh(n=12)
        mesh._par = par
        fluid = make_code_fluid(n=8)
        fluid.SetUpFluid(par, mesh=mesh)
        self.assertTrue(hasattr(fluid, 'specific_angular_momentum'))
        np.testing.assert_allclose(
            fluid.specific_angular_momentum[par.noghost:par.noghost + par.nogrid],
            0.25,
        )

        solver = Solver()
        solver.SetBoundary(mesh, fluid, par)
        solver.SetConserved(mesh, fluid)
        active = slice(par.noghost, par.noghost + par.nogrid)
        np.testing.assert_allclose(
            fluid.AngularMomentum[active], 0.25 * fluid.Mass[active]
        )
        fluid.specific_angular_momentum[:] = -1.0
        solver.SetPrimitive(mesh, fluid, par=par)
        np.testing.assert_allclose(
            fluid.specific_angular_momentum[par.noghost:par.noghost + par.nogrid],
            0.25,
        )

    def test_gas_angular_momentum_boundaries_copy_without_sign_change(self):
        fluid = Fluid()
        fluid.specific_angular_momentum = np.arange(8, dtype=float) + 1.0
        solver = Solver()
        solver.SetBoundary(None, fluid, Par('Periodic'))
        np.testing.assert_array_equal(
            fluid.specific_angular_momentum[:2], [5.0, 6.0]
        )
        np.testing.assert_array_equal(
            fluid.specific_angular_momentum[-2:], [3.0, 4.0]
        )

        fluid.specific_angular_momentum = np.arange(8, dtype=float) + 1.0
        solver.SetBoundary(None, fluid, Par('Reflecting'))
        np.testing.assert_array_equal(
            fluid.specific_angular_momentum[:2], [4.0, 3.0]
        )
        np.testing.assert_array_equal(
            fluid.specific_angular_momentum[-2:], [6.0, 5.0]
        )

    def test_gas_angular_momentum_uses_mass_flux_and_conserves_periodically(self):
        par = make_code_par('Periodic')
        par.gas_angular_momentum = True
        mesh = make_code_mesh(n=12)
        mesh._par = par
        fluid = make_code_fluid(n=8)
        fluid.vel[:] = 0.25
        fluid.SetUpFluid(par, mesh=mesh)
        fluid.specific_angular_momentum[:] = (
            np.arange(len(fluid.specific_angular_momentum), dtype=float) + 1.0
        )

        solver = Solver()
        solver.SetBoundary(mesh, fluid, par)
        solver.SetConserved(mesh, fluid)
        initial_total = np.sum(np.asarray(fluid.AngularMomentum, dtype=float))
        solver.GetTimeStep(mesh, fluid, par)
        solver.SetInterFaceFlux(mesh, fluid, par.boundcond, order=0)

        expected_flux = np.where(
            np.asarray(fluid.Mass.flux, dtype=float) >= 0.0,
            np.asarray(fluid.specific_angular_momentum.L, dtype=float),
            np.asarray(fluid.specific_angular_momentum.R, dtype=float),
        ) * np.asarray(fluid.Mass.flux, dtype=float)
        np.testing.assert_allclose(
            np.asarray(fluid.AngularMomentum.flux, dtype=float), expected_flux
        )
        solver.AddFluxes(1.0e-3, mesh, fluid, par.boundcond)
        np.testing.assert_allclose(
            np.sum(np.asarray(fluid.AngularMomentum, dtype=float)),
            initial_total,
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    def test_gas_angular_momentum_order_one_uses_bounded_donor_flux(self):
        par = make_code_par('Periodic')
        par.gas_angular_momentum = True
        par.positivity_preserving = False
        mesh = make_code_mesh(n=12)
        mesh._par = par
        fluid = make_code_fluid(n=8)
        fluid.vel[:] = 0.25
        fluid.SetUpFluid(par, mesh=mesh)
        fluid.specific_angular_momentum[:] = np.linspace(-1.0, 1.0, len(fluid.rho))

        solver = Solver()
        solver.SetBoundary(mesh, fluid, par)
        solver.SetConserved(mesh, fluid)
        solver.GetTimeStep(mesh, fluid, par)
        solver.SetInterFaceFlux(mesh, fluid, par.boundcond, order=1)

        mass_flux = np.asarray(fluid.Mass.flux, dtype=float)
        expected = mass_flux * np.asarray(
            fluid.angular_momentum_face, dtype=float
        )
        np.testing.assert_allclose(
            np.asarray(fluid.AngularMomentum.flux, dtype=float), expected
        )
        self.assertTrue(
            np.any(np.asarray(fluid.angular_momentum_face, dtype=float) < 0.0)
        )

    def test_gas_angular_momentum_reconstruction_is_locally_bounded(self):
        par = make_code_par('Periodic')
        par.gas_angular_momentum = True
        par.positivity_preserving = False
        mesh = make_code_mesh(n=12)
        mesh._par = par
        fluid = make_code_fluid(n=8)
        fluid.SetUpFluid(par, mesh=mesh)
        fluid.specific_angular_momentum[:] = np.array(
            [0.02, 0.20, 0.02, -0.10, 0.02, 0.15, 0.02, -0.05,
             0.02, 0.18, 0.02, -0.08]
        )

        solver = Solver()
        solver.SetBoundary(mesh, fluid, par)
        solver.SetConserved(mesh, fluid)
        solver.GetTimeStep(mesh, fluid, par)
        solver.SetInterFaceFlux(mesh, fluid, par.boundcond, order=1)

        j_left = np.asarray(fluid.specific_angular_momentum.L, dtype=float)
        j_right = np.asarray(fluid.specific_angular_momentum.R, dtype=float)
        j_min = np.minimum(j_left, j_right)
        j_max = np.maximum(j_left, j_right)
        for face_state in (
            fluid.specific_angular_momentum.L.first,
            fluid.specific_angular_momentum.R.first,
        ):
            np.testing.assert_array_less(
                np.asarray(face_state, dtype=float), j_max + 1.0e-14
            )
            np.testing.assert_array_less(
                j_min - 1.0e-14, np.asarray(face_state, dtype=float)
            )

    def test_spherical_origin_zeroes_optional_fluxes_after_construction(self):
        par = make_code_par('Periodic')
        par.gas_angular_momentum = True
        par.gas_rotational_energy = True
        mesh = make_code_mesh(n=12)
        mesh.coordsys = 'spherical'
        mesh._par = par
        fluid = make_code_fluid(n=8)
        fluid.SetUpFluid(par, mesh=mesh)
        fluid.specific_angular_momentum[:] = np.linspace(
            0.1, 1.0, len(fluid.specific_angular_momentum)
        )

        solver = Solver()
        solver.SetBoundary(mesh, fluid, par)
        solver.SetConserved(mesh, fluid)
        solver.GetTimeStep(mesh, fluid, par)
        solver.SetInterFaceFlux(mesh, fluid, par.boundcond, order=1)

        origin = np.where(np.asarray(mesh.boundary[:-1]) == 0.0)[0][0]
        self.assertEqual(float(fluid.Mass.flux[origin]), 0.0)
        self.assertEqual(float(fluid.Mom.flux[origin]), 0.0)
        self.assertEqual(float(fluid.Energy.flux[origin]), 0.0)
        self.assertEqual(float(fluid.AngularMomentum.flux[origin]), 0.0)
        self.assertEqual(float(fluid.rotational_energy_flux[origin]), 0.0)

    def test_reflecting_positive_rmin_preserves_j_and_uses_wall_flux(self):
        par = make_code_par('Reflecting')
        par.gas_angular_momentum = True
        par.gas_rotational_energy = True
        mesh = make_code_mesh(n=12)
        mesh.coordsys = 'spherical'
        mesh.boundary = np.arange(1.0, 14.0)
        mesh.coordinate = 0.5 * (mesh.boundary[:-1] + mesh.boundary[1:])
        mesh._par = par
        fluid = make_code_fluid(n=8)
        fluid.SetUpFluid(par, mesh=mesh)
        fluid.vel[:] = 0.2
        fluid.specific_angular_momentum[:] = np.linspace(
            0.2, 1.0, len(fluid.specific_angular_momentum)
        )
        active = slice(par.noghost, par.noghost + par.nogrid)
        active_j = np.asarray(fluid.specific_angular_momentum[active]).copy()

        solver = Solver()
        solver.SetBoundary(mesh, fluid, par)
        np.testing.assert_allclose(
            fluid.specific_angular_momentum[:par.noghost], active_j[:par.noghost][::-1]
        )
        solver.SetConserved(mesh, fluid)
        solver.GetTimeStep(mesh, fluid, par)
        solver.SetInterFaceFlux(mesh, fluid, par.boundcond, order=1)

        inner_face = par.noghost
        self.assertGreater(mesh.boundary[inner_face], 0.0)
        self.assertEqual(float(fluid.Mass.flux[inner_face]), 0.0)
        self.assertEqual(float(fluid.AngularMomentum.flux[inner_face]), 0.0)
        self.assertEqual(float(fluid.rotational_energy_flux[inner_face]), 0.0)
        self.assertGreater(float(fluid.Mom.flux[inner_face]), 0.0)

    def test_origin_adjacent_rotating_cells_are_regular_and_signed(self):
        # The first active cell is adjacent to the exact spherical origin.
        par = make_code_par('Reflecting')
        par.noghost = 1
        par.mesh.ghost_cells = 1
        par.nogrid = 4
        par.mesh.grid_cells = 4
        par.gas_angular_momentum = True
        par.gas_rotational_energy = True
        par.positivity_preserving = False
        mesh = SimpleNamespace(
            coordsys='spherical',
            boundary=np.arange(-1.0, 6.0),
            coordinate=np.arange(-0.5, 5.5),
            vol=np.ones(6),
            xdelta=np.ones(6),
            area=np.ones(6),
            _par=par,
        )
        fluid = make_code_fluid(n=6)
        fluid.rho = as_named_array(fluid.rho)
        fluid.vel = as_named_array(fluid.vel)
        fluid.vel[:] = 0.2
        fluid.specific_angular_momentum = as_named_array(
            -0.4 * mesh.coordinate**1
        )
        fluid.SetPressure()
        fluid.pre = as_named_array(fluid.pre)
        # A regular signed profile has negative angular momentum here;
        # reflection must copy j, not reverse its sign.
        solver = Solver()
        solver.SetBoundary(mesh, fluid, par)
        self.assertEqual(float(fluid.specific_angular_momentum[0]), -0.2)
        self.assertEqual(float(fluid.vel[0]), -0.2)

        solver.SetConserved(mesh, fluid)
        solver.GetTimeStep(mesh, fluid, par)
        solver.SetInterFaceFlux(mesh, fluid, par.boundcond, order=1)
        origin_face = int(np.where(mesh.boundary[:-1] == 0.0)[0][0])
        self.assertEqual(origin_face, par.noghost)
        self.assertEqual(float(fluid.AngularMomentum.flux[origin_face]), 0.0)
        self.assertEqual(float(fluid.rotational_energy_flux[origin_face]), 0.0)

        first_active = par.noghost
        rotational_energy = solver._rotational_energy_density(
            mesh, fluid, par
        )
        self.assertTrue(np.isfinite(rotational_energy[first_active]))
        self.assertGreater(rotational_energy[first_active], 0.0)

        # Source updates are cell-centred. An explicitly represented r=0 cell
        # receives no centrifugal acceleration, while the adjacent positive-r
        # cell does receive the regular source.
        source_mesh = SimpleNamespace(
            coordsys='spherical', coordinate=np.array([0.0, 0.75]),
            vol=np.ones(2),
        )
        source_fluid = SimpleNamespace(
            rho=np.ones(2), Mass=np.ones(2), Mom=np.zeros(2),
            Energy=np.ones(2), AngularMomentum=np.array([1.0, 0.75]),
        )
        source_par = SimpleNamespace(
            noghost=0, nogrid=2, gas_angular_momentum=True,
            gas_rotational_energy=True, energy_diagnostics=True,
        )
        source_par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=2)
        solver.ApplyGravity(0.1, source_mesh, source_fluid, source_par)
        self.assertEqual(float(source_fluid.Mom[0]), 0.0)
        self.assertGreater(float(source_fluid.Mom[1]), 0.0)

    def test_optional_gravity_potential_energy_is_source_balanced(self):
        par = make_code_par('Periodic')
        par.gravity_potential_energy = True
        par.externalgravity = True
        par.gravity = SimpleNamespace(
            potential_on=lambda coordinate: -np.asarray(coordinate, dtype=float),
            acceleration_on_mesh=lambda mesh, rho=None, par=None:
                np.ones_like(np.asarray(mesh.coordinate, dtype=float)),
        )
        mesh = make_code_mesh()
        mesh._par = par
        fluid = make_code_fluid()
        fluid.pre = fluid.eos.pressure(fluid.rho, fluid.temp, fluid.mu)
        solver = Solver()
        solver.SetConserved(mesh, fluid)

        np.testing.assert_allclose(
            np.asarray(fluid.GravitationalPotentialEnergy, dtype=float),
            np.asarray(fluid.Mass, dtype=float) * (-mesh.coordinate),
        )
        initial_total = (
            np.asarray(fluid.Energy, dtype=float)
            + np.asarray(fluid.GravitationalPotentialEnergy, dtype=float)
        )
        solver.ApplyGravity(1.0e-3, mesh, fluid, par)
        np.testing.assert_allclose(
            np.asarray(fluid.Energy, dtype=float)
            + np.asarray(fluid.GravitationalPotentialEnergy, dtype=float),
            initial_total,
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    def test_optional_gravity_potential_energy_uses_face_mass_flux(self):
        par = make_code_par('Periodic')
        par.gravity_potential_energy = True
        par.positivity_preserving = False
        par.externalgravity = True
        par.gravity = SimpleNamespace(
            potential_on=lambda coordinate: -np.asarray(coordinate, dtype=float),
            acceleration_on_mesh=lambda mesh, rho=None, par=None:
                np.zeros_like(np.asarray(mesh.coordinate, dtype=float)),
        )
        mesh = make_code_mesh()
        mesh._par = par
        fluid = make_code_fluid()
        fluid.pre = fluid.eos.pressure(fluid.rho, fluid.temp, fluid.mu)
        solver = Solver()
        solver.SetConserved(mesh, fluid)
        old_potential_energy = np.asarray(
            fluid.GravitationalPotentialEnergy, dtype=float
        ).copy()
        mass_flux = np.linspace(0.1, 0.8, len(fluid.Mass))
        fluid.Mass.flux = as_named_array(mass_flux)
        fluid.Mom.flux = as_named_array(np.zeros_like(mass_flux))
        fluid.Energy.flux = as_named_array(np.zeros_like(mass_flux))
        dt = 1.0e-3
        solver.AddFluxes(dt, mesh, fluid, par)
        face_potential = -np.asarray(mesh.boundary[:-1], dtype=float)
        expected = old_potential_energy + dt * (
            face_potential * mass_flux
            - np.roll(face_potential * mass_flux, -1)
        )
        np.testing.assert_allclose(
            np.asarray(fluid.GravitationalPotentialEnergy, dtype=float), expected
        )

    def test_thermochemistry_sync_preserves_rotational_energy(self):
        par = make_code_par('Periodic')
        par.gas_angular_momentum = True
        par.gas_rotational_energy = True
        fluid = SimpleNamespace(
            Mass=np.array([2.0]),
            Energy=np.array([0.0]),
            rho=np.array([1.0]),
            temp=np.array([1.0]),
            xHI=np.array([1.0]),
            mu=np.array([1.0]),
            eos=SimpleNamespace(gamma=5.0 / 3.0),
            pre=np.array([1.0]),
        )
        internal = 3.0
        kinetic = 0.5
        rotational = 0.75
        state = {
            'interior': slice(0, 1),
            'active': np.array([True]),
            'rho_g_cm3': np.array([1.0]),
            'xHI': np.array([1.0]),
            'mu': np.array([1.0]),
            'temperature_K': np.array([1.0]),
            'source_temperature_factor': 1.0,
            'thermal_coupling': True,
            'hydrogen_update_mu': False,
            'gamma': 5.0 / 3.0,
            'specific_total_energy_erg_g': np.array([internal + kinetic]),
            'specific_energy_erg_g': np.array([internal]),
            'specific_kinetic_energy_erg_g': np.array([kinetic]),
            'specific_kinetic_energy_supercomoving_erg_g': np.array([kinetic]),
            'specific_rotational_energy_code': np.array([rotational]),
        }

        _fast_sync_state_to_fluid(state, fluid, par)

        np.testing.assert_allclose(
            fluid.Energy,
            fluid.Mass * (internal + kinetic + rotational),
        )

    def test_optional_rotational_energy_is_added_and_removed_for_pressure(self):
        par = make_code_par('Periodic')
        par.gas_angular_momentum = True
        par.gas_rotational_energy = True
        par.dual_energy = True
        mesh = make_code_mesh(n=12)
        mesh.coordsys = 'spherical'
        mesh._par = par
        mesh.coordinate = np.arange(12, dtype=float) + 0.5
        fluid = make_code_fluid(n=8)
        fluid.SetUpFluid(par, mesh=mesh)
        fluid.specific_angular_momentum[:] = 0.5

        solver = Solver()
        solver.SetBoundary(mesh, fluid, par)
        solver.SetConserved(mesh, fluid)
        active = slice(par.noghost, par.noghost + par.nogrid)
        rho = np.asarray(fluid.rho[active], dtype=float)
        radius = np.asarray(mesh.coordinate[active], dtype=float)
        volume = np.asarray(mesh.vol[active], dtype=float)
        hydro_energy = np.asarray(
            fluid.eos.total_energy_density(
                fluid.rho[active], fluid.vel[active], fluid.pre[active]
            ) * volume,
            dtype=float,
        )
        expected_rotational = 0.5 * rho * 0.5**2 / radius**2 * volume
        np.testing.assert_allclose(
            np.asarray(fluid.Energy[active], dtype=float),
            hydro_energy + expected_rotational,
        )
        expected_internal = np.asarray(
            fluid.eos.thermal_energy_density(fluid.pre[active])
            * volume,
            dtype=float,
        )
        np.testing.assert_allclose(
            np.asarray(fluid.InternalEnergy[active], dtype=float),
            expected_internal,
        )
        pressure_before = np.asarray(fluid.pre[active], dtype=float).copy()
        solver.SetPrimitive(mesh, fluid, par=par)
        np.testing.assert_allclose(
            np.asarray(fluid.pre[active], dtype=float), pressure_before
        )
        solver.GetTimeStep(mesh, fluid, par)
        solver.SetInterFaceFlux(mesh, fluid, par.boundcond, order=0)
        radius_face = np.abs(np.asarray(mesh.boundary[:-1], dtype=float))
        expected_rotational_flux = np.zeros_like(radius_face)
        valid_face = radius_face > 0.0
        expected_rotational_flux[valid_face] = (
            0.5 * np.asarray(fluid.specific_angular_momentum.L, dtype=float)[valid_face]**2
            / radius_face[valid_face]**2
            * np.asarray(fluid.Mass.flux, dtype=float)[valid_face]
        )
        np.testing.assert_allclose(
            np.asarray(fluid.rotational_energy_flux, dtype=float),
            expected_rotational_flux,
        )
        old_momentum = np.asarray(fluid.Mom, dtype=float).copy()
        old_energy = np.asarray(fluid.Energy, dtype=float).copy()
        old_mass = np.asarray(fluid.Mass, dtype=float).copy()
        old_angular = np.asarray(fluid.AngularMomentum, dtype=float).copy()
        old_kinetic = 0.5 * old_momentum**2 / old_mass
        old_rotational = 0.5 * old_angular**2 / (
            old_mass * np.asarray(mesh.coordinate, dtype=float)**2
        )
        solver.ApplyGravity(1.0e-3, mesh, fluid, par)
        expected_acceleration = 0.5**2 / np.asarray(mesh.coordinate, dtype=float)**3
        np.testing.assert_allclose(
            np.asarray(fluid.Mom[active], dtype=float),
            old_momentum[active]
            + np.asarray(fluid.Mass[active], dtype=float)
            * expected_acceleration[active] * 1.0e-3,
        )
        centrifugal_work = 0.5 * (
            old_momentum
            + (
                old_momentum
                + np.asarray(fluid.Mass, dtype=float)
                * expected_acceleration * 1.0e-3
            )
        ) * expected_acceleration * 1.0e-3
        np.testing.assert_allclose(
            np.asarray(fluid.Energy, dtype=float),
            old_energy,
        )
        new_momentum = np.asarray(fluid.Mom, dtype=float)
        new_kinetic = 0.5 * new_momentum**2 / old_mass
        new_rotational = 0.5 * old_angular**2 / (
            old_mass * np.asarray(mesh.coordinate, dtype=float)**2
        )
        np.testing.assert_allclose(new_rotational, old_rotational)
        np.testing.assert_allclose(
            (old_energy - new_kinetic - new_rotational)[active],
            (
                old_energy - old_kinetic - old_rotational
                - (new_kinetic - old_kinetic)
            )[active],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            solver.last_centrifugal_work,
            np.sum(centrifugal_work[active]),
        )

    def test_manufactured_spherical_hydro_step_transports_j_conservatively(self):
        par = make_code_par('Periodic')
        par.gas_angular_momentum = True
        par.gas_rotational_energy = True
        par.positivity_preserving = False
        mesh = make_code_mesh(n=12)
        mesh.coordsys = 'spherical'
        mesh._par = par
        mesh.coordinate = np.arange(12, dtype=float) + 1.0
        fluid = make_code_fluid(n=8)
        fluid.vel[:] = 0.1 + 0.01 * np.arange(8)
        fluid.SetUpFluid(par, mesh=mesh)
        fluid.specific_angular_momentum[:] = np.linspace(
            0.2, 0.9, len(fluid.specific_angular_momentum)
        )
        solver = Solver()
        solver.SetBoundary(mesh, fluid, par)
        solver.SetConserved(mesh, fluid)
        solver.GetTimeStep(mesh, fluid, par)
        solver.SetInterFaceFlux(mesh, fluid, par.boundcond, order=0)

        old_mass = np.asarray(fluid.Mass, dtype=float).copy()
        old_momentum = np.asarray(fluid.Mom, dtype=float).copy()
        old_energy = np.asarray(fluid.Energy, dtype=float).copy()
        old_angular = np.asarray(fluid.AngularMomentum, dtype=float).copy()
        area = np.asarray(mesh.area, dtype=float)
        dt = 1.0e-4
        mass_flux_area = np.asarray(fluid.Mass.flux, dtype=float) * area
        momentum_flux_area = np.asarray(fluid.Mom.flux, dtype=float) * area
        energy_flux_area = np.asarray(fluid.Energy.flux, dtype=float) * area
        angular_flux_area = (
            np.asarray(fluid.AngularMomentum.flux, dtype=float) * area
        )
        expected_mass = old_mass + dt * (
            mass_flux_area - np.roll(mass_flux_area, -1)
        )
        expected_momentum = old_momentum + dt * (
            momentum_flux_area - np.roll(momentum_flux_area, -1)
            + np.asarray(fluid.pre, dtype=float) * (
                np.roll(area, -1) - area
            )
        )
        expected_energy = old_energy + dt * (
            energy_flux_area - np.roll(energy_flux_area, -1)
        )
        expected_angular = old_angular + dt * (
            angular_flux_area - np.roll(angular_flux_area, -1)
        )
        solver.AddFluxes(dt, mesh, fluid, par.boundcond)
        np.testing.assert_allclose(np.asarray(fluid.Mass, dtype=float), expected_mass)
        np.testing.assert_allclose(np.asarray(fluid.Mom, dtype=float), expected_momentum)
        np.testing.assert_allclose(np.asarray(fluid.Energy, dtype=float), expected_energy)
        np.testing.assert_allclose(
            np.asarray(fluid.AngularMomentum, dtype=float), expected_angular
        )
        # Rotational energy is exchanged through the conservative transport
        # flux. It must not receive a second centrifugal-work source.
        np.testing.assert_allclose(
            np.sum(np.asarray(fluid.Energy, dtype=float)),
            np.sum(old_energy),
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        radius = np.asarray(mesh.coordinate, dtype=float)
        old_rotational = 0.5 * old_angular**2 / (old_mass * radius**2)
        new_rotational = 0.5 * np.asarray(fluid.AngularMomentum, dtype=float)**2 / (
            np.asarray(fluid.Mass, dtype=float) * radius**2
        )
        self.assertGreater(np.max(np.abs(new_rotational - old_rotational)), 0.0)

    def test_periodic_boundary_wraps_interior(self):
        fluid = Fluid()
        Solver().SetBoundary(None, fluid, Par('Periodic'))

        np.testing.assert_array_equal(fluid.rho[:2].value, [4.0, 5.0])
        np.testing.assert_array_equal(fluid.rho[-2:].value, [2.0, 3.0])

    def test_periodic_boundary_wraps_neutral_fraction(self):
        fluid = Fluid()
        fluid.xHI = np.arange(8, dtype=float) / 10.0

        Solver().SetBoundary(None, fluid, Par('Periodic'))

        np.testing.assert_array_equal(fluid.xHI[:2], [0.4, 0.5])
        np.testing.assert_array_equal(fluid.xHI[-2:], [0.2, 0.3])

    def test_periodic_boundary_wraps_photon_number_density(self):
        fluid = Fluid()
        fluid.ngamma = np.arange(8, dtype=float) / unyt.cm**3

        Solver().SetBoundary(None, fluid, Par('Periodic'))

        np.testing.assert_array_equal(fluid.ngamma[:2].value, [4.0, 5.0])
        np.testing.assert_array_equal(fluid.ngamma[-2:].value, [2.0, 3.0])

    def test_reflecting_boundary_reverses_velocity(self):
        fluid = Fluid()
        Solver().SetBoundary(None, fluid, Par('Reflecting'))

        np.testing.assert_array_equal(fluid.rho[:2].value, [3.0, 2.0])
        np.testing.assert_array_equal(fluid.rho[-2:].value, [5.0, 4.0])
        np.testing.assert_array_equal(fluid.vel[:2].value, [-13.0, -12.0])
        np.testing.assert_array_equal(fluid.vel[-2:].value, [-15.0, -14.0])

    def test_open_spherical_boundary_uses_center_symmetry(self):
        fluid = Fluid()
        Solver().SetBoundary(Mesh(), fluid, Par('OpenSph'))

        np.testing.assert_array_equal(fluid.rho[:2].value, [3.0, 2.0])
        np.testing.assert_array_equal(fluid.rho[-2:].value, [5.0, 5.0])
        np.testing.assert_array_equal(fluid.vel[:2].value, [-13.0, -12.0])
        np.testing.assert_array_equal(fluid.vel[-2:].value, [15.0, 15.0])

    def test_open_spherical_boundary_skips_origin_cell_when_mesh_straddles_zero(self):
        fluid = Fluid()
        mesh = Mesh(np.linspace(-2.5, 5.5, 9))
        Solver().SetBoundary(mesh, fluid, Par('OpenSph'))

        np.testing.assert_array_equal(fluid.rho[:2].value, [4.0, 3.0])
        np.testing.assert_array_equal(fluid.rho[-2:].value, [5.0, 5.0])
        np.testing.assert_array_equal(fluid.vel[:2].value, [-14.0, -13.0])
        np.testing.assert_array_equal(fluid.vel[-2:].value, [15.0, 15.0])

    def test_inflow_spherical_boundary_sets_inner_symmetry_and_inflow_state(self):
        fluid = Fluid()
        fluid.xHI = np.arange(8, dtype=float) / 10.0
        fluid.ngamma = np.arange(8, dtype=float) / unyt.cm**3
        active_velocity = fluid.vel[2:6].copy()
        par = Par('InflowSph')
        par.CodeUnits = CODE_UNITS
        par.hydrogen_xHI_inflow = 0.25
        par.hydrogen_ngamma_inflow = 1.5 / unyt.cm**3

        Solver().SetBoundary(Mesh(), fluid, par)

        np.testing.assert_array_equal(fluid.rho[:2].value, [3.0, 2.0])
        np.testing.assert_array_equal(fluid.vel[:2].value, [-13.0, -12.0])
        np.testing.assert_array_equal(fluid.vel[2:6], active_velocity)
        np.testing.assert_array_equal(fluid.rho[-2:].value, [9.0, 9.0])
        np.testing.assert_array_equal(fluid.vel[-2:].value, [8.0, 8.0])
        np.testing.assert_array_equal(fluid.xHI[-2:], [0.25, 0.25])
        np.testing.assert_array_equal(fluid.ngamma[-2:].value, [1.5, 1.5])

    def test_outflow_spherical_boundary_sets_inner_outflow_state(self):
        fluid = Fluid()
        fluid.xHI = np.arange(8, dtype=float) / 10.0
        fluid.ngamma = np.arange(8, dtype=float) / unyt.cm**3
        par = Par('OutflowSph')
        par.CodeUnits = CODE_UNITS
        par.hydrogen_xHI_outflow = 0.75
        par.hydrogen_ngamma_outflow = 2.5 / unyt.cm**3

        Solver().SetBoundary(Mesh(), fluid, par)

        np.testing.assert_array_equal(fluid.rho[:2].value, [7.0, 7.0])
        np.testing.assert_array_equal(fluid.vel[:2].value, [6.0, 6.0])
        np.testing.assert_array_equal(fluid.xHI[:2], [0.75, 0.75])
        np.testing.assert_array_equal(fluid.ngamma[:2].value, [2.5, 2.5])
        np.testing.assert_array_equal(fluid.rho[-2:].value, [5.0, 5.0])
        np.testing.assert_array_equal(fluid.vel[-2:].value, [15.0, 15.0])

    def test_set_primitive_handles_zero_mass(self):
        fluid = Fluid()
        fluid.Mass = np.array([1.0, 0.0, 2.0]) * unyt.g
        fluid.Mom = np.array([2.0, 1.0, 0.0]) * unyt.g*unyt.cm/unyt.s
        fluid.Energy = np.array([10.0, 5.0, 1.0]) * unyt.g*unyt.cm**2/unyt.s**2

        Solver().SetPrimitive(Mesh(np.linspace(0.0, 3.0, 4)), fluid)

        self.assertEqual(fluid.vel[1], 0.0 * unyt.cm/unyt.s)
        self.assertFalse(np.any(np.isnan(fluid.rho)))
        self.assertFalse(np.any(np.isnan(fluid.vel)))
        self.assertFalse(np.any(np.isnan(fluid.pre)))

    def test_spherical_uniform_pressure_does_not_create_momentum(self):
        mesh = Mesh()
        mesh.coordsys = 'spherical'
        fluid = Fluid()
        fluid.rho = np.ones(8) * unyt.g/unyt.cm**3
        fluid.vel = np.zeros(8) * unyt.cm/unyt.s
        fluid.pre = np.ones(8) * unyt.dyn/unyt.cm**2
        fluid.time = 0.0 * unyt.s
        fluid.Mass = np.ones(8) * unyt.g
        fluid.Mom = np.zeros(8) * unyt.g*unyt.cm/unyt.s
        fluid.Energy = np.ones(8) * unyt.g*unyt.cm**2/unyt.s**2
        fluid.Mass.flux = np.zeros(8) * unyt.g/unyt.cm**2/unyt.s
        fluid.Mom.flux = np.ones(8) * unyt.dyn/unyt.cm**2
        fluid.Energy.flux = np.zeros(8) * unyt.g/unyt.s**3

        Solver().AddFluxes(1.0*unyt.s, mesh, fluid, 'OpenSph')

        np.testing.assert_allclose(fluid.Mom.value, np.zeros(8))

    def test_spherical_first_active_cell_velocity_is_not_projected(self):
        mesh = make_code_mesh(4)
        mesh.coordsys = 'spherical'
        mesh.boundary[0] = 0.0
        mesh.coordinate[0] = 0.5
        fluid = make_code_fluid(4)
        fluid.rho[:] = 1.0
        fluid.vel[:] = 0.0
        fluid.vel[0] = 3.0
        fluid.mu[:] = 1.0
        fluid.pre = as_named_array(np.ones(4))
        fluid.pre[:] = 1.0
        fluid.Mass = as_named_array(fluid.rho * mesh.vol)
        fluid.Mom = as_named_array(fluid.rho * fluid.vel * mesh.vol)
        fluid.Energy = as_named_array(
            fluid.eos.total_energy_density(
                fluid.rho, fluid.vel, fluid.pre
            ) * mesh.vol
        )
        total_energy = float(fluid.Energy[0])

        solver = Solver()
        solver.SetPrimitive(mesh, fluid)
        solver.SetConserved(mesh, fluid)

        self.assertEqual(float(fluid.vel[0]), 3.0)
        self.assertAlmostEqual(float(fluid.Energy[0]), total_energy)
        self.assertAlmostEqual(float(fluid.pre[0]), 1.0)

    def test_positivity_limiter_preserves_mass_and_internal_energy(self):
        par = make_code_par()
        par.noghost = 0
        par.mesh.ghost_cells = 0
        par.nogrid = 4
        par.mesh.grid_cells = 4
        par.positivity_preserving = True
        mesh = make_code_mesh(4)
        mesh._par = par
        fluid = SimpleNamespace(
            Mass=as_named_array(np.ones(4)),
            Mom=as_named_array(np.zeros(4)),
            Energy=as_named_array(np.ones(4)),
            time=0.0,
        )
        fluid.Mass.flux = as_named_array(np.array([0.0, 3.0, 0.0, 0.0]))
        fluid.Mom.flux = as_named_array(np.zeros(4))
        fluid.Energy.flux = as_named_array(np.array([0.0, 3.0, 0.0, 0.0]))

        Solver().AddFluxes(1.0, mesh, fluid, 'Outflow')

        self.assertTrue(np.all(fluid.Mass >= 0.0))
        self.assertTrue(np.all(fluid.Energy >= 0.0))
        self.assertAlmostEqual(float(np.sum(fluid.Mass)), 4.0)
        self.assertAlmostEqual(float(np.sum(fluid.Energy)), 4.0)

    def test_paired_face_limiter_reaches_global_admissibility_for_coupled_faces(self):
        """Neighboring restrictions must not survive a fixed repair count."""
        par = make_code_par()
        par.noghost = 0
        par.mesh.ghost_cells = 0
        par.nogrid = 5
        par.mesh.grid_cells = 5
        par.positivity_preserving = True
        par.positivity_density_floor = 0.0
        par.positivity_energy_floor = 0.0
        par.cfl_density_floor = 0.0
        par.dual_energy = True
        mesh = make_code_mesh(5)
        mesh._par = par
        mass = np.array([
            0.8673953529245595, 6.776326090692884,
            2.768587034204799, 0.42270495958486487,
            0.2879553791648065,
        ])
        momentum = np.array([
            0.09880948326072078, 0.32311661008571224,
            0.03990724798119123, -0.014152283269336843,
            0.00925101127551695,
        ])
        energy = np.array([
            0.36731554210974315, 0.14787788678725858,
            0.22036689100247855, 0.04871676701166089,
            0.02676529191260478,
        ])
        fluid = SimpleNamespace(
            Mass=as_named_array(mass.copy()),
            Mom=as_named_array(momentum.copy()),
            Energy=as_named_array(energy.copy()),
            InternalEnergy=as_named_array(energy.copy()),
            time=0.0,
        )
        mass_face = np.array([
            -1.7690080915545598, 1.7811742205546452,
            -3.723223801407037, -1.04004798128519,
            -1.2456146601584484,
        ])
        momentum_face = np.array([
            0.7153755556725097, -0.02435852042891355,
            -0.26620947455297345, 2.4873892451277824,
            -2.483994405078338,
        ])
        energy_face = np.array([
            -0.5043952152186525, -1.1852485717384549,
            -0.9156911448084882, -0.5172337136435069,
            -1.5154215992557107,
        ])
        solver = Solver()

        solver._positivity_limited_face_fluxes(
            fluid, 1.0, mesh, par,
            mass_face, momentum_face, energy_face,
        )

        self.assertTrue(np.all(np.isfinite(fluid.Mass)))
        self.assertTrue(np.all(np.isfinite(fluid.Mom)))
        self.assertTrue(np.all(np.isfinite(fluid.Energy)))
        self.assertTrue(np.all(fluid.Mass >= 0.0))
        self.assertTrue(np.all(fluid.Energy >= 0.0))
        self.assertAlmostEqual(float(np.sum(fluid.Mass)), float(np.sum(mass)))
        self.assertAlmostEqual(float(np.sum(fluid.Mom)), float(np.sum(momentum)))
        self.assertAlmostEqual(float(np.sum(fluid.Energy)), float(np.sum(energy)))
        self.assertTrue(np.any(solver._last_face_limiter_factors < 1.0))
        self.assertTrue(np.any(solver._last_face_limiter_factors > 0.0))

    def test_dual_energy_rejects_total_energy_below_kinetic_energy(self):
        """Dual energy must not hide an inadmissible conservative state."""
        par = make_code_par()
        par.noghost = 0
        par.mesh.ghost_cells = 0
        par.nogrid = 1
        par.mesh.grid_cells = 1
        par.positivity_preserving = True
        par.positivity_density_floor = 0.0
        par.positivity_energy_floor = 0.0
        par.cfl_density_floor = 0.0
        par.dual_energy = True
        mesh = make_code_mesh(1)
        mesh._par = par
        fluid = SimpleNamespace(
            Mass=as_named_array(np.array([1.0])),
            Mom=as_named_array(np.array([10.0])),
            Energy=as_named_array(np.array([1.0])),
            InternalEnergy=as_named_array(np.array([1.0])),
            time=0.0,
        )
        with self.assertRaisesRegex(ValueError, 'outside positivity domain'):
            Solver()._positivity_limited_face_fluxes(
                fluid, 1.0, mesh,
                par,
                np.zeros(1), np.zeros(1), np.zeros(1),
            )

    def test_dual_energy_prefers_dual_when_conservative_thermal_cancels(self):
        """A tiny positive E-K must not create a spurious cold cell."""
        par = make_code_par()
        par.noghost = 0
        par.mesh.ghost_cells = 0
        par.nogrid = 1
        par.mesh.grid_cells = 1
        par.cfl_density_floor = 0.0
        par.dual_energy = True
        par.dual_energy_pressure_selection = 'switch'
        mesh = make_code_mesh(1)
        mesh._par = par
        fluid = make_code_fluid(1)
        fluid.rho[:] = 1.0
        fluid.vel[:] = 10.0
        fluid.pre = as_named_array(np.ones(1))
        solver = Solver()
        solver.SetConserved(mesh, fluid)
        fluid.Energy[:] = 0.5 * fluid.Mom[:]**2 / fluid.Mass[:] + 1.0e-12
        fluid.InternalEnergy[:] = 1.0e-2

        solver.SetPrimitive(mesh, fluid, par=par)

        self.assertEqual(int(solver.dual_energy_pressure_selection_code[0]), 1)
        self.assertAlmostEqual(
            float(fluid.pre[0]),
            (fluid.eos.gamma - 1.0) * float(fluid.InternalEnergy[0]),
        )

    def test_internal_pressure_selection_uses_evolved_internal_energy(self):
        """The YAML internal mode must not reconstruct pressure from E-K."""
        par = make_code_par()
        par.noghost = 0
        par.mesh.ghost_cells = 0
        par.nogrid = 1
        par.mesh.grid_cells = 1
        par.cfl_density_floor = 0.0
        par.dual_energy = True
        par.dual_energy_pressure_selection = 'internal'
        mesh = make_code_mesh(1)
        mesh._par = par
        fluid = make_code_fluid(1)
        fluid.rho[:] = 1.0
        fluid.vel[:] = 10.0
        fluid.pre = as_named_array(np.ones(1))
        solver = Solver()
        solver.SetConserved(mesh, fluid)
        fluid.Energy[:] = 0.5 * fluid.Mom[:]**2 / fluid.Mass[:] + 1.0e-12
        fluid.InternalEnergy[:] = 1.0e-2

        solver.SetPrimitive(mesh, fluid, par=par)

        self.assertEqual(int(solver.dual_energy_pressure_selection_code[0]), 1)
        self.assertAlmostEqual(
            float(fluid.pre[0]),
            (fluid.eos.gamma - 1.0) * float(fluid.InternalEnergy[0]),
        )

    def test_switch_does_not_floor_valid_dual_energy_when_e_minus_k_fails(self):
        """Switch mode must retain valid InternalEnergy when E-K is invalid."""
        par = make_code_par()
        par.noghost = 0
        par.mesh.ghost_cells = 0
        par.nogrid = 1
        par.mesh.grid_cells = 1
        par.cfl_density_floor = 0.0
        par.dual_energy = True
        par.dual_energy_pressure_selection = 'switch'
        par.dual_energy_pressure_floor = 1.0e-20
        mesh = make_code_mesh(1)
        mesh._par = par
        fluid = make_code_fluid(1)
        fluid.rho[:] = 1.0
        fluid.vel[:] = 10.0
        fluid.pre = as_named_array(np.ones(1))
        solver = Solver()
        solver.SetConserved(mesh, fluid)
        fluid.Energy[:] = 0.5 * fluid.Mom[:]**2 / fluid.Mass[:] - 1.0
        fluid.InternalEnergy[:] = 1.0e-2

        solver.SetPrimitive(mesh, fluid, par=par)

        self.assertEqual(int(solver.dual_energy_pressure_selection_code[0]), 1)
        self.assertAlmostEqual(float(fluid.InternalEnergy[0]), 1.0e-2)
        self.assertAlmostEqual(
            float(fluid.pre[0]),
            (fluid.eos.gamma - 1.0) * 1.0e-2,
        )

    def test_spherical_origin_flux_is_zeroed(self):
        mesh = Mesh()
        mesh.coordsys = 'spherical'
        fluid = Fluid()
        fluid.Mass = np.ones(8) * unyt.g
        fluid.Mom = np.ones(8) * unyt.g*unyt.cm/unyt.s
        fluid.Energy = np.ones(8) * unyt.g*unyt.cm**2/unyt.s**2
        fluid.Mass.flux = np.ones(8) * unyt.g/unyt.cm**2/unyt.s
        fluid.Mom.flux = np.ones(8) * unyt.dyn/unyt.cm**2
        fluid.Energy.flux = np.ones(8) * unyt.g/unyt.s**3

        Solver()._zero_spherical_origin_flux(mesh, fluid)

        self.assertEqual(fluid.Mass.flux[0], 0.0 * fluid.Mass.flux.units)
        self.assertEqual(fluid.Mom.flux[0], 0.0 * fluid.Mom.flux.units)
        self.assertEqual(fluid.Energy.flux[0], 0.0 * fluid.Energy.flux.units)

    def test_spherical_first_active_cell_momentum_evolves_conservatively(self):
        mesh = Mesh()
        mesh.coordsys = 'spherical'
        fluid = Fluid()
        fluid.rho = np.ones(8) * unyt.g/unyt.cm**3
        fluid.vel = np.zeros(8) * unyt.cm/unyt.s
        fluid.pre = np.zeros(8) * unyt.dyn/unyt.cm**2
        fluid.time = 0.0 * unyt.s
        fluid.Mass = np.ones(8) * unyt.g
        fluid.Mom = np.ones(8) * unyt.g*unyt.cm/unyt.s
        fluid.Energy = np.ones(8) * unyt.g*unyt.cm**2/unyt.s**2
        fluid.Mass.flux = np.zeros(8) * unyt.g/unyt.cm**2/unyt.s
        fluid.Mom.flux = np.zeros(8) * unyt.dyn/unyt.cm**2
        fluid.Energy.flux = np.zeros(8) * unyt.g/unyt.s**3

        Solver().AddFluxes(1.0*unyt.s, mesh, fluid, 'OpenSph')

        self.assertEqual(fluid.Mom[0], 1.0 * fluid.Mom.units)

    def test_hydrogen_source_cools_and_updates_neutral_fraction(self):
        par = make_code_par()
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        fluid.rho = np.ones(8, dtype=float) * 1.0e-24
        fluid.vel = np.zeros(8, dtype=float)
        fluid.temp = np.ones(8, dtype=float) * 1.0e5
        fluid.mu = np.ones(8, dtype=float)
        fluid.xHI = np.ones(8, dtype=float) * 0.5
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)
        energy_before = fluid.Energy.copy()
        xHI_before = fluid.xHI.copy()

        Solver().ApplyThermochemistryFast(1.0e6, mesh, fluid, par)

        self.assertTrue(np.all(np.asarray(fluid.Energy)[2:6] < np.asarray(energy_before)[2:6]))
        self.assertTrue(np.any(fluid.xHI[2:6] != xHI_before[2:6]))
        np.testing.assert_array_equal(fluid.xHI[:2], xHI_before[:2])

    def test_hydrogen_chemistry_can_run_without_thermal_coupling(self):
        par = make_code_par()
        par.hydrogen_thermal_coupling = False
        par.hydrogen_collisional_ionization = False
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        fluid.rho = np.ones(8, dtype=float) * 1.0e-24
        fluid.vel = np.zeros(8, dtype=float)
        fluid.temp = np.ones(8, dtype=float) * 2.0e4
        fluid.mu = np.ones(8, dtype=float) * 0.5
        fluid.xHI = np.ones(8, dtype=float) * 0.5
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)
        energy_before = fluid.Energy.copy()
        xHI_before = fluid.xHI.copy()

        Solver().ApplyThermochemistryFast(1.0e6, mesh, fluid, par)

        np.testing.assert_allclose(np.asarray(fluid.Energy), np.asarray(energy_before))
        self.assertTrue(np.all(fluid.xHI[2:6] > xHI_before[2:6]))

    def test_hydrogen_radiation_field_attenuates_heats_and_ionizes(self):
        par = make_code_par()
        par.hydrogen_collisional_ionization = False
        par.hydrogen_radiation_field = True
        par.hydrogen_sigma_gamma = 1.0e-18
        par.hydrogen_epsilon_gamma = 1.0e-12
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        fluid.rho = np.ones(8, dtype=float)
        fluid.vel = np.zeros(8, dtype=float)
        fluid.temp = np.ones(8, dtype=float) * 1.0e4
        fluid.mu = np.ones(8, dtype=float)
        fluid.xHI = np.ones(8, dtype=float) * 0.9
        fluid.ngamma = np.ones(8, dtype=float) * 1.0e3
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)
        energy_before = fluid.Energy.copy()
        xHI_before = fluid.xHI.copy()
        ngamma_before = fluid.ngamma.copy()

        Solver().ApplyThermochemistryFast(1.0e2, mesh, fluid, par)

        np.testing.assert_allclose(np.asarray(fluid.ngamma), np.asarray(ngamma_before))
        self.assertTrue(np.all(fluid.xHI[2:6] >= xHI_before[2:6]))
        self.assertTrue(np.all(np.asarray(fluid.Energy)[2:6] <= np.asarray(energy_before)[2:6]))
        np.testing.assert_array_equal(fluid.ngamma[:2], ngamma_before[:2])

    def test_hydrogen_fixed_radiation_field_ionizes_without_attenuation(self):
        par = make_code_par()
        par.hydrogen_thermal_coupling = False
        par.hydrogen_collisional_ionization = False
        par.hydrogen_radiation_field = True
        par.hydrogen_radiation_evolution = False
        par.hydrogen_sigma_gamma = 1.0e-18
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        fluid.rho = np.ones(8, dtype=float)
        fluid.vel = np.zeros(8, dtype=float)
        fluid.temp = np.ones(8, dtype=float) * 2.0e4
        fluid.mu = np.ones(8, dtype=float)
        fluid.xHI = np.ones(8, dtype=float)
        fluid.ngamma = np.ones(8, dtype=float) * 1.0e3
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)
        energy_before = fluid.Energy.copy()
        xHI_before = fluid.xHI.copy()
        ngamma_before = fluid.ngamma.copy()

        Solver().ApplyThermochemistryFast(1.0e2, mesh, fluid, par)

        np.testing.assert_allclose(np.asarray(fluid.ngamma), np.asarray(ngamma_before))
        np.testing.assert_allclose(np.asarray(fluid.Energy), np.asarray(energy_before))
        self.assertTrue(np.all(fluid.xHI[2:6] < xHI_before[2:6]))

    def test_hydrogen_fixed_radiation_field_can_disable_recombination(self):
        par = make_code_par()
        par.hydrogen_thermal_coupling = False
        par.hydrogen_recombination = False
        par.hydrogen_collisional_ionization = False
        par.hydrogen_radiation_field = True
        par.hydrogen_radiation_evolution = False
        par.hydrogen_sigma_gamma = 1.0e-18
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        fluid.rho = np.ones(8, dtype=float)
        fluid.vel = np.zeros(8, dtype=float)
        fluid.temp = np.ones(8, dtype=float) * 2.0e4
        fluid.mu = np.ones(8, dtype=float)
        fluid.xHI = np.ones(8, dtype=float)
        fluid.ngamma = np.ones(8, dtype=float) * 1.0e3
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)

        Solver().ApplyThermochemistryFast(1.0e2, mesh, fluid, par)

        photoionization_rate = (
            unyt.c.to_value(unyt.cm / unyt.s)
            * par.hydrogen_sigma_gamma
            * np.asarray(fluid.ngamma[2:6])
        )
        expected = 1.0 / (1.0 + photoionization_rate * 1.0e2)
        np.testing.assert_allclose(fluid.xHI[2:6], expected)

    def test_radiative_transfer_supplies_ngamma_to_hydrogen_sources(self):
        par = make_code_par()
        par.hydrogen_thermal_coupling = False
        par.hydrogen_recombination = False
        par.hydrogen_collisional_ionization = False
        par.radiative_transfer = True
        par.radiative_transfer_method = 'long_characteristics'
        par.radiative_transfer_boundary_flux = 1.0e15
        par.radiative_transfer_source_photon_rate = 0.0
        par.radiative_transfer_direction = 1
        par.hydrogen_sigma_gamma = 1.0e-18
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        fluid.rho = np.ones(8, dtype=float)
        fluid.vel = np.zeros(8, dtype=float)
        fluid.temp = np.ones(8, dtype=float) * 2.0e4
        fluid.mu = np.ones(8, dtype=float)
        fluid.xHI = np.ones(8, dtype=float)
        fluid.ngamma = np.zeros(8, dtype=float)
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)

        Solver().ApplyThermochemistryFast(1.0e2, mesh, fluid, par)

        self.assertTrue(hasattr(fluid, 'ngamma'))
        self.assertTrue(np.all(np.isfinite(np.asarray(fluid.ngamma))))
        self.assertTrue(np.all(np.asarray(fluid.ngamma) >= 0.0))
        self.assertTrue(np.all(fluid.xHI[2:6] < 1.0))

    def test_hydrogen_subcycle_timestep_can_be_smaller_than_dtmax(self):
        par = make_code_par()
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        fluid.rho = np.ones(8, dtype=float) * 1.0e10
        fluid.vel = np.zeros(8, dtype=float)
        fluid.temp = np.ones(8, dtype=float) * 1.0e5
        fluid.mu = np.ones(8, dtype=float)
        fluid.xHI = np.ones(8, dtype=float) * 0.5
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)

        thermochemistry_dt, _ = Solver().GetSourceTimestepFast(mesh, fluid, par, 1.0)

        self.assertLess(thermochemistry_dt, par.dtmax)

    def test_hydrogen_subcycling_does_not_limit_hydro_timestep(self):
        par = make_code_par()
        mesh = make_code_mesh()
        mesh.xdelta = np.ones(8, dtype=float) * 1.0e12
        fluid = make_code_fluid()
        fluid.rho = np.ones(8, dtype=float) * 1.0e10
        fluid.vel = np.zeros(8, dtype=float)
        fluid.temp = np.ones(8, dtype=float) * 1.0e5
        fluid.mu = np.ones(8, dtype=float)
        fluid.xHI = np.ones(8, dtype=float) * 0.5
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)

        thermochemistry_dt, _ = Solver().GetSourceTimestepFast(mesh, fluid, par, 1.0)
        hydro_dt = Solver().GetTimeStep(mesh, fluid, par)

        self.assertLess(thermochemistry_dt, par.dtmax)
        self.assertEqual(hydro_dt, par.dtmax)

    def test_zero_density_cell_does_not_limit_hydro_timestep(self):
        par = make_code_par()
        mesh = make_code_mesh()
        mesh.xdelta = np.ones(8, dtype=float) * 1.0e12
        fluid = make_code_fluid()
        fluid.rho[3] = 0.0
        fluid.SetPressure()

        hydro_dt = Solver().GetTimeStep(mesh, fluid, par)

        self.assertTrue(np.isfinite(hydro_dt))
        self.assertEqual(hydro_dt, par.dtmax)
        self.assertEqual(float(fluid.vsignal[3]), 0.0)

    def test_ghost_cells_do_not_limit_hydro_timestep(self):
        par = make_code_par()
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        # The active cells are [2:6]; boundary ghost values must not enter
        # the CFL minimum even if a boundary update temporarily leaves them
        # with an unusable velocity.
        fluid.vel[[0, 1, 6, 7]] = 1.0e30
        fluid.SetPressure()

        hydro_dt = Solver().GetTimeStep(mesh, fluid, par)

        self.assertTrue(np.isfinite(hydro_dt))
        self.assertGreater(hydro_dt, 1.0e-8)

    def test_cfl_density_floor_treats_numerical_vacuum_as_zero_density(self):
        par = make_code_par()
        par.cfl_density_floor = 1.0e-9
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        fluid.rho[3] = 1.0e-12
        fluid.vel[3] = 1.0e30
        fluid.SetPressure()

        hydro_dt = Solver().GetTimeStep(mesh, fluid, par)

        self.assertTrue(np.isfinite(hydro_dt))
        self.assertGreater(hydro_dt, 1.0e-8)
        self.assertEqual(float(fluid.vsignal[3]), 0.0)

    def test_cfl_density_floor_preserves_low_density_conserved_energy(self):
        par = make_code_par()
        par.cfl_density_floor = 1.0e-9
        mesh = make_code_mesh()
        mesh._par = par
        fluid = make_code_fluid()
        fluid.rho[3] = 1.0e-12
        fluid.vel[3] = 1.0e3
        fluid.SetPressure()
        fluid.pre[3] = 1.0e-12
        Solver().SetConserved(mesh, fluid)
        conserved_before = (
            float(fluid.Mass[3]), float(fluid.Mom[3]), float(fluid.Energy[3])
        )

        fluid.Mass[3] *= 0.5
        fluid.Mom[3] *= 0.5
        fluid.Energy[3] *= 0.5
        Solver().SetPrimitive(mesh, fluid, par=par)
        Solver().SetConserved(mesh, fluid)

        np.testing.assert_allclose(
            [float(fluid.Mass[3]), float(fluid.Mom[3]), float(fluid.Energy[3])],
            np.asarray(conserved_before) * 0.5,
        )
        self.assertEqual(float(fluid.pre[3]), 0.0)

    def test_hydro_only_sync_refreshes_adiabatic_temperature(self):
        par = make_code_par()
        par.hydrogen_chemistry = False
        par.cie_cooling = False
        par.metal_pie_enabled = False
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        fluid.rho = np.linspace(1.0, 2.0, 8)
        fluid.temp = np.linspace(2.0, 9.0, 8)
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)
        expected = np.asarray(fluid.temp, dtype=float).copy()
        fluid.temp[:] = 0.0
        sim = Rsim.FromComponents(par, mesh, fluid)

        sim._sync_hydro_state()

        np.testing.assert_allclose(fluid.temp, expected)

    def test_negative_hydro_pressure_uses_configured_temperature_floor(self):
        par = make_code_par()
        par.hydro_temperature_floor = 3.0
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        fluid.Mass = np.asarray(fluid.rho, dtype=float).copy()
        fluid.Energy = np.zeros(8, dtype=float)
        fluid.Mom = np.zeros(8, dtype=float)

        Solver().SetPrimitive(mesh, fluid, par)
        fluid.SetTemperature()

        np.testing.assert_allclose(fluid.temp, 3.0)
        self.assertTrue(np.all(np.asarray(fluid.pre) > 0.0))

    def test_unknown_thermochemistry_network_raises_clear_error(self):
        par = Par('Periodic')
        par.thermochemistry_network = 'unknown'

        with self.assertRaisesRegex(ValueError, 'Unknown thermo-chemistry network'):
            rtc.get_network(par)

    def test_rsim_step_rejects_unknown_mode(self):
        sim = Rsim.FromComponents(Par('Periodic'), Mesh(), Fluid())

        with self.assertRaisesRegex(ValueError, 'Unknown step mode'):
            sim.Step(dt=1.0 * unyt.s, mode='unknown')

    def test_rsim_hydro_step_supports_ssprk2(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = False
        mesh = Mesh()
        fluid = RealFluid()
        fluid.time = 0.0 * unyt.s
        fluid.Mass = np.ones(8) * unyt.g
        fluid.Mom = np.ones(8) * (unyt.g * unyt.cm / unyt.s)
        fluid.Energy = np.ones(8) * (unyt.g * unyt.cm**2 / unyt.s**2)
        sim = Rsim.FromComponents(par, mesh, fluid)

        call_counts = {
            'prepare': 0,
            'advance': 0,
            'finalize': 0,
            'sync': 0,
        }

        def fake_prepare(fluid=None):
            call_counts['prepare'] += 1

        def fake_advance(dt, fluid=None):
            call_counts['advance'] += 1
            old_mass = fluid.Mass.copy()
            mass_flux = np.ones_like(fluid.Mass) * (
                unyt.g / (unyt.cm**2 * unyt.s)
            )
            return old_mass, mass_flux

        def fake_finalize(
            dt, old_mass, mass_flux, advect_chemistry=True, fluid=None,
            apply_gravity=True,
        ):
            call_counts['finalize'] += 1
            fluid.Mass = fluid.Mass + 1.0 * unyt.g
            fluid.Mom = fluid.Mom + 2.0 * (unyt.g * unyt.cm / unyt.s)
            fluid.Energy = fluid.Energy + 3.0 * (unyt.g * unyt.cm**2 / unyt.s**2)
            fluid.time += dt

        def fake_sync(fluid=None):
            call_counts['sync'] += 1

        sim.PrepareConservedStep = fake_prepare
        sim.AdvanceHydroFluxes = fake_advance
        sim.FinalizeHydroStep = fake_finalize
        sim._sync_hydro_state = fake_sync

        result = sim.Step(dt=0.25 * unyt.s, mode='hydro', hydro_integrator='ssprk2')

        self.assertEqual(result['hydro_steps'], 1)
        self.assertEqual(result['source_steps'], 0)
        self.assertEqual(call_counts['prepare'], 2)
        self.assertEqual(call_counts['advance'], 2)
        self.assertEqual(call_counts['finalize'], 2)
        self.assertEqual(call_counts['sync'], 1)
        self.assertEqual(fluid.time, 0.25 * unyt.s)
        np.testing.assert_allclose(fluid.Mass.value, np.full(8, 2.0))
        np.testing.assert_allclose(fluid.Mom.value, np.full(8, 3.0))
        np.testing.assert_allclose(fluid.Energy.value, np.full(8, 4.0))

    def test_rsim_ssprk2_strang_applies_sources_outside_rk_stages(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = False
        par.source_integrator = 'strang'
        par.externalgravity = True
        mesh = Mesh()
        fluid = RealFluid()
        fluid.time = 0.0 * unyt.s
        fluid.Mass = np.ones(8) * unyt.g
        fluid.Mom = np.ones(8) * (unyt.g * unyt.cm / unyt.s)
        fluid.Energy = np.ones(8) * (unyt.g * unyt.cm**2 / unyt.s**2)
        sim = Rsim.FromComponents(par, mesh, fluid)

        source_steps = []
        stage_gravity_flags = []

        def fake_gravity(dt, mesh, fluid, par):
            source_steps.append(dt)
            sim.solver.last_gravity_work = 0.0
            sim.solver.last_centrifugal_work = 0.0
            sim.solver.last_gravity_work_by_cell = None

        def fake_prepare(fluid=None):
            pass

        def fake_advance(dt, fluid=None):
            return fluid.Mass.copy(), np.zeros_like(fluid.Mass)

        def fake_finalize(
            dt, old_mass, mass_flux, advect_chemistry=True, fluid=None,
            apply_gravity=True,
        ):
            stage_gravity_flags.append(apply_gravity)
            fluid.time += dt

        sim.solver.ApplyGravity = fake_gravity
        sim.PrepareConservedStep = fake_prepare
        sim.AdvanceHydroFluxes = fake_advance
        sim.FinalizeHydroStep = fake_finalize
        sim._sync_hydro_state = lambda fluid=None: None

        result = sim.Step(
            dt=0.25 * unyt.s,
            mode='hydro',
            hydro_integrator='ssprk2',
        )

        self.assertEqual(result['hydro_steps'], 1)
        self.assertEqual(len(source_steps), 2)
        self.assertTrue(all(np.isclose(float(dt), 0.125) for dt in source_steps))
        self.assertEqual(stage_gravity_flags, [False, False])

    def test_rsim_source_step_advances_time_without_hydro_step(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = False
        mesh = Mesh()
        fluid = Fluid()
        fluid.time = 0.0 * unyt.s
        sim = Rsim.FromComponents(par, mesh, fluid)

        result = sim.Step(dt=0.25 * unyt.s, mode='sources')

        self.assertEqual(result['hydro_steps'], 0)
        self.assertEqual(result['source_steps'], 0)
        self.assertEqual(fluid.time, 0.25 * unyt.s)
        self.assertTrue(hasattr(fluid, 'Mass'))

    def test_rsim_evolve_uses_step_and_history_callback(self):
        par = make_code_par()
        par.hydrogen_chemistry = False
        par.timestep.dtmax = 0.2
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        fluid.rho = np.ones(8, dtype=float)
        fluid.vel = np.zeros(8, dtype=float)
        fluid.temp = np.zeros(8, dtype=float)
        fluid.mu = np.ones(8, dtype=float)
        fluid.SetPressure()
        fluid.time = 0.0 * unyt.s
        sim = Rsim.FromComponents(par, mesh, fluid)
        history = []

        counters = sim.Evolve(
            final_time=0.5 * unyt.s,
            mode='sources',
            history_callback=lambda current_sim: history.append(
                current_sim.fluid.time.to_value(unyt.s)
            ),
        )

        self.assertEqual(counters['hydro_steps'], 0)
        self.assertEqual(counters['source_steps'], 0)
        self.assertEqual(fluid.time, 0.5 * unyt.s)
        np.testing.assert_allclose(history, [0.0, 0.2, 0.4, 0.5])

    def test_rsim_evolve_uses_custom_step_backend(self):
        par = make_code_par()
        par.hydrogen_chemistry = False
        par.dtmax = 0.2
        mesh = make_code_mesh()
        fluid = make_code_fluid()
        fluid.rho = np.ones(8, dtype=float)
        fluid.vel = np.zeros(8, dtype=float)
        fluid.temp = np.zeros(8, dtype=float)
        fluid.mu = np.ones(8, dtype=float)
        fluid.SetPressure()
        fluid.time = 0.0 * unyt.s
        sim = Rsim.FromComponents(par, mesh, fluid)
        history = []
        backend_calls = []

        def fail_step(*args, **kwargs):
            raise AssertionError('Step should not be called when backend is provided')

        def custom_backend(dt=None, mode=None, **kwargs):
            backend_calls.append((dt, mode, kwargs))
            fluid.time += dt
            return {'dt': dt, 'hydro_steps': 0, 'source_steps': 0}

        sim.Step = fail_step
        sim.GetStepTime = lambda dt=None, final_time=None: min(
            0.2 * unyt.s,
            final_time - fluid.time,
        )

        counters = sim.Evolve(
            final_time=0.5 * unyt.s,
            mode='sources',
            history_callback=lambda current_sim: history.append(
                current_sim.fluid.time.to_value(unyt.s)
            ),
            step_backend=custom_backend,
        )

        self.assertEqual(counters['hydro_steps'], 0)
        self.assertEqual(counters['source_steps'], 0)
        self.assertEqual(len(backend_calls), 3)
        self.assertEqual(fluid.time, 0.5 * unyt.s)
        np.testing.assert_allclose(history, [0.0, 0.2, 0.4, 0.5])


if __name__ == '__main__':
    unittest.main()
