import unittest

import numpy as np
import unyt

from radhydropy.fluid import Fluid as RealFluid
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
import radhydropy.thermo_chemistry as rtc


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
    def test_open_boundary_fills_all_ghost_cells(self):
        fluid = Fluid()
        Solver().SetBoundary(None, fluid, Par('Open'))

        np.testing.assert_array_equal(fluid.rho[:2].value, [2.0, 2.0])
        np.testing.assert_array_equal(fluid.rho[-2:].value, [5.0, 5.0])
        np.testing.assert_array_equal(fluid.vel[:2].value, [12.0, 12.0])
        np.testing.assert_array_equal(fluid.vel[-2:].value, [15.0, 15.0])

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

    def test_spherical_center_momentum_is_projected_after_update(self):
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

        self.assertEqual(fluid.Mom[0], 0.0 * fluid.Mom.units)

    def test_hydrogen_source_cools_and_updates_neutral_fraction(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = True
        par.hydrogen_mass_fraction = 1.0
        par.hydrogen_source_CFL = 0.1
        par.hydrogen_update_mu = False
        mesh = Mesh()
        fluid = RealFluid()
        fluid.eos = EOS()
        fluid.rho = np.ones(8) * unyt.mp/unyt.cm**3
        fluid.vel = np.zeros(8) * unyt.cm/unyt.s
        fluid.temp = np.ones(8) * 1.0e5 * unyt.K
        fluid.mu = np.ones(8)
        fluid.xHI = np.ones(8) * 0.5
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)
        energy_before = fluid.Energy.copy()
        xHI_before = fluid.xHI.copy()

        Solver().ApplyThermochemistry(1.0e6 * unyt.s, mesh, fluid, par)

        self.assertTrue(np.all(fluid.Energy[2:6] < energy_before[2:6]))
        self.assertTrue(np.all(fluid.xHI[2:6] < xHI_before[2:6]))
        np.testing.assert_array_equal(fluid.xHI[:2], xHI_before[:2])

    def test_hydrogen_chemistry_can_run_without_thermal_coupling(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = True
        par.hydrogen_mass_fraction = 1.0
        par.hydrogen_source_CFL = 0.1
        par.hydrogen_update_mu = False
        par.hydrogen_thermal_coupling = False
        par.hydrogen_collisional_ionization = False
        mesh = Mesh()
        fluid = RealFluid()
        fluid.eos = EOS()
        fluid.rho = np.ones(8) * unyt.mp/unyt.cm**3
        fluid.vel = np.zeros(8) * unyt.cm/unyt.s
        fluid.temp = np.ones(8) * 2.0e4 * unyt.K
        fluid.mu = np.ones(8) * 0.5
        fluid.xHI = np.ones(8) * 0.5
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)
        energy_before = fluid.Energy.copy()
        xHI_before = fluid.xHI.copy()

        Solver().ApplyThermochemistry(1.0e6 * unyt.s, mesh, fluid, par)

        np.testing.assert_allclose(fluid.Energy.value, energy_before.value)
        self.assertTrue(np.all(fluid.xHI[2:6] > xHI_before[2:6]))

    def test_hydrogen_radiation_field_attenuates_heats_and_ionizes(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = True
        par.hydrogen_mass_fraction = 1.0
        par.hydrogen_source_CFL = 0.1
        par.hydrogen_update_mu = False
        par.hydrogen_collisional_ionization = False
        par.hydrogen_radiation_field = True
        par.hydrogen_sigma_gamma = 1.0e-18 * unyt.cm**2
        par.hydrogen_epsilon_gamma = 1.0e-12 * unyt.erg
        mesh = Mesh()
        fluid = RealFluid()
        fluid.eos = EOS()
        fluid.rho = np.ones(8) * unyt.mp/unyt.cm**3
        fluid.vel = np.zeros(8) * unyt.cm/unyt.s
        fluid.temp = np.ones(8) * 1.0e4 * unyt.K
        fluid.mu = np.ones(8)
        fluid.xHI = np.ones(8) * 0.9
        fluid.ngamma = np.ones(8) * 1.0e3 / unyt.cm**3
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)
        energy_before = fluid.Energy.copy()
        xHI_before = fluid.xHI.copy()
        ngamma_before = fluid.ngamma.copy()

        Solver().ApplyThermochemistry(1.0e2 * unyt.s, mesh, fluid, par)

        self.assertTrue(np.all(fluid.ngamma[2:6] < ngamma_before[2:6]))
        self.assertTrue(np.all(fluid.Energy[2:6] > energy_before[2:6]))
        self.assertTrue(np.all(fluid.xHI[2:6] < xHI_before[2:6]))
        np.testing.assert_array_equal(fluid.ngamma[:2], ngamma_before[:2])

    def test_hydrogen_fixed_radiation_field_ionizes_without_attenuation(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = True
        par.hydrogen_mass_fraction = 1.0
        par.hydrogen_source_CFL = 0.1
        par.hydrogen_update_mu = False
        par.hydrogen_thermal_coupling = False
        par.hydrogen_collisional_ionization = False
        par.hydrogen_radiation_field = True
        par.hydrogen_radiation_evolution = False
        par.hydrogen_sigma_gamma = 1.0e-18 * unyt.cm**2
        mesh = Mesh()
        fluid = RealFluid()
        fluid.eos = EOS()
        fluid.rho = np.ones(8) * unyt.mp/unyt.cm**3
        fluid.vel = np.zeros(8) * unyt.cm/unyt.s
        fluid.temp = np.ones(8) * 2.0e4 * unyt.K
        fluid.mu = np.ones(8)
        fluid.xHI = np.ones(8)
        fluid.ngamma = np.ones(8) * 1.0e3 / unyt.cm**3
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)
        energy_before = fluid.Energy.copy()
        xHI_before = fluid.xHI.copy()
        ngamma_before = fluid.ngamma.copy()

        Solver().ApplyThermochemistry(1.0e2 * unyt.s, mesh, fluid, par)

        np.testing.assert_allclose(fluid.ngamma.value, ngamma_before.value)
        np.testing.assert_allclose(fluid.Energy.value, energy_before.value)
        self.assertTrue(np.all(fluid.xHI[2:6] < xHI_before[2:6]))

    def test_hydrogen_fixed_radiation_field_can_disable_recombination(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = True
        par.hydrogen_mass_fraction = 1.0
        par.hydrogen_source_CFL = 0.1
        par.hydrogen_update_mu = False
        par.hydrogen_thermal_coupling = False
        par.hydrogen_recombination = False
        par.hydrogen_collisional_ionization = False
        par.hydrogen_radiation_field = True
        par.hydrogen_radiation_evolution = False
        par.hydrogen_sigma_gamma = 1.0e-18 * unyt.cm**2
        mesh = Mesh()
        fluid = RealFluid()
        fluid.eos = EOS()
        fluid.rho = np.ones(8) * unyt.mp/unyt.cm**3
        fluid.vel = np.zeros(8) * unyt.cm/unyt.s
        fluid.temp = np.ones(8) * 2.0e4 * unyt.K
        fluid.mu = np.ones(8)
        fluid.xHI = np.ones(8)
        fluid.ngamma = np.ones(8) * 1.0e3 / unyt.cm**3
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)

        Solver().ApplyThermochemistry(1.0e2 * unyt.s, mesh, fluid, par)

        photoionization_rate = (
            unyt.c.to(unyt.cm/unyt.s)
            * par.hydrogen_sigma_gamma
            * fluid.ngamma[2:6]
        ).to_value(1.0/unyt.s)
        expected = 1.0 / (1.0 + photoionization_rate * 1.0e2)
        np.testing.assert_allclose(fluid.xHI[2:6], expected)

    def test_radiative_transfer_supplies_ngamma_to_hydrogen_sources(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = True
        par.hydrogen_mass_fraction = 1.0
        par.hydrogen_source_CFL = 0.1
        par.hydrogen_update_mu = False
        par.hydrogen_thermal_coupling = False
        par.hydrogen_recombination = False
        par.hydrogen_collisional_ionization = False
        par.radiative_transfer = True
        par.radiative_transfer_method = 'long_characteristics'
        par.radiative_transfer_boundary_flux = 1.0e15 / (unyt.cm**2 * unyt.s)
        par.radiative_transfer_source_photon_rate = 0.0 / unyt.s
        par.radiative_transfer_direction = 1
        par.hydrogen_sigma_gamma = 1.0e-18 * unyt.cm**2
        mesh = Mesh()
        fluid = RealFluid()
        fluid.eos = EOS()
        fluid.rho = np.ones(8) * unyt.mp/unyt.cm**3
        fluid.vel = np.zeros(8) * unyt.cm/unyt.s
        fluid.temp = np.ones(8) * 2.0e4 * unyt.K
        fluid.mu = np.ones(8)
        fluid.xHI = np.ones(8)
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)

        Solver().ApplyThermochemistry(1.0e2 * unyt.s, mesh, fluid, par)

        self.assertTrue(hasattr(fluid, 'ngamma'))
        self.assertTrue(np.all(fluid.ngamma[2:6] > 0.0 / unyt.cm**3))
        self.assertTrue(np.all(fluid.xHI[2:6] < 1.0))

    def test_hydrogen_subcycle_timestep_can_be_smaller_than_dtmax(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = True
        par.hydrogen_mass_fraction = 1.0
        par.hydrogen_source_CFL = 0.1
        par.hydrogen_update_mu = False
        mesh = Mesh()
        fluid = RealFluid()
        fluid.eos = EOS()
        fluid.rho = np.ones(8) * 1.0e10 * unyt.mp/unyt.cm**3
        fluid.vel = np.zeros(8) * unyt.cm/unyt.s
        fluid.temp = np.ones(8) * 1.0e5 * unyt.K
        fluid.mu = np.ones(8)
        fluid.xHI = np.ones(8) * 0.5
        fluid.SetPressure()
        Solver().SetConserved(mesh, fluid)
        Solver().SetPrimitive(mesh, fluid)

        thermochemistry_dt = Solver().GetThermochemistryTimeStep(mesh, fluid, par)

        self.assertLess(thermochemistry_dt, par.dtmax)

    def test_hydrogen_subcycling_does_not_limit_hydro_timestep(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = True
        par.hydrogen_mass_fraction = 1.0
        par.hydrogen_source_CFL = 0.1
        par.hydrogen_update_mu = False
        mesh = Mesh()
        mesh.xdelta = np.ones(8) * 1.0e12 * unyt.cm
        fluid = RealFluid()
        fluid.eos = EOS()
        fluid.rho = np.ones(8) * 1.0e10 * unyt.mp/unyt.cm**3
        fluid.vel = np.zeros(8) * unyt.cm/unyt.s
        fluid.temp = np.ones(8) * 1.0e5 * unyt.K
        fluid.mu = np.ones(8)
        fluid.xHI = np.ones(8) * 0.5
        fluid.SetPressure()

        thermochemistry_dt = Solver().GetThermochemistryTimeStep(mesh, fluid, par)
        hydro_dt = Solver().GetTimeStep(mesh, fluid, par)

        self.assertLess(thermochemistry_dt, par.dtmax)
        self.assertEqual(hydro_dt, par.dtmax)

    def test_unknown_thermochemistry_network_raises_clear_error(self):
        par = Par('Periodic')
        par.thermochemistry_network = 'unknown'

        with self.assertRaisesRegex(ValueError, 'Unknown thermo-chemistry network'):
            rtc.get_network(par)

    def test_rsim_step_rejects_unknown_mode(self):
        sim = Rsim.FromComponents(Par('Periodic'), Mesh(), Fluid())

        with self.assertRaisesRegex(ValueError, 'Unknown step mode'):
            sim.Step(dt=1.0 * unyt.s, mode='unknown')

    def test_rsim_source_step_advances_time_without_hydro_step(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = False
        mesh = Mesh()
        fluid = Fluid()
        fluid.time = 0.0 * unyt.s
        sim = Rsim.FromComponents(par, mesh, fluid)

        result = sim.Step(dt=0.25 * unyt.s, mode='sources')

        self.assertEqual(result['hydro_steps'], 0)
        self.assertEqual(result['source_steps'], 1)
        self.assertEqual(fluid.time, 0.25 * unyt.s)
        self.assertTrue(hasattr(fluid, 'Mass'))

    def test_rsim_evolve_uses_step_and_history_callback(self):
        par = Par('Periodic')
        par.hydrogen_chemistry = False
        par.dtmax = 0.2 * unyt.s
        mesh = Mesh()
        fluid = RealFluid()
        fluid.eos = EOS()
        fluid.rho = np.ones(8) * unyt.g/unyt.cm**3
        fluid.vel = np.zeros(8) * unyt.cm/unyt.s
        fluid.temp = np.zeros(8) * unyt.K
        fluid.mu = np.ones(8)
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
        self.assertEqual(counters['source_steps'], 3)
        self.assertEqual(fluid.time, 0.5 * unyt.s)
        np.testing.assert_allclose(history, [0.0, 0.2, 0.4, 0.5])


if __name__ == '__main__':
    unittest.main()
