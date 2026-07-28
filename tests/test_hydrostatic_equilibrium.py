import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import unyt

from radhydropy.eos import EOS
from radhydropy.gravity import Gravity
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver


class Testing(unittest.TestCase):
    def test_hydrostatic_equilibrium_profile_balances_gravity(self):
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

        icparams = {
            'nogrid': 256,
            'coordsys': 'cartesian',
            'boxsize': 10.0 * unyt.pc,
            'time': 0.0 * unyt.s,
            'rho_ref': 1.0e-24 * unyt.g / unyt.cm**3,
            'tempini': 1.0e4 * unyt.K,
            'muini': 1.0,
            'gravity_strength': 1.0e-7 * unyt.cm / unyt.s**2,
        }

        sim = module.Simwrap(icparams)
        pressure = (
            unyt.kb * sim.fluid.temp / (sim.fluid.mu * unyt.mp) * sim.fluid.rho
        ).to(unyt.dyn / unyt.cm**2)
        coordinate = sim.mesh.coordinate.to_value(unyt.cm)
        dPdx = np.gradient(
            pressure.to_value(unyt.dyn / unyt.cm**2),
            coordinate,
        )
        expected = (
            -sim.fluid.rho * icparams['gravity_strength']
        ).to_value(unyt.dyn / unyt.cm**3)

        interior = slice(2, -2)
        np.testing.assert_allclose(
            dPdx[interior],
            expected[interior],
            rtol=2.0e-3,
            atol=0.0,
        )
        np.testing.assert_allclose(
            sim.fluid.vel.to_value(unyt.cm / unyt.s),
            0.0,
            atol=0.0,
        )

    def test_single_tiny_hydro_step_changes_state_only_slightly(self):
        example_dir = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HydrostaticEquilibrium1D'
        )
        tools_path = example_dir / 'tools.py'
        spec = importlib.util.spec_from_file_location(
            'hydrostatic_equilibrium_tools_step_test',
            tools_path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(example_dir))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)

        icparams = {
            'nogrid': 64,
            'coordsys': 'cartesian',
            'boxsize': 10.0 * unyt.cm,
            'time': 0.0 * unyt.s,
            'rho_ref': 1.0e-24 * unyt.g / unyt.cm**3,
            'tempini': 1.0e4 * unyt.K,
            'muini': 1.0,
            'gravity_strength': 1.0e-7 * unyt.cm / unyt.s**2,
        }

        simwrap = module.Simwrap(icparams)
        simwrap.fluid.eos = EOS('isothermal', gamma=1.0)
        simwrap.fluid.pre = simwrap.fluid.eos.pressure(
            simwrap.fluid.rho,
            simwrap.fluid.temp,
            simwrap.fluid.mu,
        )

        par = SimpleNamespace(
            noghost=2,
            nogrid=icparams['nogrid'],
            coordsys='cartesian',
            boundcond='Open',
            CFL=0.1,
            dtmin=1.0e-20 * unyt.s,
            dtmax=1.0 * unyt.s,
            order=0,
            gravity=Gravity(
                externalgravity=True,
                acceleration=module.constant_gravity_acceleration(
                    icparams['gravity_strength']
                ),
            ),
        )

        dx = simwrap.mesh.boundary[1] - simwrap.mesh.boundary[0]
        left_boundary = np.linspace(
            simwrap.mesh.boundary[0] - par.noghost * dx,
            simwrap.mesh.boundary[0] - dx,
            par.noghost,
        )
        right_boundary = np.linspace(
            simwrap.mesh.boundary[-1] + dx,
            simwrap.mesh.boundary[-1] + par.noghost * dx,
            par.noghost,
        )
        full_boundary = np.concatenate(
            (left_boundary, simwrap.mesh.boundary, right_boundary)
        )
        full_coordinate = 0.5 * (full_boundary[:-1] + full_boundary[1:])

        rho = simwrap.fluid.rho
        vel = simwrap.fluid.vel
        temp = simwrap.fluid.temp
        mu = simwrap.fluid.mu
        pre = simwrap.fluid.pre

        full_rho = np.concatenate(
            (
                np.ones(par.noghost) * rho[0],
                rho,
                np.ones(par.noghost) * rho[-1],
            )
        )
        full_vel = np.concatenate(
            (
                np.ones(par.noghost) * vel[0],
                vel,
                np.ones(par.noghost) * vel[-1],
            )
        )
        full_temp = np.concatenate(
            (
                np.ones(par.noghost) * temp[0],
                temp,
                np.ones(par.noghost) * temp[-1],
            )
        )
        full_mu = np.concatenate(
            (
                np.ones(par.noghost) * mu[0],
                mu,
                np.ones(par.noghost) * mu[-1],
            )
        )
        full_pre = np.concatenate(
            (
                np.ones(par.noghost) * pre[0],
                pre,
                np.ones(par.noghost) * pre[-1],
            )
        )

        mesh = SimpleNamespace(
            coordsys='cartesian',
            boundary=full_boundary,
            coordinate=full_coordinate,
            xdelta=full_boundary[1:] - full_boundary[:-1],
            area=np.ones(len(full_coordinate)) * (1.0 * unyt.cm**2),
            vol=(full_boundary[1:] - full_boundary[:-1]) * (1.0 * unyt.cm**2),
        )
        fluid = SimpleNamespace(
            rho=full_rho,
            vel=full_vel,
            temp=full_temp,
            mu=full_mu,
            pre=full_pre,
            eos=simwrap.fluid.eos,
            time=0.0 * unyt.s,
        )
        fluid.cs = fluid.eos.sound_speed(
            fluid.rho,
            fluid.pre,
            temp=fluid.temp,
            mu=fluid.mu,
        )
        fluid.vsignal = np.absolute(fluid.vel) + fluid.cs

        sim = Rsim.FromComponents(par, mesh, fluid, solver=Solver())

        rho_before = sim.fluid.rho.copy()
        vel_before = sim.fluid.vel.copy()
        pre_before = sim.fluid.pre.copy()
        dt = 1.0e-12 * unyt.s
        result = sim.Step(dt=dt, mode='hydro', advect_chemistry=False)

        interior = slice(par.noghost, par.noghost + par.nogrid)
        rho_rel = np.max(
            np.abs(
                (
                    sim.fluid.rho[interior] - rho_before[interior]
                ).to_value(rho_before.units)
                / rho_before[interior].to_value(rho_before.units)
            )
        )
        pre_rel = np.max(
            np.abs(
                (
                    sim.fluid.pre[interior] - pre_before[interior]
                ).to_value(pre_before.units)
                / pre_before[interior].to_value(pre_before.units)
            )
        )

        self.assertEqual(result['hydro_steps'], 1)
        self.assertEqual(sim.fluid.time, dt)
        self.assertLess(rho_rel, 1.0e-10)
        self.assertLess(pre_rel, 1.0e-10)
        self.assertLess(
            np.max(np.abs(sim.fluid.vel[interior] - vel_before[interior])).to_value(unyt.cm / unyt.s),
            1.0e-8,
        )

    def test_single_tiny_hydro_step_with_ssprk2_changes_state_only_slightly(self):
        example_dir = (
            Path(__file__).resolve().parents[1]
            / 'example'
            / 'HydrostaticEquilibrium1D'
        )
        tools_path = example_dir / 'tools.py'
        spec = importlib.util.spec_from_file_location(
            'hydrostatic_equilibrium_tools_ssprk2_test',
            tools_path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(example_dir))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)

        icparams = {
            'nogrid': 64,
            'coordsys': 'cartesian',
            'boxsize': 10.0 * unyt.cm,
            'time': 0.0 * unyt.s,
            'rho_ref': 1.0e-24 * unyt.g / unyt.cm**3,
            'tempini': 1.0e4 * unyt.K,
            'muini': 1.0,
            'gravity_strength': 1.0e-7 * unyt.cm / unyt.s**2,
        }

        simwrap = module.Simwrap(icparams)
        simwrap.fluid.eos = EOS('isothermal', gamma=1.0)
        simwrap.fluid.pre = simwrap.fluid.eos.pressure(
            simwrap.fluid.rho,
            simwrap.fluid.temp,
            simwrap.fluid.mu,
        )

        par = SimpleNamespace(
            noghost=2,
            nogrid=icparams['nogrid'],
            coordsys='cartesian',
            boundcond='Open',
            CFL=0.1,
            dtmin=1.0e-20 * unyt.s,
            dtmax=1.0 * unyt.s,
            order=0,
            gravity=Gravity(
                externalgravity=True,
                acceleration=module.constant_gravity_acceleration(
                    icparams['gravity_strength']
                ),
            ),
        )

        dx = simwrap.mesh.boundary[1] - simwrap.mesh.boundary[0]
        left_boundary = np.linspace(
            simwrap.mesh.boundary[0] - par.noghost * dx,
            simwrap.mesh.boundary[0] - dx,
            par.noghost,
        )
        right_boundary = np.linspace(
            simwrap.mesh.boundary[-1] + dx,
            simwrap.mesh.boundary[-1] + par.noghost * dx,
            par.noghost,
        )
        full_boundary = np.concatenate(
            (left_boundary, simwrap.mesh.boundary, right_boundary)
        )
        full_coordinate = 0.5 * (full_boundary[:-1] + full_boundary[1:])

        rho = simwrap.fluid.rho
        vel = simwrap.fluid.vel
        temp = simwrap.fluid.temp
        mu = simwrap.fluid.mu
        pre = simwrap.fluid.pre

        full_rho = np.concatenate(
            (
                np.ones(par.noghost) * rho[0],
                rho,
                np.ones(par.noghost) * rho[-1],
            )
        )
        full_vel = np.concatenate(
            (
                np.ones(par.noghost) * vel[0],
                vel,
                np.ones(par.noghost) * vel[-1],
            )
        )
        full_temp = np.concatenate(
            (
                np.ones(par.noghost) * temp[0],
                temp,
                np.ones(par.noghost) * temp[-1],
            )
        )
        full_mu = np.concatenate(
            (
                np.ones(par.noghost) * mu[0],
                mu,
                np.ones(par.noghost) * mu[-1],
            )
        )
        full_pre = np.concatenate(
            (
                np.ones(par.noghost) * pre[0],
                pre,
                np.ones(par.noghost) * pre[-1],
            )
        )

        mesh = SimpleNamespace(
            coordsys='cartesian',
            boundary=full_boundary,
            coordinate=full_coordinate,
            xdelta=full_boundary[1:] - full_boundary[:-1],
            area=np.ones(len(full_coordinate)) * (1.0 * unyt.cm**2),
            vol=(full_boundary[1:] - full_boundary[:-1]) * (1.0 * unyt.cm**2),
        )
        fluid = SimpleNamespace(
            rho=full_rho,
            vel=full_vel,
            temp=full_temp,
            mu=full_mu,
            pre=full_pre,
            eos=simwrap.fluid.eos,
            time=0.0 * unyt.s,
        )
        fluid.cs = fluid.eos.sound_speed(
            fluid.rho,
            fluid.pre,
            temp=fluid.temp,
            mu=fluid.mu,
        )
        fluid.vsignal = np.absolute(fluid.vel) + fluid.cs

        sim = Rsim.FromComponents(par, mesh, fluid, solver=Solver())

        rho_before = sim.fluid.rho.copy()
        vel_before = sim.fluid.vel.copy()
        pre_before = sim.fluid.pre.copy()
        dt = 1.0e-12 * unyt.s
        result = sim.Step(
            dt=dt,
            mode='hydro',
            advect_chemistry=False,
            hydro_integrator='ssprk2',
        )

        interior = slice(par.noghost, par.noghost + par.nogrid)
        rho_rel = np.max(
            np.abs(
                (
                    sim.fluid.rho[interior] - rho_before[interior]
                ).to_value(rho_before.units)
                / rho_before[interior].to_value(rho_before.units)
            )
        )
        pre_rel = np.max(
            np.abs(
                (
                    sim.fluid.pre[interior] - pre_before[interior]
                ).to_value(pre_before.units)
                / pre_before[interior].to_value(pre_before.units)
            )
        )

        self.assertEqual(result['hydro_steps'], 1)
        self.assertEqual(sim.fluid.time, dt)
        self.assertLess(rho_rel, 1.0e-10)
        self.assertLess(pre_rel, 1.0e-10)
        self.assertLess(
            np.max(np.abs(sim.fluid.vel[interior] - vel_before[interior])).to_value(unyt.cm / unyt.s),
            1.0e-8,
        )


if __name__ == '__main__':
    unittest.main()
