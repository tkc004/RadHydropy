import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from tests.parameter_fixtures import parameter_namespace

import numpy as np
import unyt

from radhydropy.arrays import as_named_array
from radhydropy.eos import EOS
from radhydropy.gravity import Gravity
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
from radhydropy.units import CodeUnits


def _to_float(value, unit=None):
    if hasattr(value, "to_value") and unit is not None:
        return np.asarray(value.to_value(unit), dtype=float)
    return np.asarray(value, dtype=float)


def _code_units():
    return CodeUnits.from_mapping(
        {
            "UnitMass_in_cgs": 1.0,
            "UnitLength_in_cgs": 1.0,
            "UnitVelocity_in_cgs": 1.0,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        }
    )


def _floatify_hydrostatic_simwrap(simwrap, code_units):
    simwrap.fluid.eos = EOS("isothermal", gamma=1.0, code_units=code_units)
    simwrap.mesh.boundary = _to_float(simwrap.mesh.boundary, code_units.length_unit)
    simwrap.mesh.coordinate = _to_float(
        simwrap.mesh.coordinate,
        code_units.length_unit,
    )
    simwrap.mesh.area = _to_float(simwrap.mesh.area, code_units.area_unit)
    simwrap.mesh.vol = _to_float(simwrap.mesh.vol, code_units.volume_unit)
    simwrap.fluid.rho_code = as_named_array(
        _to_float(simwrap.fluid.rho_code, code_units.density_unit)
    )
    simwrap.fluid.temp_code = as_named_array(
        _to_float(simwrap.fluid.temp_code, code_units.temperature_unit)
    )
    simwrap.fluid.mu = as_named_array(_to_float(simwrap.fluid.mu))
    simwrap.fluid.vel_code = as_named_array(
        _to_float(simwrap.fluid.vel_code, code_units.velocity_unit)
    )
    simwrap.fluid.pre_code = as_named_array(
        _to_float(
            simwrap.fluid.eos.pressure(
                simwrap.fluid.rho_code,
                simwrap.fluid.temp_code,
                simwrap.fluid.mu,
            ),
            code_units.pressure_unit,
        )
    )
    return simwrap


def _load_hydrostatic_tools():
    example_dir = (
        Path(__file__).resolve().parents[1]
        / "example"
        / "HydrostaticEquilibrium1D"
    )
    tools_path = example_dir / "tools.py"
    spec = importlib.util.spec_from_file_location(
        "hydrostatic_equilibrium_tools_test",
        tools_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(example_dir))
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _hydrostatic_base_icparams(nogrid):
    return {
        "nogrid": nogrid,
        "coordsys": "cartesian",
        "box_size": 10.0 * unyt.cm,
        "current_time": 0.0 * unyt.s,
        "reference_density": 1.0e-24 * unyt.g / unyt.cm**3,
        "initial_temperature": 1.0e4 * unyt.K,
        "mean_molecular_weight": 1.0,
        "gravity_strength": 1.0e-7 * unyt.cm / unyt.s**2,
    }


def _build_hydrostatic_step_sim(nogrid, integrator=None):
    module = _load_hydrostatic_tools()
    icparams = _hydrostatic_base_icparams(nogrid)
    code_units = _code_units()

    simwrap = _floatify_hydrostatic_simwrap(
        module.build_initial_condition({
            "par": {"mesh": {"grid_cells": nogrid}},
            "initial_condition": icparams,
            "example": {},
            "_code_units": code_units,
        }),
        code_units,
    )
    par = parameter_namespace(
        noghost=2,
        nogrid=icparams["nogrid"],
        coordsys="cartesian",
        boundcond="Open",
        CFL=0.1,
        dtmin=1.0e-20,
        dtmax=1.0,
        order=0,
        riemann_solver='Rusanov',
        gravity=Gravity(
            externalgravity=True,
            acceleration=module.constant_gravity_acceleration(
                icparams["gravity_strength"],
                code_units=code_units,
            ),
            code_units=code_units,
        ),
    )

    dx = np.asarray(simwrap.mesh.boundary[1] - simwrap.mesh.boundary[0], dtype=float)
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
    full_boundary = np.concatenate((left_boundary, simwrap.mesh.boundary, right_boundary))
    full_coordinate = 0.5 * (full_boundary[:-1] + full_boundary[1:])

    rho_code = np.asarray(simwrap.fluid.rho_code, dtype=float)
    vel_code = np.asarray(simwrap.fluid.vel_code, dtype=float)
    temp_code = np.asarray(simwrap.fluid.temp_code, dtype=float)
    mu = np.asarray(simwrap.fluid.mu, dtype=float)
    pre_code = np.asarray(simwrap.fluid.pre_code, dtype=float)

    full_rho = as_named_array(
        np.concatenate(
            (
                np.ones(par.noghost, dtype=float) * rho_code[0],
                rho_code,
                np.ones(par.noghost, dtype=float) * rho_code[-1],
            )
        )
    )
    full_vel = as_named_array(
        np.concatenate(
            (
                np.ones(par.noghost, dtype=float) * vel_code[0],
                vel_code,
                np.ones(par.noghost, dtype=float) * vel_code[-1],
            )
        )
    )
    full_temp = as_named_array(
        np.concatenate(
            (
                np.ones(par.noghost, dtype=float) * temp_code[0],
                temp_code,
                np.ones(par.noghost, dtype=float) * temp_code[-1],
            )
        )
    )
    full_mu = as_named_array(
        np.concatenate(
            (
                np.ones(par.noghost, dtype=float) * mu[0],
                mu,
                np.ones(par.noghost, dtype=float) * mu[-1],
            )
        )
    )
    full_pre = as_named_array(
        np.concatenate(
            (
                np.ones(par.noghost, dtype=float) * pre_code[0],
                pre_code,
                np.ones(par.noghost, dtype=float) * pre_code[-1],
            )
        )
    )

    mesh = SimpleNamespace(
        coordsys="cartesian",
        boundary=full_boundary,
        coordinate=full_coordinate,
        xdelta=full_boundary[1:] - full_boundary[:-1],
        area=np.ones(len(full_coordinate), dtype=float),
        vol=full_boundary[1:] - full_boundary[:-1],
    )
    fluid = SimpleNamespace(
        rho_code=full_rho,
        vel_code=full_vel,
        temp_code=full_temp,
        mu=full_mu,
        pre_code=full_pre,
        eos=simwrap.fluid.eos,
        time_code=0.0,
    )
    fluid.cs_code = as_named_array(
        np.asarray(
            fluid.eos.sound_speed(
                fluid.rho_code,
                fluid.pre_code,
                temp=fluid.temp_code,
                mu=fluid.mu,
            ),
            dtype=float,
        )
    )
    fluid.vsignal_code = as_named_array(np.absolute(fluid.vel_code) + fluid.cs_code)

    solver = Solver()
    solver.SetConserved(mesh, fluid)
    sim = Rsim.FromComponents(par, mesh, fluid, solver=solver)
    if integrator is not None:
        return module, sim, par, icparams, code_units, fluid, integrator
    return module, sim, par, icparams, code_units, fluid


class Testing(unittest.TestCase):
    def test_hydrostatic_equilibrium_profile_balances_gravity(self):
        module = _load_hydrostatic_tools()
        icparams = {
            "nogrid": 256,
            "coordsys": "cartesian",
            "box_size": 10.0 * unyt.pc,
            "current_time": 0.0 * unyt.s,
            "reference_density": 1.0e-24 * unyt.g / unyt.cm**3,
            "initial_temperature": 1.0e4 * unyt.K,
            "mean_molecular_weight": 1.0,
            "gravity_strength": 1.0e-7 * unyt.cm / unyt.s**2,
        }
        code_units = _code_units()

        sim = _floatify_hydrostatic_simwrap(
            module.build_initial_condition({
                "par": {"mesh": {"grid_cells": icparams["nogrid"]}},
                "initial_condition": icparams,
                "example": {},
                "_code_units": code_units,
            }),
            code_units,
        )
        pressure = np.asarray(
            sim.fluid.eos.pressure(sim.fluid.rho_code, sim.fluid.temp_code, sim.fluid.mu),
            dtype=float,
        )
        coordinate = np.asarray(sim.mesh.coordinate, dtype=float)
        dPdx = np.gradient(pressure, coordinate)
        gravity_strength = _to_float(
            icparams["gravity_strength"],
            code_units.length_unit / code_units.time_unit**2,
        )
        expected = -np.asarray(sim.fluid.rho_code, dtype=float) * gravity_strength

        interior = slice(2, -2)
        np.testing.assert_allclose(
            dPdx[interior],
            expected[interior],
            rtol=2.0e-3,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(sim.fluid.vel_code, dtype=float),
            0.0,
            atol=0.0,
        )

    def test_single_tiny_hydro_step_changes_state_only_slightly(self):
        module, sim, par, icparams, _, fluid = _build_hydrostatic_step_sim(64)

        rho_before = sim.fluid.rho_code.copy()
        vel_code_before = sim.fluid.vel_code.copy()
        pre_before = sim.fluid.pre_code.copy()
        dt = 1.0e-12
        result = sim.Step(dt=dt, mode="hydro", advect_chemistry=False)

        interior = slice(par.noghost, par.noghost + par.nogrid)
        rho_rel = np.max(
            np.abs((sim.fluid.rho_code[interior] - rho_before[interior]) / rho_before[interior])
        )
        pre_rel = np.max(
            np.abs((sim.fluid.pre_code[interior] - pre_before[interior]) / pre_before[interior])
        )

        self.assertEqual(result["hydro_steps"], 1)
        self.assertEqual(sim.fluid.time_code, dt)
        self.assertLess(rho_rel, 1.0e-10)
        self.assertLess(pre_rel, 1.0e-10)
        self.assertLess(
            np.max(np.abs(sim.fluid.vel_code[interior] - vel_code_before[interior])),
            1.0e-8,
        )

    def test_single_tiny_hydro_step_with_ssprk2_changes_state_only_slightly(self):
        module, sim, par, icparams, _, fluid = _build_hydrostatic_step_sim(64)

        rho_before = sim.fluid.rho_code.copy()
        vel_code_before = sim.fluid.vel_code.copy()
        pre_before = sim.fluid.pre_code.copy()
        dt = 1.0e-12
        result = sim.Step(
            dt=dt,
            mode="hydro",
            advect_chemistry=False,
            hydro_integrator="ssprk2",
        )

        interior = slice(par.noghost, par.noghost + par.nogrid)
        rho_rel = np.max(
            np.abs((sim.fluid.rho_code[interior] - rho_before[interior]) / rho_before[interior])
        )
        pre_rel = np.max(
            np.abs((sim.fluid.pre_code[interior] - pre_before[interior]) / pre_before[interior])
        )

        self.assertEqual(result["hydro_steps"], 1)
        self.assertEqual(sim.fluid.time_code, dt)
        self.assertLess(rho_rel, 1.0e-10)
        self.assertLess(pre_rel, 1.0e-10)
        self.assertLess(
            np.max(np.abs(sim.fluid.vel_code[interior] - vel_code_before[interior])),
            1.0e-8,
        )


if __name__ == "__main__":
    unittest.main()
