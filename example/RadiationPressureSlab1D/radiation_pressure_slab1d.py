"""One-dimensional slab accelerated by direct radiation pressure.

The incoming photon flux is transported from the left boundary.  The example
records the gas momentum and compares it with the momentum carried by the
absorbed photons, ``E_absorbed / c``.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
example_root = Path(__file__).resolve().parents[1]
if str(example_root) not in sys.path:
    sys.path.insert(0, str(example_root))

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "radhydropy-matplotlib"))

from radhydropy.example_config import load_example_parameters
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import radhydropy.io as rio
import example_utils as eu


DEFAULT_CONFIG = Path(__file__).resolve().with_name("radiation_pressure_slab1d.yaml")


class _Par:
    pass


class _Mesh:
    pass


class _Fluid:
    pass


def write_initial_condition(config, filename):
    code_units = CodeUnits.from_mapping(config["CodeUnits"])
    par = _Par()
    par.CodeUnits = code_units
    par.unit_system = code_units.unit_system
    par.nogrid = config["nogrid"]
    par.noghost = config.get("noghost", 2)
    par.coordsys = config["coordsys"]
    par.time = 0.0 * unyt.s
    par.boxsize = config["boxsize"]

    mesh = _Mesh()
    boxsize = config["boxsize"]
    dx = boxsize / par.nogrid
    mesh.boundary = np.linspace(0.0 * boxsize, boxsize, par.nogrid + 1)

    fluid = _Fluid()
    fluid.rho = np.ones(par.nogrid) * config["rhoini"]
    fluid.vel = np.ones(par.nogrid) * config["vini"]
    fluid.temp = np.ones(par.nogrid) * config["tempini"]
    fluid.mu = np.ones(par.nogrid) * config["muini"]

    rio.writehdf5(
        type("InitialCondition", (), {"par": par, "mesh": mesh, "fluid": fluid})(),
        filename,
    )


def _total_momentum(fluid, par):
    interior = slice(par.noghost, par.noghost + par.nogrid)
    return float(np.sum(np.asarray(fluid.Mom[interior], dtype=float)))


def _absorbed_momentum(source_result, mesh, par, dt):
    absorbed = source_result.get("absorbed_photon_rate")
    energies = source_result.get("photon_energy_erg")
    if absorbed is None or energies is None:
        return 0.0
    absorbed = np.asarray(absorbed, dtype=float)
    if absorbed.ndim == 1:
        absorbed = absorbed[None, :]
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    interior = slice(par.noghost, par.noghost + par.nogrid)
    volume = np.asarray(mesh.vol[interior], dtype=float)
    absorbed_energy = np.sum(absorbed * energies[:, None], axis=0)
    direction = float(source_result.get("direction", 1))
    return direction * float(np.sum(absorbed_energy * volume * dt) / unyt.c.to_value(unyt.cm / unyt.s))


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    runparams, icparams = load_example_parameters(config_filename, rundir)
    eu.clean_previous_outputs(runparams)
    config = {**runparams, **icparams}
    write_initial_condition(config, runparams["ICfilename"])

    sim = Rsim(runparams)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    time_s = [0.0]
    gas_momentum = [_total_momentum(sim.fluid, sim.par)]
    expected_momentum = [0.0]
    expected = 0.0
    sim.solver.GetTimeStep(sim.mesh, sim.fluid, sim.par)

    while float(np.asarray(sim.fluid.time)) < float(np.asarray(sim.par.timesim)):
        remaining = float(np.asarray(sim.par.timesim)) - float(np.asarray(sim.fluid.time))
        # Keep this demonstration on a fixed, conservative source timestep so
        # the momentum-budget comparison is not obscured by a CFL diagnostic.
        dt = min(float(np.asarray(sim.par.dtmax)), remaining)

        sim.PrepareConservedStep()
        old_mass, mass_flux = sim.AdvanceHydroFluxes(dt)
        sim.FinalizeHydroStep(dt, old_mass, mass_flux)

        source_result = sim.ApplyThermochemistrySources(dt)
        sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
        sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
        sim.solver.ApplyRadiationPressure(
            dt,
            sim.mesh,
            sim.fluid,
            sim.par,
            source_result,
        )
        sim._sync_hydro_state()
        expected += _absorbed_momentum(source_result, sim.mesh, sim.par, dt)

        time_s.append(float(np.asarray(sim.fluid.time)))
        gas_momentum.append(_total_momentum(sim.fluid, sim.par))
        expected_momentum.append(expected)

    time = np.asarray(time_s) * unyt.s
    gas = np.asarray(gas_momentum) * (unyt.g * unyt.cm / unyt.s)
    expected = np.asarray(expected_momentum) * (unyt.g * unyt.cm / unyt.s)
    figure = Path(runparams["savedir"]) / "RadiationPressureSlab1D_Momentum.jpg"
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(time.to_value(unyt.s), gas.to_value(unyt.g * unyt.cm / unyt.s), label="gas momentum")
    plt.plot(time.to_value(unyt.s), expected.to_value(unyt.g * unyt.cm / unyt.s), "--", label="absorbed photons / c")
    plt.xlabel("time [s]")
    plt.ylabel(r"total momentum [g cm s$^{-1}$]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure, dpi=180)
    plt.close()

    relative_error = abs(gas[-1] - expected[-1]) / max(abs(expected[-1]), 1.0e-300 * gas[-1].units)
    print("final gas momentum = %.6e g cm/s" % gas[-1].to_value(unyt.g * unyt.cm / unyt.s))
    print("expected momentum = %.6e g cm/s" % expected[-1].to_value(unyt.g * unyt.cm / unyt.s))
    print("relative error = %.6e" % float(relative_error))
    print("figure = %s" % figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the radiation-pressure slab example.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
