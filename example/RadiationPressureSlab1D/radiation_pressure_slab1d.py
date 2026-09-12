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

from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, quantity_to_value
import radhydropy.io as rio
import example_utils as eu


DEFAULT_CONFIG = Path(__file__).resolve().with_name("radiation_pressure_slab1d.yaml")


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config["_code_units"]
    grid_cells = int(config["par"]['mesh']['grid_cells'])
    writer = InitialConditionWriter(par_config=config["par"], code_units=code_units)
    writer.box_size = writer.radquantity(initial['box_size_proper'])
    writer.mesh.boundary_radarray = writer.radarray(
        np.linspace(0.0, 1.0, grid_cells + 1) * initial['box_size_proper']
    )
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(grid_cells) * initial['rho_proper']
    )
    writer.fluid.vel_radarray = writer.radarray(
        np.ones(grid_cells) * initial['vel_proper']
    )
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(grid_cells) * initial['temperature_proper']
    )
    writer.simulation.fluid.mu = np.full(
        grid_cells, initial['mean_molecular_weight']
    )
    writer.simulation.fluid.xHI = np.full(
        grid_cells, config["par"]['chemistry']['hydrogen_xHI_initial']
    )
    writer.simulation.par.simulation.time_proper_code = quantity_to_value(
        initial.get('time_proper', 0.0 * unyt.s), code_units.time_unit
    )
    return writer


def write_initial_condition(config):
    writer = build_initial_condition(config)
    writer.write(config['par']['simulation']['initial_condition_filename'], validate=True)


def _total_momentum(fluid, config):
    par = config['_runtime_par']
    interior = slice(par.mesh.ghost_cells, par.mesh.ghost_cells + par.mesh.grid_cells)
    return float(np.sum(np.asarray(fluid.Mom_code[interior], dtype=float)))


def _absorbed_momentum(source_result, mesh, config, dt):
    par = config['_runtime_par']
    absorbed = source_result.get("absorbed_photon_rate")
    energies = source_result.get("photon_energy_cgs_erg")
    if absorbed is None or energies is None:
        return 0.0
    absorbed = np.asarray(absorbed, dtype=float)
    if absorbed.ndim == 1:
        absorbed = absorbed[None, :]
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    interior = slice(par.mesh.ghost_cells, par.mesh.ghost_cells + par.mesh.grid_cells)
    volume_proper_code = np.asarray(mesh.geometry_state.volume_proper_code[interior], dtype=float)
    absorbed_energy = np.sum(absorbed * energies[:, None], axis=0)
    direction = float(source_result.get("direction", 1))
    return direction * float(np.sum(absorbed_energy * volume_proper_code * dt) / unyt.c.to_value(unyt.cm / unyt.s))


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)
    config['_code_units'] = CodeUnits.from_mapping(config['par']['units']['CodeUnits'])

    eu.clean_previous_outputs(config)
    write_initial_condition(config)

    sim = rio.loadhdf5(
        config, config["par"]['simulation']['initial_condition_filename']
    )
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
    config['_runtime_par'] = sim.par

    time_s = [0.0]
    gas_momentum = [_total_momentum(sim.fluid, config)]
    expected_momentum = [0.0]
    expected = 0.0
    sim.solver.GetTimeStep(sim.mesh, sim.fluid, sim.par)

    final_time = config["par"]["simulation"]["final_time"].to_value(unyt.s)
    dtmax = config["par"]["timestep"]["dtmax"].to_value(unyt.s)
    while float(np.asarray(sim.fluid.time_proper_code)) < final_time:
        remaining = final_time - float(np.asarray(sim.fluid.time_proper_code))
        # Keep this demonstration on a fixed, conservative source timestep so
        # the momentum-budget comparison is not obscured by a CFL diagnostic.
        dt = min(float(dtmax), remaining)

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
        expected += _absorbed_momentum(source_result, sim.mesh, config, dt)

        time_s.append(float(np.asarray(sim.fluid.time_proper_code)))
        gas_momentum.append(_total_momentum(sim.fluid, config))
        expected_momentum.append(expected)

    time_cgs_s_unyt = np.asarray(time_s) * unyt.s
    gas = np.asarray(gas_momentum) * (unyt.g * unyt.cm / unyt.s)
    expected = np.asarray(expected_momentum) * (unyt.g * unyt.cm / unyt.s)
    figure = Path(config["par"]["output"]["directory"]) / "RadiationPressureSlab1D_Momentum.jpg"
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(time_cgs_s_unyt.to_value(unyt.s), gas.to_value(unyt.g * unyt.cm / unyt.s), label="gas momentum")
    plt.plot(time_cgs_s_unyt.to_value(unyt.s), expected.to_value(unyt.g * unyt.cm / unyt.s), "--", label="absorbed photons / c")
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
