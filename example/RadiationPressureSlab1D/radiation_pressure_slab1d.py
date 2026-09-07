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

from radhydropy.rsim import Rsim
from radhydropy.arrays import as_named_array
from radhydropy.runtime_fields import MeshGeometryState, PROPER_RUNTIME_FIELDS
from radhydropy.units import quantity_to_value
import radhydropy.io as rio
import example_utils as eu


DEFAULT_CONFIG = Path(__file__).resolve().with_name("radiation_pressure_slab1d.yaml")


def build_initial_condition(config):
    par_config = config['par']
    initial = config['initial_condition']
    grid_cells = int(par_config['mesh']['grid_cells'])
    sim = Rsim(par_config)
    code_units = sim.par.units.CodeUnits
    sim.par.simulation.box_size = quantity_to_value(
        initial['box_size'], code_units.length_unit
    )
    sim.par.simulation.time_code = quantity_to_value(
        initial.get('current_time', 0.0 * unyt.s), code_units.time_unit
    )
    sim.mesh.boundary_proper_code = as_named_array(quantity_to_value(
        np.linspace(0.0, initial['box_size'].to_value(unyt.cm), grid_cells + 1) * unyt.cm,
        code_units.length_unit,
    ))
    boundary = sim.mesh.boundary_proper_code
    width = np.diff(boundary)
    area = np.ones(grid_cells) * quantity_to_value(
        par_config['mesh']['area'], code_units.area_unit
    )
    volume = width * area
    coordinate = 0.5 * (boundary[1:] + boundary[:-1])
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        coordinate=coordinate,
        boundary=boundary,
        width=width,
        area=area,
        volume=volume,
    )
    sim.fluid.rho_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * initial['initial_density'], code_units.density_unit
    ))
    sim.fluid.vel_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * initial['velocity'], code_units.velocity_unit
    ))
    sim.fluid.temp_proper_code = as_named_array(quantity_to_value(
        np.ones(grid_cells) * initial['temperature'], code_units.temperature_unit
    ))
    sim.fluid.mu = np.ones(grid_cells) * initial['mean_molecular_weight']
    sim.fluid.xHI = np.ones(
        grid_cells
    ) * par_config['chemistry']['hydrogen_xHI_initial']
    sim.fluid.SetFluidTime(sim.par.simulation.time_code)
    sim.fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    sim.fluid.SetPressure()
    sim.fluid._refresh_runtime_state()
    return sim


def write_initial_condition(config):
    sim = build_initial_condition(config)
    rio.writehdf5(sim, config['par']['simulation']['initial_condition_filename'])


def _total_momentum(fluid, par):
    interior = slice(par.mesh.ghost_cells, par.mesh.ghost_cells + par.mesh.grid_cells)
    return float(np.sum(np.asarray(fluid.Mom_code[interior], dtype=float)))


def _absorbed_momentum(source_result, mesh, par, dt):
    absorbed = source_result.get("absorbed_photon_rate")
    energies = source_result.get("photon_energy_cgs_erg")
    if absorbed is None or energies is None:
        return 0.0
    absorbed = np.asarray(absorbed, dtype=float)
    if absorbed.ndim == 1:
        absorbed = absorbed[None, :]
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    interior = slice(par.mesh.ghost_cells, par.mesh.ghost_cells + par.mesh.grid_cells)
    volume = np.asarray(mesh.geometry_state.volume_proper_code[interior], dtype=float)
    absorbed_energy = np.sum(absorbed * energies[:, None], axis=0)
    direction = float(source_result.get("direction", 1))
    return direction * float(np.sum(absorbed_energy * volume * dt) / unyt.c.to_value(unyt.cm / unyt.s))


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    config = eu.load_nested_example_config(config_filename)
    runtime = config['par']
    eu.clean_previous_outputs(runtime)
    write_initial_condition(config)

    sim = Rsim(runtime)
    rio.readhdf5(
        sim.par,
        sim.mesh,
        sim.fluid,
        runtime['simulation']['initial_condition_filename'],
    )
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    time_s = [0.0]
    gas_momentum = [_total_momentum(sim.fluid, sim.par)]
    expected_momentum = [0.0]
    expected = 0.0
    sim.solver.GetTimeStep(sim.mesh, sim.fluid, sim.par)

    final_time = runtime["simulation"]["final_time"].to_value(unyt.s)
    dtmax = runtime["timestep"]["dtmax"].to_value(unyt.s)
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
        expected += _absorbed_momentum(source_result, sim.mesh, sim.par, dt)

        time_s.append(float(np.asarray(sim.fluid.time_proper_code)))
        gas_momentum.append(_total_momentum(sim.fluid, sim.par))
        expected_momentum.append(expected)

    time = np.asarray(time_s) * unyt.s
    gas = np.asarray(gas_momentum) * (unyt.g * unyt.cm / unyt.s)
    expected = np.asarray(expected_momentum) * (unyt.g * unyt.cm / unyt.s)
    figure = Path(runtime["output"]["savedir"]) / "RadiationPressureSlab1D_Momentum.jpg"
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
