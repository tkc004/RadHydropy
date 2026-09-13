"""Initial conditions and plotting for spherical advection."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt
import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import quantity_to_value
import advection_sph_analytic as asa


def build_initial_condition(config):
    initial = config["initial_condition"]
    units = config["_code_units"]
    n = int(initial["grid_cells"])
    box_size_proper_unyt = initial["box_size_proper"]
    boundary_proper_unyt = np.linspace(0.0, 1.0, n + 1) * box_size_proper_unyt
    coordinate_proper_unyt = 0.5 * (boundary_proper_unyt[:-1] + boundary_proper_unyt[1:])
    rho_proper_unyt = np.ones(n) * initial["rho_proper"]
    rho_proper_unyt[(coordinate_proper_unyt < .25 * box_size_proper_unyt) | (coordinate_proper_unyt > .75 * box_size_proper_unyt)] *= .01
    writer = InitialConditionWriter(par_config=config["par"], code_units=units, ic_config=config["initial_condition"])
    writer.box_size = writer.radquantity(box_size_proper_unyt)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(np.ones(n) * initial["vel_proper"])
    writer.fluid.temp_radarray = writer.radarray(np.ones(n) * initial["temperature_proper"])
    writer.simulation.fluid.mu = np.full(n, float(initial["mean_molecular_weight"]))
    return writer


def plot_snapshot(filename, config, **kwargs):
    sim = rio.loadhdf5(config, filename)
    units = config["_code_units"]
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    boundary_proper_code = np.asarray(
        sim.mesh.boundary_radarray.to_value(units.length_unit), dtype=float
    )
    coordinate_proper_code = .5 * (boundary_proper_code[:-1] + boundary_proper_code[1:])
    time_proper_code = float(np.asarray(sim.fluid.time_proper_code).flat[0])
    analytic_density = asa.top_hat_density_profile(
        coordinate_proper_code[first:last],
        time_proper_code,
        quantity_to_value(config["initial_condition"]["vel_proper"], units.velocity_unit),
        quantity_to_value(config["initial_condition"]["box_size_proper"], units.length_unit),
        quantity_to_value(config["initial_condition"]["rho_proper"], units.density_unit),
    )
    rho_snapshot_proper_code = sim.fluid.rho_radarray.to_value(units.density_unit)
    plt.plot(coordinate_proper_code[first:last] * units.length_unit,
             rho_snapshot_proper_code[first:last] * units.density_unit,
             **kwargs)
    plt.plot(
        coordinate_proper_code[first:last] * units.length_unit,
        analytic_density * units.density_unit,
        color=kwargs.get("color"),
        linestyle="--",
        label="analytic",
    )
