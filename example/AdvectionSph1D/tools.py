"""Initial conditions and plotting for spherical advection."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.units import quantity_to_value
from basic_hydro_utils import make_initial_condition
import advection_sph_analytic as asa


def build_initial_condition(config):
    initial = config["initial_condition"]
    units = config["_code_units"]
    n = int(initial["grid_cells"])
    size = quantity_to_value(initial["box_size"], units.length_unit)
    boundary_proper_code = np.linspace(size / n, size + size / n, n + 1)
    coordinate_proper_code = 0.5 * (boundary_proper_code[:-1] + boundary_proper_code[1:])
    rho_proper_code = np.full(n, quantity_to_value(initial["initial_density"], units.density_unit))
    rho_proper_code[(coordinate_proper_code < .25 * size) | (coordinate_proper_code > .75 * size)] *= .01
    return make_initial_condition(config, boundary_proper_code=boundary_proper_code, rho_proper_code=rho_proper_code,
        vel_proper_code=np.full(n, quantity_to_value(initial["initial_velocity"], units.velocity_unit)),
        temp_proper_code=np.full(n, quantity_to_value(initial["initial_temperature"], units.temperature_unit)),
        mu_dimensionless=np.full(n, initial["mean_molecular_weight"]))


def ReadandPlot(filename, config, **kwargs):
    sim = Rsim(config["par"])
    rio.readhdf5(sim.par, sim.mesh, sim.fluid, filename)
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    boundary_proper_code = np.asarray(sim.mesh.boundary_proper_code, dtype=float)
    coordinate_proper_code = .5 * (boundary_proper_code[:-1] + boundary_proper_code[1:])
    units = config["_code_units"]
    time_proper_code = float(np.asarray(sim.fluid.time_proper_code).flat[0])
    analytic_density = asa.top_hat_density_profile(
        coordinate_proper_code[first:last],
        time_proper_code,
        quantity_to_value(config["initial_condition"]["initial_velocity"], units.velocity_unit),
        quantity_to_value(config["initial_condition"]["box_size"], units.length_unit),
        quantity_to_value(config["initial_condition"]["initial_density"], units.density_unit),
    )
    plt.plot(coordinate_proper_code[first:last] * units.length_unit,
             np.asarray(sim.fluid.rho_proper_code)[first:last] * units.density_unit,
             **kwargs)
    plt.plot(
        coordinate_proper_code[first:last] * units.length_unit,
        analytic_density * units.density_unit,
        color=kwargs.get("color"),
        linestyle="--",
        label="analytic",
    )
