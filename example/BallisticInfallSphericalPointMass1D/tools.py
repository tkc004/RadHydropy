"""Helpers for spherical ballistic infall."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt
import radhydropy.io as rio
from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS
from radhydropy.rsim import Rsim
from radhydropy.units import code_quantity_to_cgs, code_unit_scales, quantity_to_value, time_seconds
from basic_hydro_utils import make_initial_condition

ACCELERATION_UNIT = unyt.cm / unyt.s**2

def spherical_cell_centers(boundary):
    coordinate = .5 * (boundary[1:] + boundary[:-1])
    denom = boundary[1:]**3 - boundary[:-1]**3
    mask = denom != 0
    coordinate[mask] = .75 * (boundary[1:][mask]**4 - boundary[:-1][mask]**4) / denom[mask]
    return coordinate

def point_mass_acceleration(point_mass, softening=0.0, code_units=None):
    scales = code_unit_scales(code_units) if code_units is not None else None
    mass = point_mass.to_value(unyt.g) if hasattr(point_mass, "to_value") else float(point_mass) * (scales["mass_g"] if scales else 1)
    soft = softening.to_value(unyt.cm) if hasattr(softening, "to_value") else float(softening) * (scales["length_cgs_cm"] if scales else 1)
    def acceleration(coordinate):
        radius = coordinate.to_value(code_units.length_unit) if hasattr(coordinate, "to_value") and code_units is not None else np.asarray(coordinate, dtype=float)
        if scales is not None: radius = radius * scales["length_cgs_cm"]
        radius = np.maximum(radius, soft)
        return (-GRAVITATIONAL_CONSTANT_CGS * mass / radius**2) * ACCELERATION_UNIT
    return acceleration

def ballistic_density_profile(coordinate, rho_ref):
    return np.ones(np.shape(coordinate), dtype=float) * rho_ref

def ballistic_velocity_profile(coordinate, point_mass, time, softening=0.0, code_units=None):
    t = time_seconds(time, code_units) * unyt.s if code_units is not None else float(time) * unyt.s
    return point_mass_acceleration(point_mass, softening, code_units)(coordinate) * t

def build_initial_condition(config):
    ic = config["initial_condition"]
    units = config["_code_units"]
    n = int(ic["grid_cells"])
    boundary = np.linspace(quantity_to_value(ic["inner_radius"], units.length_unit), quantity_to_value(ic["outer_radius"], units.length_unit), n + 1)
    center = spherical_cell_centers(boundary)
    return make_initial_condition(config, boundary,
        ballistic_density_profile(center, quantity_to_value(ic["reference_density"], units.density_unit)),
        np.zeros(n), np.full(n, quantity_to_value(ic["initial_temperature"], units.temperature_unit)),
        np.full(n, ic["mean_molecular_weight"]), area=4*np.pi*boundary[:-1]**2)

def ReadandPlot(filename, config, **kwargs):
    ic, units = config["initial_condition"], config["_code_units"]
    sim = Rsim(config["par"]); rio.readhdf5(sim.par, sim.mesh, sim.fluid, filename)
    first = int(sim.par.mesh.ghost_cells); last = first + int(sim.par.mesh.grid_cells)
    boundary = np.asarray(sim.mesh.boundary_proper_code, dtype=float)
    x = spherical_cell_centers(boundary)[first:last]
    rho = np.asarray(sim.fluid.rho_proper_code)[first:last]
    vel = np.asarray(sim.fluid.vel_proper_code)[first:last]
    time_code = float(np.asarray(sim.fluid.time_proper_code).flat[0])
    analytic_rho = ballistic_density_profile(
        x, quantity_to_value(ic["reference_density"], units.density_unit)
    )
    analytic_vel = ballistic_velocity_profile(
        x, ic["point_mass"], time_code, code_units=units
    ).to_value(unyt.cm / unyt.s) / units.velocity_in_cgs
    plt.subplot(1, 2, 1)
    plt.plot(x, rho, label="numerical", **kwargs)
    plt.plot(x, analytic_rho, color="black", linestyle="--", linewidth=2.0, label="analytic", zorder=5)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, vel, label="numerical", **kwargs)
    plt.plot(x, analytic_vel, color="black", linestyle="--", linewidth=2.0, label="free-fall", zorder=5)
    plt.legend()
