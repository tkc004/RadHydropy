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

def spherical_cell_centers(boundary_proper_code):
    x_proper_code = .5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
    denom = boundary_proper_code[1:]**3 - boundary_proper_code[:-1]**3
    mask = denom != 0
    x_proper_code[mask] = .75 * (boundary_proper_code[1:][mask]**4 - boundary_proper_code[:-1][mask]**4) / denom[mask]
    return x_proper_code

def point_mass_acceleration(point_mass, softening=0.0, code_unit_system=None):
    scales = code_unit_scales(code_unit_system) if code_unit_system is not None else None
    point_mass_g = point_mass.to_value(unyt.g) if hasattr(point_mass, "to_value") else float(point_mass) * (scales["mass_g"] if scales else 1)
    soft = softening.to_value(unyt.cm) if hasattr(softening, "to_value") else float(softening) * (scales["length_cgs_cm"] if scales else 1)
    def acceleration(x_proper_code):
        radius_proper_cgs_cm = x_proper_code.to_value(code_unit_system.length_unit) if hasattr(x_proper_code, "to_value") and code_unit_system is not None else np.asarray(x_proper_code, dtype=float)
        if scales is not None: radius_proper_cgs_cm = radius_proper_cgs_cm * scales["length_cgs_cm"]
        radius_proper_cgs_cm = np.maximum(radius_proper_cgs_cm, soft)
        return (-GRAVITATIONAL_CONSTANT_CGS * point_mass_g / radius_proper_cgs_cm**2) * ACCELERATION_UNIT
    return acceleration

def ballistic_density_profile(x_proper_code, rho_reference_proper_code):
    return np.ones(np.shape(x_proper_code), dtype=float) * rho_reference_proper_code

def ballistic_velocity_profile(x_proper_code, point_mass, time_proper_code, softening=0.0, code_unit_system=None):
    t = time_seconds(time_proper_code, code_unit_system) * unyt.s if code_unit_system is not None else float(time_proper_code) * unyt.s
    return point_mass_acceleration(point_mass, softening, code_unit_system)(x_proper_code) * t

def build_initial_condition(config):
    ic = config["initial_condition"]
    units = config["_code_units"]
    n = int(ic["grid_cells"])
    boundary_proper_code = np.linspace(quantity_to_value(ic["radius_inner_proper"], units.length_unit), quantity_to_value(ic["radius_outer_proper"], units.length_unit), n + 1)
    coordinate_proper_code = spherical_cell_centers(boundary_proper_code)
    return make_initial_condition(config, boundary_proper_code=boundary_proper_code,
        rho_proper_code=ballistic_density_profile(coordinate_proper_code, quantity_to_value(ic["rho_reference_proper"], units.density_unit)),
        vel_proper_code=np.zeros(n), temp_proper_code=np.full(n, quantity_to_value(ic["temperature_proper"], units.temperature_unit)),
        mu_dimensionless=np.full(n, ic["mean_molecular_weight"]), area_proper_code=4*np.pi*boundary_proper_code[:-1]**2)

def plot_snapshot(filename, config, **kwargs):
    ic, units = config["initial_condition"], config["_code_units"]
    sim = Rsim(config["par"]); rio.readhdf5(sim.par, sim.mesh, sim.fluid, filename)
    first = int(sim.par.mesh.ghost_cells); last = first + int(sim.par.mesh.grid_cells)
    boundary_proper_code = np.asarray(sim.mesh.boundary_proper_code, dtype=float)
    coordinate_proper_code = spherical_cell_centers(boundary_proper_code)[first:last]
    rho_proper_code = np.asarray(sim.fluid.rho_proper_code)[first:last]
    vel_proper_code = np.asarray(sim.fluid.vel_proper_code)[first:last]
    time_proper_code = float(np.asarray(sim.fluid.time_proper_code).flat[0])
    analytic_rho = ballistic_density_profile(
        coordinate_proper_code, quantity_to_value(ic["rho_reference_proper"], units.density_unit)
    )
    analytic_vel = ballistic_velocity_profile(
        coordinate_proper_code, ic["point_mass"], time_proper_code,
        code_unit_system=units,
    ).to_value(unyt.cm / unyt.s) / units.velocity_in_cgs
    plt.subplot(1, 2, 1)
    plt.plot(coordinate_proper_code, rho_proper_code, label="numerical", **kwargs)
    plt.plot(coordinate_proper_code, analytic_rho, color="black", linestyle="--", linewidth=2.0, label="analytic", zorder=5)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(coordinate_proper_code, vel_proper_code, label="numerical", **kwargs)
    plt.plot(coordinate_proper_code, analytic_vel, color="black", linestyle="--", linewidth=2.0, label="free-fall", zorder=5)
    plt.legend()
