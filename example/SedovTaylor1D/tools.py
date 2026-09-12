"""Initial conditions and plotting for the cartesian Sedov benchmark."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt
import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import quantity_to_value
from example.SedovTaylor1D import SedovTaylor_analytic as sa

def set_plot_style():
    plt.rcParams.update({"figure.figsize": (18, 6), "font.size": 14})

def build_initial_condition(config):
    ic, par, units = config["initial_condition"], config["par"], config["_code_units"]
    n = int(ic["grid_cells"])
    size_proper_unyt = ic["box_size_proper"]
    boundary_proper_unyt = np.linspace(0.0, 1.0, n + 1) * size_proper_unyt
    rho_proper_unyt = np.ones(n) * ic["rho_proper"]
    mu_dimensionless = np.full(n, float(ic["mean_molecular_weight"]))
    temp_proper_unyt = np.zeros(n) * unyt.K
    volume_proper_unyt = par["mesh"]["area_proper"] * np.diff(boundary_proper_unyt)
    cut = 1
    # Convert the deposited explosion energy into the pressure of the
    # injection cell; the writer will later reconstruct pressure from T.
    pressure_proper_unyt = (
        (par["hydrodynamics"]["gamma"] - 1)
        * ic["explosion_energy"]
        / volume_proper_unyt[cut]
    )
    writer = InitialConditionWriter(par_config=par, code_units=units)
    # Invert the configured EOS so the temperature and deposited energy are
    # thermodynamically consistent when the writer prepares the IC.
    temp_proper_unyt[cut] = (
        pressure_proper_unyt
        * mu_dimensionless[cut]
        * unyt.mp
        / (rho_proper_unyt[cut] * unyt.kb)
    ).to(unyt.K)
    writer.box_size = writer.radquantity(size_proper_unyt)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(np.zeros(n) * (unyt.cm / unyt.s))
    writer.fluid.temp_radarray = writer.radarray(temp_proper_unyt)
    writer.simulation.fluid.mu = mu_dimensionless
    return writer

def plot_snapshot(filename, config, **kwargs):
    sim = rio.loadhdf5(config, filename)
    grid_cells = int(sim.par.mesh.grid_cells)
    ghost_cells = int(sim.par.mesh.ghost_cells)
    units = config["_code_units"]
    boundary_values = np.asarray(
        sim.mesh.boundary_radarray.to(units.length_unit).value,
        dtype=float,
    )
    if boundary_values.size == grid_cells + 1:
        boundary_active = boundary_values
    elif boundary_values.size == grid_cells + 2 * ghost_cells + 1:
        boundary_active = boundary_values[ghost_cells:ghost_cells + grid_cells + 1]
    else:
        raise ValueError("unexpected Sedov Cartesian boundary RadArray shape")
    x = .5 * (boundary_active[:-1] + boundary_active[1:])
    box_size_proper_code = quantity_to_value(
        config["initial_condition"]["box_size_proper"],
        units.length_unit,
    )
    left_half = x <= 0.5 * box_size_proper_code
    def active_values(radarray, unit):
        values = np.asarray(radarray.to(unit).value, dtype=float)
        if values.size == grid_cells:
            return values
        if values.size == grid_cells + 2 * ghost_cells:
            return values[ghost_cells:ghost_cells + grid_cells]
        raise ValueError("unexpected Sedov Cartesian fluid RadArray shape")

    rho_proper_code = active_values(sim.fluid.rho_radarray, units.density_unit)
    temp_proper_code = active_values(sim.fluid.temp_radarray, units.temperature_unit)
    vel_proper_code = active_values(sim.fluid.vel_radarray, units.velocity_unit)
    mu_values = np.asarray(sim.fluid.mu, dtype=float)
    if mu_values.size != grid_cells:
        mu_values = mu_values[ghost_cells:ghost_cells + grid_cells]
    pre_proper_code = sim.fluid.eos.pressure(
        rho_proper_code, temp_proper_code, mu_values
    )
    color = kwargs.get("color")
    plt.subplot(1,3,1); plt.plot(x[left_half], np.asarray(pre_proper_code)[left_half], **kwargs)
    plt.subplot(1,3,2); plt.plot(x[left_half], vel_proper_code[left_half], **kwargs)
    plt.subplot(1,3,3); plt.plot(x[left_half], rho_proper_code[left_half], **kwargs)
    time_proper_unyt = float(np.asarray(sim.fluid.time_proper_code).flat[0]) * units.time_unit
    if time_proper_unyt > 0.0 * unyt.s:
        ic, par, units = config["initial_condition"], config["par"], config["_code_units"]
        explosion_energy_unyt = ic["explosion_energy"]
        area_proper_unyt = par["mesh"]["area_proper"]
        density_proper_unyt = ic["rho_proper"]
        analytic_radius_unyt, analytic_density_unyt, analytic_velocity_unyt, analytic_pressure_unyt, shock_radius_unyt = sa.get_blastwave_solution(
            explosion_energy_unyt, density_proper_unyt * area_proper_unyt, 1, par["hydrodynamics"]["gamma"], 0.0, time_proper_unyt
        )
        analytic_radius_unyt = unyt.uconcatenate((analytic_radius_unyt, unyt.unyt_array([shock_radius_unyt, 2.0 * shock_radius_unyt])))
        analytic_density_unyt = unyt.uconcatenate((analytic_density_unyt, np.zeros(2) * analytic_density_unyt.units + density_proper_unyt * area_proper_unyt))
        analytic_velocity_unyt = unyt.uconcatenate((analytic_velocity_unyt, np.zeros(2) * analytic_velocity_unyt.units))
        analytic_pressure_unyt = unyt.uconcatenate((analytic_pressure_unyt, np.zeros(2) * analytic_pressure_unyt.units))
        analytic_left_half = analytic_radius_unyt <= 0.5 * ic["box_size_proper"]
        plt.subplot(1,3,1); plt.plot(analytic_radius_unyt[analytic_left_half].in_cgs(), analytic_pressure_unyt[analytic_left_half].in_cgs(), color=color, linestyle="--")
        plt.subplot(1,3,2); plt.plot(analytic_radius_unyt[analytic_left_half].in_cgs(), analytic_velocity_unyt[analytic_left_half].in_cgs(), color=color, linestyle="--")
        plt.subplot(1,3,3); plt.plot(analytic_radius_unyt[analytic_left_half].in_cgs(), (analytic_density_unyt[analytic_left_half] / area_proper_unyt).in_cgs(), color=color, linestyle="--")
