"""Initial conditions and plotting for the cartesian Sedov benchmark."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt
import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import quantity_to_value
import SedovTaylor_analytic as sa

def set_plot_style():
    plt.rcParams.update({"figure.figsize": (18, 6), "font.size": 14})

def build_initial_condition(config):
    ic, par, units = config["initial_condition"], config["par"], config["_code_units"]
    n = int(ic["grid_cells"])
    size_proper_unyt = ic["box_size_proper"]
    boundary_proper_unyt = np.linspace(0.0, 1.0, n + 1) * size_proper_unyt
    boundary_proper_code = boundary_proper_unyt.to_value(units.length_unit)
    rho_proper_code = np.full(n, quantity_to_value(ic["rho_proper"], units.density_unit))
    mu_dimensionless = np.full(n, float(ic["mean_molecular_weight"]))
    temp_proper_code = np.zeros(n)
    volume_proper_code = quantity_to_value(par["mesh"]["area_proper"], units.area_unit) * np.diff(boundary_proper_code)
    cut = 1
    energy_proper_code = quantity_to_value(ic["explosion_energy"], units.energy_unit)
    pre_proper_code = (par["hydrodynamics"]["gamma"]-1) * energy_proper_code / volume_proper_code[cut]
    from radhydropy.rsim import Rsim
    probe = Rsim(config["par"]).fluid.eos.temperature(rho_proper_code[cut], pre_proper_code, mu_dimensionless[cut])
    temp_proper_code[cut] = float(np.asarray(probe))
    writer = InitialConditionWriter(par_config=par, code_units=units)
    writer.box_size = writer.radquantity(size_proper_unyt)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(unyt.unyt_array(rho_proper_code, units.density_unit))
    writer.fluid.vel_radarray = writer.radarray(unyt.unyt_array(np.zeros(n), units.velocity_unit))
    writer.fluid.temp_radarray = writer.radarray(unyt.unyt_array(temp_proper_code, units.temperature_unit))
    writer.simulation.fluid.mu = mu_dimensionless
    return writer

def plot_snapshot(filename, config, **kwargs):
    sim = rio.loadhdf5(config, filename)
    first=int(sim.par.mesh.ghost_cells); last=first+int(sim.par.mesh.grid_cells)
    b=np.asarray(sim.mesh.boundary_radarray.to_value(config["_code_units"].length_unit))
    x=.5*(b[:-1]+b[1:])[first:last]
    box_size_proper_code = quantity_to_value(
        config["initial_condition"]["box_size_proper"],
        config["_code_units"].length_unit,
    )
    left_half = x <= 0.5 * box_size_proper_code
    rho_proper_code = sim.fluid.rho_radarray.to_value(config["_code_units"].density_unit)
    temp_proper_code = sim.fluid.temp_radarray.to_value(config["_code_units"].temperature_unit)
    vel_proper_code = sim.fluid.vel_radarray.to_value(config["_code_units"].velocity_unit)
    pre_proper_code = sim.fluid.eos.pressure(rho_proper_code, temp_proper_code, sim.fluid.mu)
    color = kwargs.get("color")
    plt.subplot(1,3,1); plt.plot(x[left_half], np.asarray(pre_proper_code)[first:last][left_half], **kwargs)
    plt.subplot(1,3,2); plt.plot(x[left_half], vel_proper_code[first:last][left_half], **kwargs)
    plt.subplot(1,3,3); plt.plot(x[left_half], rho_proper_code[first:last][left_half], **kwargs)
    time_proper_unyt = float(np.asarray(sim.fluid.time_proper_code).flat[0]) * config["_code_units"].time_unit
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
