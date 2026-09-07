"""Initial conditions and plotting for the cartesian Sedov benchmark."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt
from radhydropy.rsim import Rsim
from radhydropy.units import quantity_to_value
from basic_hydro_utils import make_initial_condition
import SedovTaylor_analytic as sa

def set_plot_style():
    plt.rcParams.update({"figure.figsize": (18, 6), "font.size": 14})

def build_initial_condition(config):
    ic, par, units = config["initial_condition"], config["par"], config["_code_units"]
    n = int(ic["grid_cells"]); size = quantity_to_value(ic["box_size"], units.length_unit)
    boundary = np.linspace(-.5*size/n, size+.5*size/n, n+1)
    rho = np.full(n, quantity_to_value(ic["initial_density"], units.density_unit)); mu = np.full(n, ic["mean_molecular_weight"])
    temp = np.zeros(n); volume = np.full(n, quantity_to_value(par["mesh"]["area"], units.area_unit) * np.diff(boundary)); cut = 1
    energy = quantity_to_value(ic["explosion_energy"], units.energy_unit)
    pressure = (par["hydrodynamics"]["gamma"]-1) * energy / volume[cut]
    probe = Rsim(config["par"]).fluid.eos.temperature(rho[cut], pressure, mu[cut])
    temp[cut] = float(np.asarray(probe))
    return make_initial_condition(config, boundary_proper_code=boundary, rho_proper_code=rho, vel_proper_code=np.zeros(n), temp_proper_code=temp, mu_dimensionless=mu, area_proper_code=np.full(n, quantity_to_value(par["mesh"]["area"], units.area_unit)))

def ReadandPlot(filename, config, **kwargs):
    sim = Rsim(config["par"]); import radhydropy.io as rio; rio.readhdf5(sim.par, sim.mesh, sim.fluid, filename)
    first=int(sim.par.mesh.ghost_cells); last=first+int(sim.par.mesh.grid_cells); b=np.asarray(sim.mesh.boundary_proper_code); x=.5*(b[:-1]+b[1:])[first:last]
    pressure = sim.fluid.eos.pressure(sim.fluid.rho_proper_code, sim.fluid.temp_proper_code, sim.fluid.mu)
    color = kwargs.get("color")
    plt.subplot(1,3,1); plt.plot(x, np.asarray(pressure)[first:last], **kwargs)
    plt.subplot(1,3,2); plt.plot(x, np.asarray(sim.fluid.vel_proper_code)[first:last], **kwargs)
    plt.subplot(1,3,3); plt.plot(x, np.asarray(sim.fluid.rho_proper_code)[first:last], **kwargs)
    time = float(np.asarray(sim.fluid.time_proper_code).flat[0]) * config["_code_units"].time_unit
    if time > 0.0 * unyt.s:
        ic, par, units = config["initial_condition"], config["par"], config["_code_units"]
        energy = ic["explosion_energy"]
        area = par["mesh"]["area"]
        density = ic["initial_density"]
        r, rho, velocity, analytic_pressure, shock_radius = sa.get_blastwave_solution(
            energy, density * area, 1, par["hydrodynamics"]["gamma"], 0.0, time
        )
        analytic_r = unyt.uconcatenate((r, unyt.unyt_array([shock_radius, 2.0 * shock_radius])))
        analytic_rho = unyt.uconcatenate((rho, np.zeros(2) * rho.units + density * area))
        analytic_velocity = unyt.uconcatenate((velocity, np.zeros(2) * velocity.units))
        analytic_pressure = unyt.uconcatenate((analytic_pressure, np.zeros(2) * analytic_pressure.units))
        plt.subplot(1,3,1); plt.plot(analytic_r.in_cgs(), analytic_pressure.in_cgs(), color=color, linestyle="--")
        plt.subplot(1,3,2); plt.plot(analytic_r.in_cgs(), analytic_velocity.in_cgs(), color=color, linestyle="--")
        plt.subplot(1,3,3); plt.plot(analytic_r.in_cgs(), (analytic_rho / area).in_cgs(), color=color, linestyle="--")
