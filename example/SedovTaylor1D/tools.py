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
    n = int(ic["grid_cells"]); size = quantity_to_value(ic["box_size_proper"], units.length_unit)
    boundary_proper_code = np.linspace(-.5*size/n, size+.5*size/n, n+1)
    rho_proper_code = np.full(n, quantity_to_value(ic["initial_density"], units.density_unit)); mu_dimensionless = np.full(n, ic["mean_molecular_weight"])
    temp_proper_code = np.zeros(n); volume_proper_code = np.full(n, quantity_to_value(par["mesh"]["area_proper"], units.area_unit) * np.diff(boundary_proper_code)); cut = 1
    energy_proper_code = quantity_to_value(ic["explosion_energy"], units.energy_unit)
    pre_proper_code = (par["hydrodynamics"]["gamma"]-1) * energy_proper_code / volume_proper_code[cut]
    probe = Rsim(config["par"]).fluid.eos.temperature(rho_proper_code[cut], pre_proper_code, mu_dimensionless[cut])
    temp_proper_code[cut] = float(np.asarray(probe))
    return make_initial_condition(config, boundary_proper_code=boundary_proper_code, rho_proper_code=rho_proper_code, vel_proper_code=np.zeros(n), temp_proper_code=temp_proper_code, mu_dimensionless=mu_dimensionless, area_proper_code=np.full(n, quantity_to_value(par["mesh"]["area_proper"], units.area_unit)))

def ReadandPlot(filename, config, **kwargs):
    sim = Rsim(config["par"]); import radhydropy.io as rio; rio.readhdf5(sim.par, sim.mesh, sim.fluid, filename)
    first=int(sim.par.mesh.ghost_cells); last=first+int(sim.par.mesh.grid_cells); b=np.asarray(sim.mesh.boundary_proper_code); x=.5*(b[:-1]+b[1:])[first:last]
    pre_proper_code = sim.fluid.eos.pressure(sim.fluid.rho_proper_code, sim.fluid.temp_proper_code, sim.fluid.mu)
    color = kwargs.get("color")
    plt.subplot(1,3,1); plt.plot(x, np.asarray(pre_proper_code)[first:last], **kwargs)
    plt.subplot(1,3,2); plt.plot(x, np.asarray(sim.fluid.vel_proper_code)[first:last], **kwargs)
    plt.subplot(1,3,3); plt.plot(x, np.asarray(sim.fluid.rho_proper_code)[first:last], **kwargs)
    time_proper_unyt = float(np.asarray(sim.fluid.time_proper_code).flat[0]) * config["_code_units"].time_unit
    if time_proper_unyt > 0.0 * unyt.s:
        ic, par, units = config["initial_condition"], config["par"], config["_code_units"]
        explosion_energy_unyt = ic["explosion_energy"]
        area_proper_unyt = par["mesh"]["area_proper"]
        density_proper_unyt = ic["initial_density"]
        analytic_radius_unyt, analytic_density_unyt, analytic_velocity_unyt, analytic_pressure_unyt, shock_radius_unyt = sa.get_blastwave_solution(
            explosion_energy_unyt, density_proper_unyt * area_proper_unyt, 1, par["hydrodynamics"]["gamma"], 0.0, time_proper_unyt
        )
        analytic_radius_unyt = unyt.uconcatenate((analytic_radius_unyt, unyt.unyt_array([shock_radius_unyt, 2.0 * shock_radius_unyt])))
        analytic_density_unyt = unyt.uconcatenate((analytic_density_unyt, np.zeros(2) * analytic_density_unyt.units + density_proper_unyt * area_proper_unyt))
        analytic_velocity_unyt = unyt.uconcatenate((analytic_velocity_unyt, np.zeros(2) * analytic_velocity_unyt.units))
        analytic_pressure_unyt = unyt.uconcatenate((analytic_pressure_unyt, np.zeros(2) * analytic_pressure_unyt.units))
        plt.subplot(1,3,1); plt.plot(analytic_radius_unyt.in_cgs(), analytic_pressure_unyt.in_cgs(), color=color, linestyle="--")
        plt.subplot(1,3,2); plt.plot(analytic_radius_unyt.in_cgs(), analytic_velocity_unyt.in_cgs(), color=color, linestyle="--")
        plt.subplot(1,3,3); plt.plot(analytic_radius_unyt.in_cgs(), (analytic_density_unyt / area_proper_unyt).in_cgs(), color=color, linestyle="--")
