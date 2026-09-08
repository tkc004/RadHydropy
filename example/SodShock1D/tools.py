"""Initial conditions and plotting for the Sod shock tube."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from radhydropy.rsim import Rsim
from radhydropy.units import quantity_to_value
from basic_hydro_utils import make_initial_condition
from sodshock_analytic import shocktubecal, shocktubeanalyticgraph

def build_initial_condition(config):
    ic, units = config["initial_condition"], config["_code_units"]; n=int(ic["grid_cells"]); size=quantity_to_value(ic["box_size"],units.length_unit); b=np.linspace(-size/n,size+size/n,n+1); x=.5*(b[:-1]+b[1:]); mid=(x>.25*size)&(x<.75*size)
    rho=np.full(n,quantity_to_value(ic["initial_density"],units.density_unit)); rho[mid]*=ic["density_ratio"]; temp=np.full(n,quantity_to_value(ic["initial_temperature"],units.temperature_unit)); temp[mid]*=ic["temperature_ratio"]
    return make_initial_condition(config, boundary_proper_code=b, rho_proper_code=rho, vel_proper_code=np.full(n,quantity_to_value(ic["initial_velocity"],units.velocity_unit)), temp_proper_code=temp, mu_dimensionless=np.full(n,ic["mean_molecular_weight"]), area_proper_code=np.ones(n)*quantity_to_value(config["par"]["mesh"]["area"],units.area_unit))

def getAnalyticSolution(config, state):
    ic = config["initial_condition"]
    units = config["_code_units"]
    rho_high = quantity_to_value(ic["initial_density"], units.density_unit)
    rho_low = rho_high * ic["density_ratio"]
    temp_high = quantity_to_value(ic["initial_temperature"], units.temperature_unit)
    temp_low = temp_high * ic["temperature_ratio"]
    mu = ic["mean_molecular_weight"]
    pressure_low = float(np.asarray(state.fluid.eos.pressure(rho_low, temp_low, mu)))
    pressure_high = float(np.asarray(state.fluid.eos.pressure(rho_high, temp_high, mu)))
    rho2, rho3, p2, v2, vt, vs, _ = shocktubecal(
        config["par"]["hydrodynamics"]["gamma"], rho_low, rho_high,
        pressure_low, pressure_high,
    )
    time = float(np.asarray(state.fluid.time_proper_code).flat[0])
    if time <= 0.0:
        return None
    boundary_proper_code = np.asarray(state.mesh.boundary_proper_code, dtype=float)
    centers = 0.5 * (boundary_proper_code[:-1] + boundary_proper_code[1:])
    box = quantity_to_value(ic["box_size"], units.length_unit)
    interface = 0.25 * box
    left, _, _ = shocktubeanalyticgraph(
        config["par"]["hydrodynamics"]["gamma"], rho_low, rho2, rho3,
        rho_high, pressure_low, p2, pressure_high, v2, vt, vs,
        time, centers, interface,
    )
    mirrored, _, _ = shocktubeanalyticgraph(
        config["par"]["hydrodynamics"]["gamma"], rho_low, rho2, rho3,
        rho_high, pressure_low, p2, pressure_high, v2, vt, vs,
        time, box - centers, interface,
    )
    return np.where(centers <= 0.5 * box, left, mirrored)

def ReadandPlot(filename, config, **kwargs):
    sim=Rsim(config["par"]); import radhydropy.io as rio; rio.readhdf5(sim.par,sim.mesh,sim.fluid,filename); first=int(sim.par.mesh.ghost_cells); last=first+int(sim.par.mesh.grid_cells); b=np.asarray(sim.mesh.boundary_proper_code); x=.5*(b[:-1]+b[1:]); plt.plot(x[first:last],np.asarray(sim.fluid.rho_proper_code)[first:last],**kwargs)
    analytic = getAnalyticSolution(config, sim)
    if analytic is not None:
        plt.plot(x[first:last], analytic[first:last], color=kwargs.get("color"), linestyle="--", label="analytic")
