"""Initial conditions and plotting for spherical Sedov-Taylor."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from radhydropy.rsim import Rsim
from radhydropy.units import quantity_to_value
from basic_hydro_utils import make_initial_condition

def set_plot_style():
    plt.rcParams.update({"figure.figsize": (18, 6), "font.size": 14})

def build_initial_condition(config):
    ic, par, units = config["initial_condition"], config["par"], config["_code_units"]
    n=int(ic["grid_cells"]); start=quantity_to_value(ic["injection_radius"], units.length_unit); size=quantity_to_value(ic["box_size_proper"], units.length_unit)
    boundary_proper_code=np.linspace(start,start+size,n+1); coordinate_proper_code=.5*(boundary_proper_code[:-1]+boundary_proper_code[1:]); rho_proper_code=np.full(n,quantity_to_value(ic["rho_proper"],units.density_unit)); mu_dimensionless=np.full(n,ic["mean_molecular_weight"]); temp_proper_code=np.zeros(n)
    volume_proper_code=4*np.pi/3*np.diff(boundary_proper_code**3); cut=(coordinate_proper_code < quantity_to_value(ic["explosion_radius"],units.length_unit)) & (coordinate_proper_code >= start); energy_proper_code=quantity_to_value(ic["explosion_energy"],units.energy_unit); pre_proper_code=(par["hydrodynamics"]["gamma"]-1)*energy_proper_code/np.sum(volume_proper_code[cut]); temp_proper_code[cut]=np.asarray(Rsim(config["par"]).fluid.eos.temperature(rho_proper_code[cut],np.full(np.count_nonzero(cut),pre_proper_code),mu_dimensionless[cut]),dtype=float)
    return make_initial_condition(config, boundary_proper_code=boundary_proper_code, rho_proper_code=rho_proper_code, vel_proper_code=np.zeros(n), temp_proper_code=temp_proper_code, mu_dimensionless=mu_dimensionless, area_proper_code=4*np.pi*boundary_proper_code[:-1]**2)

def ReadandPlot(filename, config, **kwargs):
    sim=Rsim(config["par"]); import radhydropy.io as rio; rio.readhdf5(sim.par,sim.mesh,sim.fluid,filename); first=int(sim.par.mesh.ghost_cells); last=first+int(sim.par.mesh.grid_cells); boundary_proper_code=np.asarray(sim.mesh.boundary_proper_code); coordinate_proper_code=.5*(boundary_proper_code[:-1]+boundary_proper_code[1:])[first:last]
    pre_proper_code=sim.fluid.eos.pressure(sim.fluid.rho_proper_code,sim.fluid.temp_proper_code,sim.fluid.mu)
    plt.subplot(1,3,1); plt.plot(coordinate_proper_code,np.asarray(pre_proper_code)[first:last],**kwargs); plt.subplot(1,3,2); plt.plot(coordinate_proper_code,np.asarray(sim.fluid.vel_proper_code)[first:last],**kwargs); plt.subplot(1,3,3); plt.plot(coordinate_proper_code,np.asarray(sim.fluid.rho_proper_code)[first:last],**kwargs)
