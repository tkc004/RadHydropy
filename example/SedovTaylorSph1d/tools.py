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
    n=int(ic["grid_cells"]); start=quantity_to_value(ic["injection_radius"], units.length_unit); size=quantity_to_value(ic["box_size"], units.length_unit)
    boundary=np.linspace(start,start+size,n+1); center=.5*(boundary[:-1]+boundary[1:]); rho=np.full(n,quantity_to_value(ic["initial_density"],units.density_unit)); mu=np.full(n,ic["mean_molecular_weight"]); temp=np.zeros(n)
    volume=4*np.pi/3*np.diff(boundary**3); cut=(center < quantity_to_value(ic["explosion_radius"],units.length_unit)) & (center >= start); energy=quantity_to_value(ic["explosion_energy"],units.energy_unit); pressure=(par["hydrodynamics"]["gamma"]-1)*energy/np.sum(volume[cut]); temp[cut]=np.asarray(Rsim(par).fluid.eos.temperature(rho[cut],np.full(np.count_nonzero(cut),pressure),mu[cut]),dtype=float)
    return make_initial_condition(config,boundary,rho,np.zeros(n),temp,mu,area=4*np.pi*boundary[:-1]**2)

def ReadandPlot(filename, config, **kwargs):
    sim=Rsim(config["par"]); import radhydropy.io as rio; rio.readhdf5(sim.par,sim.mesh,sim.fluid,filename); first=int(sim.par.mesh.ghost_cells); last=first+int(sim.par.mesh.grid_cells); b=np.asarray(sim.mesh.boundary_proper_code); x=.5*(b[:-1]+b[1:])[first:last]
    pressure=sim.fluid.eos.pressure(sim.fluid.rho_proper_code,sim.fluid.temp_proper_code,sim.fluid.mu)
    plt.subplot(1,3,1); plt.plot(x,np.asarray(pressure)[first:last],**kwargs); plt.subplot(1,3,2); plt.plot(x,np.asarray(sim.fluid.vel_proper_code)[first:last],**kwargs); plt.subplot(1,3,3); plt.plot(x,np.asarray(sim.fluid.rho_proper_code)[first:last],**kwargs)
