"""Initial conditions and diagnostics for high-Mach advection."""
import numpy as np
from radhydropy.eos import EOS
from radhydropy.rsim import Rsim
from radhydropy.units import quantity_to_value
from basic_hydro_utils import make_initial_condition

def build_initial_condition(config):
    initial, par, units = config["initial_condition"], config["par"], config["_code_units"]
    n=int(initial["grid_cells"]); size=quantity_to_value(initial["box_size"],units.length_unit); boundary=np.linspace(0,size,n+1); x=.5*(boundary[:-1]+boundary[1:]); left=x < .5*size
    rho=np.where(left, initial.get("rho_left",initial.get("initial_density", 1.0)), initial.get("rho_right",initial.get("initial_density", 1.0)))
    vel=np.full(n,quantity_to_value(initial["initial_velocity"],units.velocity_unit)); mu=np.full(n,initial["mean_molecular_weight"])
    if "temp_left" in initial or "temp_right" in initial:
        temp=np.where(left,initial.get("temp_left",initial.get("initial_temperature",0)),initial.get("temp_right",initial.get("initial_temperature",0)))
    elif "pressure_initial" in initial:
        rho_left = quantity_to_value(initial["rho_left"], units.density_unit)
        pressure = quantity_to_value(initial["pressure_initial"], units.pressure_unit)
        temp=np.full(n, pressure / rho_left)
    else: temp=np.full(n,quantity_to_value(initial["initial_temperature"],units.temperature_unit))
    temp=np.asarray([quantity_to_value(v,units.temperature_unit) if hasattr(v,"to_value") else float(v) for v in temp])
    rho=np.asarray([quantity_to_value(v,units.density_unit) if hasattr(v,"to_value") else float(v) for v in rho])
    return make_initial_condition(config,boundary,rho,vel,temp,mu,area=np.ones(n)*quantity_to_value(par["mesh"]["area"],units.area_unit))

def _physical(state):
    first=int(state.par.mesh.ghost_cells); last=first+int(state.par.mesh.grid_cells); b=np.asarray(state.mesh.boundary_proper_code); return first,last,.5*(b[:-1]+b[1:])

def energy_components(state):
    first,last,_=_physical(state); rho=np.asarray(state.fluid.rho_proper_code); vel=np.asarray(state.fluid.vel_proper_code); temp=np.asarray(state.fluid.temp_proper_code); mu=np.asarray(state.fluid.mu); eos=state.fluid.eos or EOS("polytropic",float(state.par.hydrodynamics.gamma),state.par.units.CodeUnits); pressure=np.asarray(eos.pressure(rho,temp,mu)); vol=np.asarray(state.mesh.volume_proper_code); kinetic=.5*rho*vel**2*vol; thermal=pressure/(eos.gamma-1)*vol; return {"total":float(np.sum((kinetic+thermal)[first:last])),"kinetic":float(np.sum(kinetic[first:last])),"thermal":float(np.sum(thermal[first:last]))}

def entropy_profile(state):
    first,last,x=_physical(state); rho=np.asarray(state.fluid.rho_proper_code); temp=np.asarray(state.fluid.temp_proper_code); gamma=float(state.par.hydrodynamics.gamma); return x[first:last],temp[first:last]/rho[first:last]**(gamma-1)

def primitive_profiles(state):
    first,last,x=_physical(state); return x[first:last],np.asarray(state.fluid.rho_proper_code)[first:last],np.asarray(state.fluid.temp_proper_code)[first:last]
