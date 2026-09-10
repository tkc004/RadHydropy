"""Initial conditions and diagnostics for high-Mach advection."""
import numpy as np
from radhydropy.eos import EOS
from radhydropy.rsim import Rsim
from radhydropy.units import quantity_to_value
from basic_hydro_utils import make_initial_condition

def build_initial_condition(config):
    initial, par, units = config["initial_condition"], config["par"], config["_code_units"]
    n=int(initial["grid_cells"]); size_proper_code=quantity_to_value(initial["box_size_proper"],units.length_unit); boundary_proper_code=np.linspace(0,size_proper_code,n+1); x_proper_code=.5*(boundary_proper_code[:-1]+boundary_proper_code[1:]); left=x_proper_code < .5*size_proper_code
    rho_proper_code=np.where(left, initial.get("rho_left_proper",initial.get("rho_proper", 1.0)), initial.get("rho_right_proper",initial.get("rho_proper", 1.0)))
    vel_proper_code=np.full(n,quantity_to_value(initial["vel_proper"],units.velocity_unit)); mu=np.full(n,initial["mean_molecular_weight"])
    if "temperature_left_proper" in initial or "temperature_right_proper" in initial:
        zero_temperature_proper_unyt = 0.0 * units.temperature_unit
        temp_proper_code=np.where(left,initial.get("temperature_left_proper",initial.get("temperature_proper",zero_temperature_proper_unyt)),initial.get("temperature_right_proper",initial.get("temperature_proper",zero_temperature_proper_unyt)))
    elif "pressure_initial_proper" in initial:
        rho_left = quantity_to_value(initial["rho_left_proper"], units.density_unit)
        pressure_proper_code = quantity_to_value(initial["pressure_initial_proper"], units.pressure_unit)
        temp_proper_code=np.full(n, pressure_proper_code / rho_left)
    else: temp_proper_code=np.full(n,quantity_to_value(initial["temperature_proper"],units.temperature_unit))
    temp_proper_code=np.asarray([quantity_to_value(v,units.temperature_unit) if hasattr(v,"to_value") else float(v) for v in temp_proper_code])
    rho_proper_code=np.asarray([quantity_to_value(v,units.density_unit) if hasattr(v,"to_value") else float(v) for v in rho_proper_code])
    return make_initial_condition(config, boundary_proper_code=boundary_proper_code, rho_proper_code=rho_proper_code, vel_proper_code=vel_proper_code, temp_proper_code=temp_proper_code, mu_dimensionless=mu, area_proper_code=np.ones(n)*quantity_to_value(config["par"]["mesh"]["area_proper"],units.area_unit))

def _physical(state):
    first=int(state.par.mesh.ghost_cells); last=first+int(state.par.mesh.grid_cells); b=np.asarray(state.mesh.boundary_proper_code); return first,last,.5*(b[:-1]+b[1:])

def energy_components(state):
    first,last,_=_physical(state); rho_proper_code=np.asarray(state.fluid.rho_proper_code); vel_proper_code=np.asarray(state.fluid.vel_proper_code); temp_proper_code=np.asarray(state.fluid.temp_proper_code); mu=np.asarray(state.fluid.mu); eos=state.fluid.eos or EOS("polytropic",float(state.par.hydrodynamics.gamma),state.par.units.CodeUnits); pressure_proper_code=np.asarray(eos.pressure(rho_proper_code,temp_proper_code,mu)); volume_proper_code=np.asarray(state.mesh.volume_proper_code); kinetic_energy_proper_code=.5*rho_proper_code*vel_proper_code**2*volume_proper_code; thermal_energy_proper_code=pressure_proper_code/(eos.gamma-1)*volume_proper_code; return {"total_energy_proper_code":float(np.sum((kinetic_energy_proper_code+thermal_energy_proper_code)[first:last])),"kinetic_energy_proper_code":float(np.sum(kinetic_energy_proper_code[first:last])),"thermal_energy_proper_code":float(np.sum(thermal_energy_proper_code[first:last]))}

def entropy_profile(state):
    first,last,x_proper_code=_physical(state); rho_proper_code=np.asarray(state.fluid.rho_proper_code); temp_proper_code=np.asarray(state.fluid.temp_proper_code); gamma=float(state.par.hydrodynamics.gamma); return x_proper_code[first:last],temp_proper_code[first:last]/rho_proper_code[first:last]**(gamma-1)

def primitive_profiles(state):
    first,last,x_proper_code=_physical(state); return x_proper_code[first:last],np.asarray(state.fluid.rho_proper_code)[first:last],np.asarray(state.fluid.temp_proper_code)[first:last]
