"""Initial conditions and plotting for spherical Sedov-Taylor."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from radhydropy.rsim import Rsim
from radhydropy.units import quantity_to_value
from basic_hydro_utils import make_initial_condition
from SedovTaylor_analytic import get_blastwave_solution

def set_plot_style():
    plt.rcParams.update({"figure.figsize": (18, 6), "font.size": 14})

def build_initial_condition(config):
    ic, par, units = config["initial_condition"], config["par"], config["_code_units"]
    n=int(ic["grid_cells"]); start=quantity_to_value(ic["radius_injection_proper"], units.length_unit); size=quantity_to_value(ic["box_size_proper"], units.length_unit)
    boundary_proper_code=np.linspace(start,start+size,n+1); coordinate_proper_code=.5*(boundary_proper_code[:-1]+boundary_proper_code[1:]); rho_proper_code=np.full(n,quantity_to_value(ic["rho_proper"],units.density_unit)); mu_dimensionless=np.full(n,ic["mean_molecular_weight"]); temp_proper_code=np.zeros(n)
    volume_proper_code=4*np.pi/3*np.diff(boundary_proper_code**3); cut=(coordinate_proper_code < quantity_to_value(ic["radius_explosion_proper"],units.length_unit)) & (coordinate_proper_code >= start); energy_proper_code=quantity_to_value(ic["explosion_energy"],units.energy_unit); pre_proper_code=(par["hydrodynamics"]["gamma"]-1)*energy_proper_code/np.sum(volume_proper_code[cut]); temp_proper_code[cut]=np.asarray(Rsim(config["par"]).fluid.eos.temperature(rho_proper_code[cut],np.full(np.count_nonzero(cut),pre_proper_code),mu_dimensionless[cut]),dtype=float)
    return make_initial_condition(config, boundary_proper_code=boundary_proper_code, rho_proper_code=rho_proper_code, vel_proper_code=np.zeros(n), temp_proper_code=temp_proper_code, mu_dimensionless=mu_dimensionless, area_proper_code=4*np.pi*coordinate_proper_code**2)

def plot_snapshot(filename, config, **kwargs):
    sim = Rsim(config["par"])
    import radhydropy.io as rio

    rio.readhdf5(sim.par, sim.mesh, sim.fluid, filename)
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    boundary_proper_code = np.asarray(sim.mesh.boundary_proper_code)
    coordinate_proper_code = .5 * (
        boundary_proper_code[:-1] + boundary_proper_code[1:]
    )[first:last]
    pressure_proper_code = sim.fluid.eos.pressure(
        sim.fluid.rho_proper_code,
        sim.fluid.temp_proper_code,
        sim.fluid.mu,
    )

    units = config["_code_units"]
    length_cgs_per_code = float(units.length_unit.to_value("cm"))
    density_cgs_per_code = float(units.density_unit.to_value("g/cm**3"))
    velocity_cgs_per_code = float(units.velocity_unit.to_value("cm/s"))
    pressure_cgs_per_code = float(units.pressure_unit.to_value("erg/cm**3"))
    time_cgs_per_code = float(units.time_unit.to_value("s"))
    coordinate_proper_cgs_cm = coordinate_proper_code * length_cgs_per_code
    pressure_proper_cgs_erg_cm3 = (
        np.asarray(pressure_proper_code)[first:last] * pressure_cgs_per_code
    )
    velocity_proper_cgs_cm_s = (
        np.asarray(sim.fluid.vel_proper_code)[first:last] * velocity_cgs_per_code
    )
    rho_proper_cgs_g_cm3 = (
        np.asarray(sim.fluid.rho_proper_code)[first:last] * density_cgs_per_code
    )

    time_proper_cgs_s = (
        float(np.asarray(sim.fluid.time_proper_code).reshape(-1)[0])
        * time_cgs_per_code
    )
    ic = config["initial_condition"]
    analytic_radius_cgs_cm, analytic_rho_cgs_g_cm3, analytic_velocity_cgs_cm_s, analytic_pressure_cgs_erg_cm3, _ = get_blastwave_solution(
        float(ic["explosion_energy"].to_value("erg")),
        float(ic["rho_proper"].to_value("g/cm**3")),
        3,
        float(config["par"]["hydrodynamics"]["gamma"]),
        0,
        time_proper_cgs_s,
    )
    analytic_valid = (
        np.isfinite(analytic_radius_cgs_cm)
        & np.isfinite(analytic_rho_cgs_g_cm3)
        & np.isfinite(analytic_velocity_cgs_cm_s)
        & np.isfinite(analytic_pressure_cgs_erg_cm3)
    )

    pressure_axis = plt.subplot(1, 3, 1)
    pressure_axis.plot(coordinate_proper_cgs_cm, pressure_proper_cgs_erg_cm3, **kwargs)
    pressure_axis.plot(
        analytic_radius_cgs_cm[analytic_valid],
        analytic_pressure_cgs_erg_cm3[analytic_valid],
        "k--",
        linewidth=1.5,
    )
    velocity_axis = plt.subplot(1, 3, 2)
    velocity_axis.plot(coordinate_proper_cgs_cm, velocity_proper_cgs_cm_s, **kwargs)
    velocity_axis.plot(
        analytic_radius_cgs_cm[analytic_valid],
        analytic_velocity_cgs_cm_s[analytic_valid],
        "k--",
        linewidth=1.5,
    )
    density_axis = plt.subplot(1, 3, 3)
    density_axis.plot(coordinate_proper_cgs_cm, rho_proper_cgs_g_cm3, **kwargs)
    density_axis.plot(
        analytic_radius_cgs_cm[analytic_valid],
        analytic_rho_cgs_g_cm3[analytic_valid],
        "k--",
        linewidth=1.5,
    )
