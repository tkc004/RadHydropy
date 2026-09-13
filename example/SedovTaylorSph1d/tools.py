"""Initial conditions and plotting for spherical Sedov-Taylor."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt
import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from example.SedovTaylorSph1d.SedovTaylor_analytic import get_blastwave_solution

def set_plot_style():
    plt.rcParams.update({"figure.figsize": (18, 6), "font.size": 14})

def build_initial_condition(config):
    ic, par, units = config["initial_condition"], config["par"], config["_code_units"]
    n = int(ic["grid_cells"])
    start_proper_unyt = ic["radius_injection_proper"]
    size_proper_unyt = ic["box_size_proper"]
    boundary_proper_unyt = start_proper_unyt + np.linspace(0.0, 1.0, n + 1) * size_proper_unyt
    coordinate_proper_unyt = 0.5 * (boundary_proper_unyt[:-1] + boundary_proper_unyt[1:])
    rho_proper_unyt = np.ones(n) * ic["rho_proper"]
    mu_dimensionless = np.full(n, float(ic["mean_molecular_weight"]))
    temp_proper_unyt = np.zeros(n) * unyt.K
    volume_proper_unyt = 4 * np.pi / 3 * np.diff(boundary_proper_unyt**3)
    cut = (coordinate_proper_unyt < ic["radius_explosion_proper"]) & (coordinate_proper_unyt >= boundary_proper_unyt[0])
    # Spread the deposited explosion energy over the injection volume and
    # obtain the pressure that corresponds to that energy.
    pressure_proper_unyt = (
        (float(par["hydrodynamics"]["gamma"]) - 1)
        * ic["explosion_energy"]
        / np.sum(volume_proper_unyt[cut])
    )
    writer = InitialConditionWriter(par_config=par, code_units=units, ic_config=config["initial_condition"])
    # Invert the configured EOS so the temperature written into the IC gives
    # the same pressure, density, and mean molecular weight.
    temp_proper_unyt[cut] = (
        pressure_proper_unyt
        * mu_dimensionless[cut]
        * unyt.mp
        / (rho_proper_unyt[cut] * unyt.kb)
    ).to(unyt.K)
    writer.box_size = writer.radquantity(start_proper_unyt + size_proper_unyt)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(np.zeros(n) * (unyt.cm / unyt.s))
    writer.fluid.temp_radarray = writer.radarray(temp_proper_unyt)
    writer.simulation.fluid.mu = mu_dimensionless
    return writer

def plot_snapshot(filename, config, **kwargs):
    sim = rio.loadhdf5(config, filename)
    units = config["_code_units"]
    first = int(sim.par.mesh.ghost_cells)
    count = int(sim.par.mesh.grid_cells)
    boundary_values = np.asarray(
        sim.mesh.boundary_radarray.to(units.length_unit).value,
        dtype=float,
    )
    if boundary_values.size == count + 1:
        boundary_proper_code = boundary_values
    elif boundary_values.size == count + 2 * first + 1:
        boundary_proper_code = boundary_values[first:first + count + 1]
    else:
        raise ValueError("unexpected Sedov spherical boundary RadArray shape")
    coordinate_proper_code = .5 * (
        boundary_proper_code[:-1] + boundary_proper_code[1:]
    )

    def active_values(radarray, unit):
        values = np.asarray(radarray.to(unit).value, dtype=float)
        if values.size == count:
            return values
        if values.size == count + 2 * first:
            return values[first:first + count]
        raise ValueError("unexpected Sedov spherical fluid RadArray shape")

    rho_proper_code = active_values(sim.fluid.rho_radarray, units.density_unit)
    temp_proper_code = active_values(sim.fluid.temp_radarray, units.temperature_unit)
    mu_values = np.asarray(sim.fluid.mu, dtype=float)
    if mu_values.size != count:
        mu_values = mu_values[first:first + count]
    pressure_proper_code = sim.fluid.eos.pressure(
        rho_proper_code,
        temp_proper_code,
        mu_values,
    )

    length_cgs_per_code = float(units.length_unit.to_value("cm"))
    density_cgs_per_code = float(units.density_unit.to_value("g/cm**3"))
    velocity_cgs_per_code = float(units.velocity_unit.to_value("cm/s"))
    pressure_cgs_per_code = float(units.pressure_unit.to_value("erg/cm**3"))
    time_cgs_per_code = float(units.time_unit.to_value("s"))
    coordinate_proper_cgs_cm = coordinate_proper_code * length_cgs_per_code
    pressure_proper_cgs_erg_cm3 = (
        np.asarray(pressure_proper_code) * pressure_cgs_per_code
    )
    velocity_proper_cgs_cm_s = (
        active_values(sim.fluid.vel_radarray, units.velocity_unit) * velocity_cgs_per_code
    )
    rho_proper_cgs_g_cm3 = (
        rho_proper_code * density_cgs_per_code
    )

    time_proper_cgs_s = (
        float(np.asarray(sim.fluid.time_proper_code).reshape(-1)[0])
        * time_cgs_per_code
    )
    analytic_valid = None
    if time_proper_cgs_s > 0.0:
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
    if analytic_valid is not None:
        pressure_axis.plot(
            analytic_radius_cgs_cm[analytic_valid],
            analytic_pressure_cgs_erg_cm3[analytic_valid],
            "k--",
            linewidth=1.5,
        )
    velocity_axis = plt.subplot(1, 3, 2)
    velocity_axis.plot(coordinate_proper_cgs_cm, velocity_proper_cgs_cm_s, **kwargs)
    if analytic_valid is not None:
        velocity_axis.plot(
            analytic_radius_cgs_cm[analytic_valid],
            analytic_velocity_cgs_cm_s[analytic_valid],
            "k--",
            linewidth=1.5,
        )
    density_axis = plt.subplot(1, 3, 3)
    density_axis.plot(coordinate_proper_cgs_cm, rho_proper_cgs_g_cm3, **kwargs)
    if analytic_valid is not None:
        density_axis.plot(
            analytic_radius_cgs_cm[analytic_valid],
            analytic_rho_cgs_g_cm3[analytic_valid],
            "k--",
            linewidth=1.5,
        )
