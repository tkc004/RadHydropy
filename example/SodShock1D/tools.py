"""Initial conditions and plotting for the Sod shock tube."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import unyt
from radhydropy.initial_condition_writer import InitialConditionWriter
import radhydropy.io as rio


def shocktubecal(gamma, rho1, rho5, p1, p5):
    """Return the analytic Sod shock-tube intermediate states."""
    # p5 and rho5 are higher than p1 and rho1; see Pfrommer et al. 2006.
    mu2 = (gamma - 1.0) / (gamma + 1.0)
    c1 = np.sqrt(gamma * p1 / rho1)
    c5 = np.sqrt(gamma * p5 / rho5)

    def pressure_residual(p2):
        return (
            (p2 / p1 - 1.0)
            * np.sqrt((1.0 - mu2) / gamma / (p2 / p1 + mu2))
            - 2.0 / (gamma - 1.0)
            * c5 / c1
            * (1.0 - np.power(p2 / p5, (gamma - 1.0) / (2.0 * gamma)))
        )

    p2 = fsolve(pressure_residual, 0.4)[0]
    rho3 = rho5 * np.power(p2 / p5, 1.0 / gamma)
    rho2 = rho1 * (p2 + mu2 * p1) / (p1 + mu2 * p2)
    v2 = 2.0 * c5 / (gamma - 1.0) * (
        1.0 - np.power(p2 / p5, (gamma - 1.0) / (2.0 * gamma))
    )
    vt = c5 - v2 / (1.0 - mu2)
    vs = v2 / (1.0 - rho1 / rho2)
    return rho2, rho3, p2, v2, vt, vs, vs / c1


def shocktubeanalyticgraph(
    gamma, rho1, rho2, rho3, rho5, p1, p2, p5, v2, vt, vs,
    time_proper_code, xcor, xint,
):
    """Evaluate the analytic Sod solution at the supplied coordinates."""
    mu2 = (gamma - 1.0) / (gamma + 1.0)
    c1 = np.sqrt(gamma * p1 / rho1)
    c5 = np.sqrt(gamma * p5 / rho5)
    xnor = np.asarray(xcor) - xint
    rho_ana = np.zeros(len(xnor))
    p_ana = np.zeros(len(xnor))
    v_ana = np.zeros(len(xnor))
    logical5 = xnor < -c5 * time_proper_code
    logical4 = np.logical_and(
        xnor > -c5 * time_proper_code,
        xnor < -vt * time_proper_code,
    )
    logical3 = np.logical_and(
        xnor > -vt * time_proper_code,
        xnor < v2 * time_proper_code,
    )
    logical2 = np.logical_and(
        xnor > v2 * time_proper_code,
        xnor < vs * time_proper_code,
    )
    logical1 = xnor > vs * time_proper_code
    xnor4 = xnor[logical4]
    rho_ana[logical5] = rho5
    rho_ana[logical4] = rho5 * np.power(
        -mu2 * xnor4 / c5 / time_proper_code + (1.0 - mu2),
        2.0 / (gamma - 1.0),
    )
    rho_ana[logical3] = rho3
    rho_ana[logical2] = rho2
    rho_ana[logical1] = rho1
    p_ana[logical5] = p5
    p_ana[logical4] = p5 * np.power(
        -mu2 * xnor4 / c5 / time_proper_code + (1.0 - mu2),
        2.0 * gamma / (gamma - 1.0),
    )
    p_ana[logical3] = p2
    p_ana[logical2] = p2
    p_ana[logical1] = p1
    v_ana[logical4] = (1.0 - mu2) * (xnor4 / time_proper_code + c5)
    v_ana[logical3] = v2
    v_ana[logical2] = v2
    return rho_ana, p_ana, v_ana


def build_initial_condition(config):
    ic = config["initial_condition"]
    units = config["_code_units"]
    grid_cells = int(ic["grid_cells"])
    size_proper_unyt = ic["box_size_proper"]
    boundary_proper_unyt = np.linspace(0.0, 1.0, grid_cells + 1) * size_proper_unyt
    x_proper_unyt = 0.5 * (
        boundary_proper_unyt[:-1] + boundary_proper_unyt[1:]
    )
    shocked_cells = (x_proper_unyt > 0.25 * size_proper_unyt) & (
        x_proper_unyt < 0.75 * size_proper_unyt
    )

    mu_dimensionless = np.full(
        grid_cells,
        float(ic["mean_molecular_weight"]),
    )

    writer = InitialConditionWriter(
        par_config=config["par"],
        code_units=units,
    )
    writer.box_size = writer.radquantity(size_proper_unyt)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(grid_cells)
        * ic["rho_proper"]
        * np.where(shocked_cells, ic["density_ratio"], 1.0)
    )
    writer.fluid.vel_radarray = writer.radarray(
        np.ones(grid_cells) * ic["vel_proper"]
    )
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(grid_cells)
        * ic["temperature_proper"]
        * np.where(shocked_cells, ic["temperature_ratio"], 1.0)
    )
    writer.simulation.fluid.mu = mu_dimensionless
    return writer

def analytic_density_profile(config, state):
    ic = config["initial_condition"]
    units = config["_code_units"]
    rho_high_proper_cgs_g_cm3 = float(
        ic["rho_proper"].to_value(unyt.g / unyt.cm**3)
    )
    rho_low_proper_cgs_g_cm3 = (
        rho_high_proper_cgs_g_cm3 * ic["density_ratio"]
    )
    temp_high_cgs_K = float(ic["temperature_proper"].to_value(unyt.K))
    temp_low_cgs_K = temp_high_cgs_K * ic["temperature_ratio"]
    mu_dimensionless = float(ic["mean_molecular_weight"])
    pressure_low_proper_cgs_erg_cm3 = float(
        (
            rho_low_proper_cgs_g_cm3
            * unyt.g / unyt.cm**3
            / (mu_dimensionless * unyt.mp)
            * unyt.kb
            * temp_low_cgs_K
            * unyt.K
        ).to_value(unyt.erg / unyt.cm**3)
    )
    pressure_high_proper_cgs_erg_cm3 = float(
        (
            rho_high_proper_cgs_g_cm3
            * unyt.g / unyt.cm**3
            / (mu_dimensionless * unyt.mp)
            * unyt.kb
            * temp_high_cgs_K
            * unyt.K
        ).to_value(unyt.erg / unyt.cm**3)
    )
    rho2, rho3, p2, v2, vt, vs, _ = shocktubecal(
        config["par"]["hydrodynamics"]["gamma"],
        rho_low_proper_cgs_g_cm3,
        rho_high_proper_cgs_g_cm3,
        pressure_low_proper_cgs_erg_cm3,
        pressure_high_proper_cgs_erg_cm3,
    )
    time_proper_cgs_s = float(
        (
            float(np.asarray(state.fluid.time_proper_code).flat[0])
            * units.time_unit
        ).to_value(unyt.s)
    )
    if time_proper_cgs_s <= 0.0:
        return None
    boundary_proper_cgs_cm = np.asarray(
        state.mesh.boundary_radarray.to_cgs().to_value(unyt.cm),
        dtype=float,
    )
    centers_proper_cgs_cm = 0.5 * (
        boundary_proper_cgs_cm[:-1] + boundary_proper_cgs_cm[1:]
    )
    box_proper_cgs_cm = float(ic["box_size_proper"].to_value(unyt.cm))
    interface_proper_cgs_cm = 0.25 * box_proper_cgs_cm
    left, _, _ = shocktubeanalyticgraph(
        config["par"]["hydrodynamics"]["gamma"],
        rho_low_proper_cgs_g_cm3,
        rho2,
        rho3,
        rho_high_proper_cgs_g_cm3,
        pressure_low_proper_cgs_erg_cm3,
        p2,
        pressure_high_proper_cgs_erg_cm3,
        v2,
        vt,
        vs,
        time_proper_cgs_s,
        centers_proper_cgs_cm,
        interface_proper_cgs_cm,
    )
    mirrored, _, _ = shocktubeanalyticgraph(
        config["par"]["hydrodynamics"]["gamma"],
        rho_low_proper_cgs_g_cm3,
        rho2,
        rho3,
        rho_high_proper_cgs_g_cm3,
        pressure_low_proper_cgs_erg_cm3,
        p2,
        pressure_high_proper_cgs_erg_cm3,
        v2,
        vt,
        vs,
        time_proper_cgs_s,
        box_proper_cgs_cm - centers_proper_cgs_cm,
        interface_proper_cgs_cm,
    )
    return np.where(
        centers_proper_cgs_cm <= 0.5 * box_proper_cgs_cm,
        left,
        mirrored,
    )

def plot_snapshot(filename, config, **kwargs):
    sim = rio.loadhdf5(config, filename)
    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)

    boundary_proper_cgs_cm = np.asarray(
        sim.mesh.boundary_radarray.to_cgs().to_value(unyt.cm),
        dtype=float,
    )
    rho_proper_cgs_g_cm3 = np.asarray(
        sim.fluid.rho_radarray.to_cgs().to_value(unyt.g / unyt.cm**3),
        dtype=float,
    )
    x_proper_cgs_cm = 0.5 * (
        boundary_proper_cgs_cm[:-1] + boundary_proper_cgs_cm[1:]
    )
    plt.plot(
        x_proper_cgs_cm[first:last],
        rho_proper_cgs_g_cm3[first:last],
        **kwargs,
    )
    analytic_density_proper_cgs_g_cm3 = analytic_density_profile(config, sim)
    if analytic_density_proper_cgs_g_cm3 is not None:
        plt.plot(
            x_proper_cgs_cm[first:last],
            analytic_density_proper_cgs_g_cm3[first:last],
            color=kwargs.get("color"),
            linestyle="--",
            label="analytic",
        )
