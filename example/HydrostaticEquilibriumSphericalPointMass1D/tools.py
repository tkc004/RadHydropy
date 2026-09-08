"""Helper utilities for the spherical hydrostatic point-mass example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, GRAVITATIONAL_CONSTANT_CGS, PROTON_MASS_CGS
import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.units import (
    CodeUnits,
    code_quantity_to_cgs,
    code_unit_scales,
    quantity_to_value,
)
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, PROPER_RUNTIME_FIELDS

SPEED_SQUARED_UNIT = unyt.cm**2 / unyt.s**2
DENSITY_UNIT = unyt.g / unyt.cm**3
ACCELERATION_UNIT = unyt.cm / unyt.s**2


def sound_speed_squared(temperature_proper_code, mu, code_unit_system=None):
    """Return the isothermal sound speed squared."""
    if hasattr(temperature_proper_code, "to_value"):
        temp_value = float(temperature_proper_code.to_value(unyt.K))
    elif code_unit_system is not None:
        temp_value = float(np.asarray(temperature_proper_code, dtype=float)) * code_unit_scales(code_unit_system)["temperature_cgs_K"]
    else:
        temp_value = float(temperature_proper_code)
    mu_value = float(np.asarray(mu, dtype=float))
    return (
        BOLTZMANN_CONSTANT_CGS
        * temp_value
        / (mu_value * PROTON_MASS_CGS)
    ) * SPEED_SQUARED_UNIT


def spherical_cell_centers(boundary_proper_code):
    """Return spherical cell centers consistent with the mesh geometry."""
    x_proper_code = 0.5 * (boundary_proper_code[1:] + boundary_proper_code[:-1])
    vol_denom = boundary_proper_code[1:]**3 - boundary_proper_code[:-1]**3
    nonzero = vol_denom != 0.0
    x_proper_code[nonzero] = 0.75 * (
        boundary_proper_code[1:][nonzero]**4 - boundary_proper_code[:-1][nonzero]**4
    ) / vol_denom[nonzero]
    return x_proper_code


def point_mass_hydrostatic_density_profile(
    x_proper_code,
    rho_ref,
    temperature_proper_code,
    mu,
    point_mass,
    reference_radius,
    code_unit_system=None,
):
    """Return the exact isothermal hydrostatic density profile."""
    c_s2 = sound_speed_squared(temperature_proper_code, mu, code_unit_system=code_unit_system)
    c_s2_value = c_s2.to_value(unyt.cm**2 / unyt.s**2)
    if hasattr(x_proper_code, "to_value"):
        coord_value = x_proper_code.to_value(unyt.cm)
    elif code_unit_system is not None:
        coord_value = np.asarray(x_proper_code, dtype=float) * code_unit_scales(code_unit_system)["length_cgs_cm"]
    else:
        coord_value = np.asarray(x_proper_code, dtype=float)
    if hasattr(reference_radius, "to_value"):
        reference_radius_value = reference_radius.to_value(unyt.cm)
    elif code_unit_system is not None:
        reference_radius_value = np.asarray(reference_radius, dtype=float) * code_unit_scales(code_unit_system)["length_cgs_cm"]
    else:
        reference_radius_value = float(reference_radius)
    if hasattr(point_mass, "to_value"):
        point_mass_value = point_mass.to_value(unyt.g)
    elif code_unit_system is not None:
        point_mass_value = np.asarray(point_mass, dtype=float) * code_unit_scales(code_unit_system)["mass_g"]
    else:
        point_mass_value = float(point_mass)
    if hasattr(rho_ref, "to_value"):
        rho_value = rho_ref.to_value(unyt.g / unyt.cm**3)
    elif code_unit_system is not None:
        rho_value = np.asarray(rho_ref, dtype=float) * code_unit_scales(code_unit_system)["density_cgs_g_cm3"]
    else:
        rho_value = float(rho_ref)
    phi_ref = -GRAVITATIONAL_CONSTANT_CGS * point_mass_value / reference_radius_value
    phi = -GRAVITATIONAL_CONSTANT_CGS * point_mass_value / coord_value
    exponent = -(phi - phi_ref) / c_s2_value
    return rho_value * np.exp(exponent) * DENSITY_UNIT


def point_mass_acceleration(point_mass, softening=0.0, code_unit_system=None):
    """Return a callable for a point-mass gravitational acceleration field."""
    if hasattr(point_mass, "to_value"):
        point_mass = point_mass.to_value(unyt.g)
    elif code_unit_system is not None:
        point_mass = np.asarray(point_mass, dtype=float) * code_unit_scales(code_unit_system)["mass_g"]
    else:
        point_mass = float(point_mass)
    if hasattr(softening, "to_value"):
        softening = softening.to_value(unyt.cm)
    elif code_unit_system is not None:
        softening = np.asarray(softening, dtype=float) * code_unit_scales(code_unit_system)["length_cgs_cm"]
    else:
        softening = float(softening)

    def _acceleration(x_proper_code):
        if hasattr(x_proper_code, "to_value"):
            radius = x_proper_code.to_value(unyt.cm)
        elif code_unit_system is not None:
            radius = np.asarray(x_proper_code, dtype=float) * code_unit_scales(code_unit_system)["length_cgs_cm"]
        else:
            radius = np.asarray(x_proper_code, dtype=float)
        radius = np.maximum(radius, softening)
        return (
            -GRAVITATIONAL_CONSTANT_CGS * point_mass / radius**2
        ) * ACCELERATION_UNIT

    return _acceleration


def build_initial_condition(config):
    code_unit_system = CodeUnits.from_mapping(
        config['par']['units']['CodeUnits']
    )
    initial_condition = config['initial_condition']
    grid_cells = int(config['par']['mesh']['grid_cells'])
    sim = Rsim(config['par'])
    sim.par.mesh.grid_cells = grid_cells
    sim.par.mesh.ghost_cells = 0
    sim.par.simulation.coordinate_system = initial_condition['coordinate_system']
    sim.par.simulation.time_proper_code = quantity_to_value(initial_condition['time_proper'], code_unit_system.time_unit)
    sim.par.simulation.box_size_proper_code = quantity_to_value(initial_condition['box_size_proper'], code_unit_system.length_unit)

    sim.mesh.boundary_proper_code = np.linspace(
        initial_condition['inner_radius'],
        initial_condition['outer_radius'],
        grid_cells + 1,
    )
    sim.mesh.x_proper_code = spherical_cell_centers(sim.mesh.boundary_proper_code)
    dx = sim.mesh.boundary_proper_code[1] - sim.mesh.boundary_proper_code[0]
    sim.mesh.area_proper_code = 4.0 * np.pi * sim.mesh.boundary_proper_code[:-1]**2
    sim.mesh.volume_proper_code = (
        np.absolute(sim.mesh.boundary_proper_code[1:]**3 - sim.mesh.boundary_proper_code[:-1]**3)
        * 4.0
        * np.pi
        / 3.0
    )

    sim.fluid.temp_proper_code = np.ones(grid_cells) * quantity_to_value(initial_condition['temperature_proper'], code_unit_system.temperature_unit)
    sim.fluid.mu = np.ones(grid_cells) * initial_condition['mean_molecular_weight']
    sim.fluid.vel_proper_code = np.zeros(grid_cells, dtype=float)
    sim.fluid.rho_proper_code = point_mass_hydrostatic_density_profile(
        sim.mesh.x_proper_code,
        initial_condition['reference_density'],
        initial_condition['temperature_proper'],
        initial_condition['mean_molecular_weight'],
        initial_condition['point_mass'],
        reference_radius=sim.mesh.x_proper_code[0],
        code_unit_system=code_unit_system,
    )
    boundary_proper_code = quantity_to_value(
        sim.mesh.boundary_proper_code, code_unit_system.length_unit
    )
    coordinate_proper_code = quantity_to_value(
        sim.mesh.x_proper_code, code_unit_system.length_unit
    )
    sim.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=coordinate_proper_code,
        boundary_proper_code=boundary_proper_code,
        width_proper_code=np.diff(boundary_proper_code),
        area_proper_code=sim.mesh.area_proper_code,
        volume_proper_code=sim.mesh.volume_proper_code,
    )
    sim.fluid.rho_proper_code = quantity_to_value(sim.fluid.rho_proper_code, code_unit_system.density_unit)
    sim.fluid.vel_proper_code = np.zeros(grid_cells)
    sim.fluid.temp_proper_code = quantity_to_value(sim.fluid.temp_proper_code, code_unit_system.temperature_unit)
    sim.fluid.pre_proper_code = sim.fluid.rho_proper_code * sim.fluid.temp_proper_code
    sim.fluid.time_proper_code = 0.0
    sim.fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    sim.fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS, rho_proper_code=sim.fluid.rho_proper_code,
        vel_proper_code=sim.fluid.vel_proper_code, pre_proper_code=sim.fluid.pre_proper_code,
        temp_proper_code=sim.fluid.temp_proper_code, time_proper_code=0.0, mu_dimensionless=sim.fluid.mu,
    )


    sim.fluid.SetUpFluid(sim.par, sim.mesh)
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=0)
    return Rsim.FromComponents(sim.par, sim.mesh, sim.fluid, sim.solver)
def ReadandPlot(outfilename, config, **kwargs):
    """Read a snapshot and compare it with the analytic hydrostatic profile."""
    code_units_mapping = config['par']['units']['CodeUnits']
    code_units_obj = CodeUnits.from_mapping(code_units_mapping) if code_units_mapping is not None else None
    rout = build_initial_condition(config)
    if code_units_obj is not None:
        rout.par.unit_system = code_units_obj.unit_system
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    color = kwargs.get('color', 'C0')
    nghost = int(config['par']['mesh']['ghost_cells'])
    boundary_proper_code = rout.mesh.geometry_state.boundary_proper_code
    xall = spherical_cell_centers(boundary_proper_code)
    if nghost > 0:
        # Mesh geometry contains physical cells; only fluid arrays carry
        # ghost cells after HDF5 reload.
        xcoord = xall
        rho_num = rout.fluid.rho_proper_code[nghost:-nghost]
        vel_code_num = rout.fluid.vel_proper_code[nghost:-nghost]
    else:
        xcoord = xall
        rho_num = rout.fluid.rho_proper_code
        vel_code_num = rout.fluid.vel_proper_code
    rho_analytic = point_mass_hydrostatic_density_profile(
        xcoord,
        config['initial_condition']['reference_density'],
        config['initial_condition']['temperature_proper'],
        config['initial_condition']['mean_molecular_weight'],
        config['initial_condition']['point_mass'],
        reference_radius=xcoord[0],
        code_unit_system=code_units_obj,
    )
    zero_velocity = np.zeros(len(xcoord)) * unyt.cm / unyt.s
    x_units = getattr(xcoord, 'units', code_units_obj.length_unit.units if code_units_obj is not None else unyt.cm)
    rho_units = getattr(rho_num, 'units', code_units_obj.density_unit.units if code_units_obj is not None else unyt.g / unyt.cm**3)
    vel_units = getattr(vel_code_num, 'units', code_units_obj.velocity_unit.units if code_units_obj is not None else unyt.cm / unyt.s)
    xplot = code_quantity_to_cgs(xcoord, code_units_obj, 'length_cgs_cm')
    rho_num_plot = code_quantity_to_cgs(rho_num, code_units_obj, 'density_cgs_g_cm3')
    rho_analytic_plot = quantity_to_value(rho_analytic, unyt.g / unyt.cm**3)
    vel_num_plot = code_quantity_to_cgs(vel_code_num, code_units_obj, 'velocity_cgs_cm_s')
    zero_velocity_plot = quantity_to_value(zero_velocity, unyt.cm / unyt.s)

    plt.subplot(1, 2, 1)
    plt.plot(xplot, rho_num_plot, **kwargs)
    plt.plot(
        xplot,
        rho_analytic_plot,
        ls='dashed',
        color=color,
    )
    plt.xlabel(rf"$r \; [{x_units.latex_repr}]$")
    plt.ylabel(rf"$\rho \; [{rho_units.latex_repr}]$")

    plt.subplot(1, 2, 2)
    plt.plot(xplot, vel_num_plot, **kwargs)
    plt.plot(
        xplot,
        zero_velocity_plot,
        ls='dashed',
        color=color,
    )
    plt.xlabel(rf"$r \; [{x_units.latex_repr}]$")
    plt.ylabel(rf"$v \; [{vel_units.latex_repr}]$")
