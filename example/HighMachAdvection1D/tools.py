"""Initial conditions and analysis helpers for high-Mach advection."""

import numpy as np
from types import SimpleNamespace

from radhydropy.eos import EOS


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def build_initial_condition(config):
    initial = config['initial_condition']
    code_units = config['_code_units']
    grid_cells = int(initial['grid_cells'])
    result = SimpleNamespace(par=Par(), mesh=Mesh(), fluid=Fluid())
    result.par.units = SimpleNamespace(CodeUnits=code_units)
    result.par.unit_system = code_units.unit_system
    result.par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=grid_cells)
    result.par.simulation = SimpleNamespace(
        coordinate_system=initial['coordinate_system'],
        box_size=initial['box_size'], current_time=initial['current_time'],
    )
    result.par.nogrid = grid_cells
    result.par.coordsys = initial['coordinate_system']
    result.par.boxsize = initial['box_size']
    result.par.time = initial['current_time']
    result.mesh.boundary = np.linspace(0.0 * result.par.boxsize, result.par.boxsize, grid_cells + 1)
    center = 0.5 * (result.mesh.boundary[:-1] + result.mesh.boundary[1:])
    result.fluid.rho_code = np.where(
        center < 0.5 * result.par.boxsize,
        initial.get('rho_left', initial['initial_density']),
        initial.get('rho_right', initial['initial_density']),
    )
    result.fluid.vel_code = np.ones(grid_cells) * initial['initial_velocity']
    result.fluid.mu = np.ones(grid_cells) * initial['mean_molecular_weight']
    if 'temp_left' in initial or 'temp_right' in initial:
        result.fluid.temp_code = np.where(
            center < 0.5 * result.par.boxsize,
            initial.get('temp_left', initial.get('initial_temperature', 0.0)),
            initial.get('temp_right', initial.get('initial_temperature', 0.0)),
        )
    else:
        result.fluid.temp_code = np.ones(grid_cells) * initial['initial_temperature']
    return result


def energy_components(state):
    """Return total, kinetic, and thermal energy for a loaded snapshot."""
    # HDF5 snapshots store primitive fields, but not the runtime EOS object or
    # derived pressure.  Rebuild those here before evaluating the energy sum.
    if not hasattr(state.fluid, "eos") or state.fluid.eos is None:
        state.fluid.eos = EOS(
            getattr(state.par, "EOStype", "polytropic"),
            float(getattr(state.par, "gamma", 5.0 / 3.0)),
            getattr(state.par, "CodeUnits", None),
        )
    if not hasattr(state.fluid, "mu"):
        state.fluid.mu = np.ones_like(np.asarray(state.fluid.rho_code, dtype=float))
    if not hasattr(state.fluid, "pre"):
        state.fluid.pre_code = state.fluid.eos.pressure(
            state.fluid.rho_code,
            state.fluid.temp_code,
            state.fluid.mu,
        )
    rho_code = np.asarray(state.fluid.rho_code, dtype=float)
    velocity = np.asarray(state.fluid.vel_code, dtype=float)
    pressure = np.asarray(state.fluid.pre_code, dtype=float)
    if hasattr(state.mesh, "vol"):
        volume = np.asarray(state.mesh.vol, dtype=float)
    else:
        boundary = np.asarray(state.mesh.boundary, dtype=float)
        volume = np.diff(boundary)
    kinetic = 0.5 * rho_code * velocity**2 * volume
    thermal = pressure / (state.fluid.eos.gamma - 1.0) * volume
    total = kinetic + thermal
    first = int(getattr(state.par, "noghost", 0))
    last = first + int(state.par.nogrid)
    return {
        "total": float(np.sum(total[first:last])),
        "kinetic": float(np.sum(kinetic[first:last])),
        "thermal": float(np.sum(thermal[first:last])),
    }


def entropy_profile(state):
    """Return physical-cell radius and ``T/rho**(gamma-1)`` entropy proxy."""
    rho_code = np.asarray(state.fluid.rho_code, dtype=float)
    temperature = np.asarray(state.fluid.temp_code, dtype=float)
    boundary = np.asarray(state.mesh.boundary, dtype=float)
    first = int(getattr(state.par, "noghost", 0))
    last = first + int(state.par.nogrid)
    radius = 0.5 * (boundary[:-1] + boundary[1:])
    gamma = float(getattr(state.par, "gamma", 5.0 / 3.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = temperature / rho_code ** (gamma - 1.0)
    return radius[first:last], entropy[first:last]


def primitive_profiles(state):
    """Return physical-cell radius, density, and temperature profiles."""
    rho_code = np.asarray(state.fluid.rho_code, dtype=float)
    temperature = np.asarray(state.fluid.temp_code, dtype=float)
    boundary = np.asarray(state.mesh.boundary, dtype=float)
    first = int(getattr(state.par, "noghost", 0))
    last = first + int(state.par.nogrid)
    radius = 0.5 * (boundary[:-1] + boundary[1:])
    return radius[first:last], rho_code[first:last], temperature[first:last]

