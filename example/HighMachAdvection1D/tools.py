"""Initial conditions and analysis helpers for high-Mach advection."""

import numpy as np

from radhydropy.eos import EOS


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


class Simwrap:
    """Build a uniform periodic gas state in the requested code units."""

    def __init__(self, icparams, code_units=None):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.CodeUnits = code_units
        if code_units is not None:
            self.par.unit_system = code_units.unit_system
        self.par.nogrid = int(icparams["nogrid"])
        self.par.coordsys = "cartesian"
        self.par.boxsize = icparams["boxsize"]
        self.par.time = icparams["time"]
        self.mesh.boundary = np.linspace(
            0.0 * self.par.boxsize,
            self.par.boxsize,
            self.par.nogrid + 1,
        )
        cell_center = 0.5 * (self.mesh.boundary[:-1] + self.mesh.boundary[1:])
        if "rho_left" in icparams or "rho_right" in icparams:
            rho_left = icparams.get("rho_left", icparams.get("rhoini"))
            rho_right = icparams.get("rho_right", icparams.get("rhoini"))
            self.fluid.rho = np.where(cell_center < 0.5 * self.par.boxsize, rho_left, rho_right)
        else:
            self.fluid.rho = np.ones(self.par.nogrid) * icparams["rhoini"]
        self.fluid.vel = np.ones(self.par.nogrid) * icparams["vini"]
        self.fluid.mu = np.ones(self.par.nogrid) * icparams["muini"]
        if "temp_left" in icparams or "temp_right" in icparams:
            temp_left = icparams.get("temp_left", icparams.get("tempini", 0.0))
            temp_right = icparams.get("temp_right", icparams.get("tempini", 0.0))
            self.fluid.temp = np.where(
                cell_center < 0.5 * self.par.boxsize,
                temp_left,
                temp_right,
            )
        elif "pressureini" in icparams:
            pressure = float(icparams["pressureini"])
            pressure_factor = np.longdouble(
                code_units.unit_conversion["boltzmann_code"]
                / code_units.unit_conversion["proton_mass_code"]
            )
            self.fluid.temp = np.asarray(
                self.fluid.mu * pressure / (self.fluid.rho * pressure_factor),
                dtype=float,
            )
        else:
            self.fluid.temp = np.ones(self.par.nogrid) * icparams["tempini"]


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
        state.fluid.mu = np.ones_like(np.asarray(state.fluid.rho, dtype=float))
    if not hasattr(state.fluid, "pre"):
        state.fluid.pre = state.fluid.eos.pressure(
            state.fluid.rho,
            state.fluid.temp,
            state.fluid.mu,
        )
    rho = np.asarray(state.fluid.rho, dtype=float)
    velocity = np.asarray(state.fluid.vel, dtype=float)
    pressure = np.asarray(state.fluid.pre, dtype=float)
    if hasattr(state.mesh, "vol"):
        volume = np.asarray(state.mesh.vol, dtype=float)
    else:
        boundary = np.asarray(state.mesh.boundary, dtype=float)
        volume = np.diff(boundary)
    kinetic = 0.5 * rho * velocity**2 * volume
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
    rho = np.asarray(state.fluid.rho, dtype=float)
    temperature = np.asarray(state.fluid.temp, dtype=float)
    boundary = np.asarray(state.mesh.boundary, dtype=float)
    first = int(getattr(state.par, "noghost", 0))
    last = first + int(state.par.nogrid)
    radius = 0.5 * (boundary[:-1] + boundary[1:])
    gamma = float(getattr(state.par, "gamma", 5.0 / 3.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = temperature / rho ** (gamma - 1.0)
    return radius[first:last], entropy[first:last]


def primitive_profiles(state):
    """Return physical-cell radius, density, and temperature profiles."""
    rho = np.asarray(state.fluid.rho, dtype=float)
    temperature = np.asarray(state.fluid.temp, dtype=float)
    boundary = np.asarray(state.mesh.boundary, dtype=float)
    first = int(getattr(state.par, "noghost", 0))
    last = first + int(state.par.nogrid)
    radius = 0.5 * (boundary[:-1] + boundary[1:])
    return radius[first:last], rho[first:last], temperature[first:last]
