"""Analytic and I/O helpers for the self-gravitating n=1 polytrope."""

import numpy as np
import unyt
from types import SimpleNamespace

from radhydropy.constants import (
    BOLTZMANN_CONSTANT_CGS,
    GRAVITATIONAL_CONSTANT_CGS,
    PROTON_MASS_CGS,
)
import radhydropy.io as rio
from radhydropy.eos import EOS
from radhydropy.units import CodeUnits


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def spherical_cell_centers(boundary):
    inner = boundary[:-1]
    outer = boundary[1:]
    return 0.75 * (outer**4 - inner**4) / (outer**3 - inner**3)


def polytropic_constant(radius):
    """Return K for an n=1 sphere with radius ``pi * radius``."""
    radius = radius.to(unyt.cm)
    return 2.0 * np.pi * (
        GRAVITATIONAL_CONSTANT_CGS * unyt.cm**3 / (unyt.g * unyt.s**2)
    ) * radius**2


def equilibrium_density(radius, central_density, polytropic_radius):
    """Return rho_c sin(x)/x, with x=r/a and R=pi*a."""
    radius = radius.to(unyt.cm)
    scale = polytropic_radius.to(unyt.cm)
    x = np.asarray(radius / scale, dtype=float)
    profile = np.ones_like(x)
    nonzero = x != 0.0
    profile[nonzero] = np.sin(x[nonzero]) / x[nonzero]
    return central_density.to(unyt.g / unyt.cm**3) * profile


def equilibrium_pressure(density, polytropic_k):
    return polytropic_k * density**2


def equilibrium_temperature(density, polytropic_k, mu):
    pressure = equilibrium_pressure(density, polytropic_k)
    return (
        pressure * mu * PROTON_MASS_CGS
        * (unyt.g)
        / (
            density
            * BOLTZMANN_CONSTANT_CGS
            * (unyt.erg / unyt.K)
        )
    ).to(unyt.K)


def hydrostatic_residual(radius, density, pressure, acceleration):
    """Return dP/dr + rho*g on cell centers using centered differences."""
    radius = np.asarray(radius, dtype=float)
    density = np.asarray(density, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    gradient = np.gradient(pressure, radius)
    return gradient + density * acceleration


class Simwrap:
    def __init__(self, icparams, code_units):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        grid_cells = int(icparams['nogrid'])
        box_size = np.ones(1) * icparams['boxsize']
        self.par.time = np.ones(1) * icparams['time']
        self.par.simulation = SimpleNamespace(
            current_time=self.par.time,
            box_size=box_size,
            coordinate_system='spherical',
        )
        self.par.mesh = SimpleNamespace(grid_cells=grid_cells, ghost_cells=0)
        self.par.hydrodynamics = SimpleNamespace(gamma=2.0)

        self.mesh.boundary = np.linspace(
            icparams['rmin'], icparams['rmax'], grid_cells + 1
        )
        self.mesh.coordinate = spherical_cell_centers(self.mesh.boundary)
        self.mesh.area = 4.0 * np.pi * self.mesh.boundary[:-1]**2
        self.mesh.vol = 4.0 * np.pi / 3.0 * (
            self.mesh.boundary[1:]**3 - self.mesh.boundary[:-1]**3
        )

        radius = np.asarray(self.mesh.coordinate, dtype=float) * code_units.length_unit
        density = equilibrium_density(
            radius,
            icparams['central_density'],
            icparams['polytropic_radius'],
        )
        k_poly = polytropic_constant(icparams['polytropic_radius'])
        self.fluid.rho_code = density
        self.fluid.temp_code = equilibrium_temperature(density, k_poly, icparams['mu'])
        self.fluid.mu = np.ones(grid_cells) * icparams['mu']
        radius_fraction = radius / icparams['polytropic_radius']
        self.fluid.vel_code = (
            icparams['velocity_perturbation']
            * np.asarray(radius_fraction, dtype=float)
            * code_units.velocity_unit
        )


def write_used_parameters(*args, **kwargs):
    return None


def read_output(filename, runparams):
    code_units = CodeUnits.from_mapping(runparams['CodeUnits'])
    result = Simwrap.__new__(Simwrap)
    result.par = Par()
    result.mesh = Mesh()
    result.fluid = Fluid()
    result.par.units = SimpleNamespace(CodeUnits=code_units)
    result.par.simulation = SimpleNamespace(coordinate_system='spherical')
    result.par.mesh = SimpleNamespace(grid_cells=int(runparams['nogrid']), ghost_cells=int(runparams.get('noghost', 2)))
    result.par.hydrodynamics = SimpleNamespace(
        eos_type=runparams.get('EOStype', 'polytropic'),
        gamma=float(runparams.get('gamma', 2.0)),
    )
    rio.readhdf5(result.par, result.mesh, result.fluid, filename)
    result.mesh.coordinate = spherical_cell_centers(result.mesh.boundary)
    result.fluid.eos = EOS(
        result.par.hydrodynamics.eos_type,
        result.par.hydrodynamics.gamma,
        code_units,
    )
    result.fluid.pre_code = result.fluid.eos.pressure(
        result.fluid.rho_code,
        result.fluid.temp_code,
        result.fluid.mu,
    )
    return result
