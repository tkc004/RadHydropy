"""Analytic setup and diagnostics for the pressure-supported core test."""

import numpy as np
import unyt

from radhydropy.constants import (
    BOLTZMANN_CONSTANT_CGS,
    GRAVITATIONAL_CONSTANT_CGS,
    PROTON_MASS_CGS,
)
from radhydropy.units import code_unit_scales, quantity_to_value
from radhydropy.runtime_fields import MeshGeometryState, FluidRuntimeState, PROPER_RUNTIME_FIELDS
from radhydropy.rsim import Rsim


def spherical_cell_centers(boundary):
    boundary = np.asarray(boundary, dtype=float)
    denominator = boundary[1:] ** 3 - boundary[:-1] ** 3
    center = 0.5 * (boundary[1:] + boundary[:-1])
    valid = denominator != 0.0
    center[valid] = 0.75 * (
        boundary[1:][valid] ** 4 - boundary[:-1][valid] ** 4
    ) / denominator[valid]
    return center


def point_mass_density(
    radius, rho_ref, temperature, mu, point_mass, reference_radius,
):
    """Exact isothermal hydrostatic density around a point mass."""
    radius_cgs_cm = np.asarray(radius.to_value(unyt.cm), dtype=float)
    ref_cm = float(reference_radius.to_value(unyt.cm))
    rho_ref_cgs = float(rho_ref.to_value(unyt.g / unyt.cm**3))
    mass_g = float(point_mass.to_value(unyt.g))
    sound_speed_squared = (
        BOLTZMANN_CONSTANT_CGS * float(temperature.to_value(unyt.K))
        / (float(mu) * PROTON_MASS_CGS)
    )
    potential_difference = (
        -GRAVITATIONAL_CONSTANT_CGS * mass_g / radius_cgs_cm
        + GRAVITATIONAL_CONSTANT_CGS * mass_g / ref_cm
    )
    return rho_ref_cgs * np.exp(-potential_difference / sound_speed_squared) * (
        unyt.g / unyt.cm**3
    )


class InitialCondition(Rsim):
    """Minimal HDF5-compatible analytic initial-condition container."""

    def __init__(self, config, code_units):
        initial_condition = config["initial_condition"]
        grid_cells = int(config["par"]["mesh"]["grid_cells"])
        super().__init__(config["par"])
        self.par.mesh.grid_cells = grid_cells
        self.par.simulation.coordinate_system = "spherical"
        self.par.simulation.time_proper_code = 0.0
        self.par.simulation.box_size = np.asarray(
            [float(initial_condition["outer_radius"].to_value(code_units.length_unit))]
        )

        self.mesh.boundary = np.linspace(
            float(initial_condition["inner_radius"].to_value(code_units.length_unit)),
            float(initial_condition["outer_radius"].to_value(code_units.length_unit)),
            grid_cells + 1,
        )
        self.mesh.coordinate = spherical_cell_centers(self.mesh.boundary)
        self.mesh.area = 4.0 * np.pi * self.mesh.boundary[:-1] ** 2
        self.mesh.vol = (
            (self.mesh.boundary[1:] ** 3 - self.mesh.boundary[:-1] ** 3)
            * 4.0 * np.pi / 3.0
        )
        self.fluid.rho_proper_code = point_mass_density(
            self.mesh.coordinate * code_units.length_unit,
            initial_condition["reference_density"],
            initial_condition["initial_temperature"],
            initial_condition["mean_molecular_weight"],
            initial_condition["point_mass"],
            self.mesh.coordinate[0] * code_units.length_unit,
        )
        scales = code_unit_scales(code_units)
        self.fluid.rho_proper_code = quantity_to_value(
            self.fluid.rho_proper_code, code_units.density_unit
        )
        self.fluid.temp_proper_code = np.full(
            grid_cells,
            float(initial_condition["initial_temperature"].to_value(unyt.K))
            / scales["temperature_cgs_K"],
        )
        self.fluid.mu = np.full(grid_cells, float(initial_condition["mean_molecular_weight"]))
        self.fluid.vel_proper_code = np.zeros(grid_cells)
        self.mesh.geometry_state = MeshGeometryState.from_arrays(
            PROPER_RUNTIME_FIELDS, coordinate=self.mesh.coordinate,
            boundary=self.mesh.boundary, width=np.diff(self.mesh.boundary),
            area=self.mesh.area, volume=self.mesh.vol,
        )
        self.fluid.pre_proper_code = self.fluid.rho_proper_code * self.fluid.temp_proper_code
        self.fluid.time_proper_code = 0.0
        self.fluid.runtime_fields = PROPER_RUNTIME_FIELDS
        self.fluid.runtime_state = FluidRuntimeState.from_arrays(
            PROPER_RUNTIME_FIELDS, density=self.fluid.rho_proper_code,
            velocity=self.fluid.vel_proper_code, pressure=self.fluid.pre_proper_code,
            temperature=self.fluid.temp_proper_code, time=0.0, mu=self.fluid.mu,
        )
        self.solver.SetConserved(self.mesh, self.fluid, verbose=0)


def analytic_density_code(radius_code, config, code_units):
    initial_condition = config["initial_condition"]
    density = point_mass_density(
        np.asarray(radius_code) * code_units.length_unit,
        initial_condition["reference_density"],
        initial_condition["initial_temperature"],
        initial_condition["mean_molecular_weight"],
        initial_condition["point_mass"],
        float(radius_code[0]) * code_units.length_unit,
    )
    return quantity_to_value(density, code_units.density_unit)
