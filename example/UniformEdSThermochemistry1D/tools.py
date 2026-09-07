"""Initial conditions and analytic reference for the uniform EdS source test."""

from types import SimpleNamespace

import numpy as np
import unyt

from radhydropy.constants import PROTON_MASS_CGS
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
)
from radhydropy.units import quantity_to_value


class UniformEdSInitialCondition:
    """Build a few-cell uniform supercomoving initial condition."""

    def __init__(self, config, code_unit_system, cosmology):
        initial_condition = config["initial_condition"]
        mesh_config = config["par"]["mesh"]
        self.par = SimpleNamespace()
        self.mesh = SimpleNamespace()
        self.fluid = SimpleNamespace()

        count = int(mesh_config["grid_cells"])
        rmin = quantity_to_value(initial_condition["inner_radius"], code_unit_system.length_unit)
        rmax = quantity_to_value(initial_condition["outer_radius"], code_unit_system.length_unit)
        initial_time = float(initial_condition["initial_cosmic_time"])

        self.par.code_unit_system = SimpleNamespace(CodeUnits=code_unit_system)
        self.par.CodeUnits = code_unit_system
        self.par.units = SimpleNamespace(CodeUnits=code_unit_system)
        self.par.unit_system = code_unit_system.unit_system
        self.par.nogrid = count
        self.par.coordsys = "spherical"
        self.par.boxsize = np.asarray([rmax])
        self.par.time_proper_code = np.asarray([initial_time], dtype=float)
        self.par.cosmological_expansion = False
        self.par.supercomoving_coordinates = False
        self.par.cosmological_gravity = False
        self.par.selfgravity = False
        self.par.externalgravity = False
        self.par.cosmology = cosmology
        self.par.cosmology_type = cosmology.type_name
        self.par.cosmology_t_ref = cosmology.t_ref
        self.par.cosmology_a_ref = cosmology.a_ref
        self.par.coordinate_frame = "physical"
        self.par.time_coordinate = "cosmic"
        self.par.velocity_representation = "physical"
        self.par.density_representation = "physical"
        self.par.pressure_representation = "physical"
        self.par.temperature_representation = "physical"
        self.par.simulation = SimpleNamespace(
            time_proper_code=initial_time,
            box_size=np.asarray([rmax]),
            coordinate_system="spherical",
        )
        self.par.mesh = SimpleNamespace(grid_cells=count, ghost_cells=0)

        self.mesh.boundary_proper_code = np.linspace(rmin, rmax, count + 1)
        self.mesh.x_proper_code = 0.75 * (
            self.mesh.boundary_proper_code[1:] ** 4 - self.mesh.boundary_proper_code[:-1] ** 4
        ) / np.maximum(
            self.mesh.boundary_proper_code[1:] ** 3 - self.mesh.boundary_proper_code[:-1] ** 3,
            1.0e-300,
        )
        self.mesh.area_proper_code = 4.0 * np.pi * self.mesh.boundary_proper_code[:-1] ** 2
        self.mesh.volume_proper_code = 4.0 * np.pi / 3.0 * np.diff(
            self.mesh.boundary_proper_code ** 3
        )

        nH = float(initial_condition["hydrogen_density_cgs_cm3"])
        hydrogen_fraction = float(initial_condition["hydrogen_mass_fraction"])
        rho_physical = nH * PROTON_MASS_CGS / hydrogen_fraction
        rho_proper_code = rho_physical / float(code_unit_system.density_unit.to_value("g/cm**3"))

        temperature = float(initial_condition["temperature_cgs_K"])
        xHI = float(initial_condition["xHI"])
        mu = 1.0 / (hydrogen_fraction * (2.0 - xHI))

        self.fluid.rho_proper_code = np.full(count, rho_proper_code)
        self.fluid.vel_proper_code = np.zeros(count)
        temperature_unit_cgs_K = float(code_unit_system.temperature_unit.to_value("K"))
        self.fluid.temp_proper_code = np.full(count, temperature / temperature_unit_cgs_K)
        self.fluid.xHI = np.full(count, xHI)
        self.fluid.mu = np.full(count, mu)
        self.fluid.pre_proper_code = (
            self.fluid.rho_proper_code * self.fluid.temp_proper_code
        )
        self.fluid.time_proper_code = initial_time
        self.mesh.geometry_state = MeshGeometryState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            coordinate=self.mesh.x_proper_code,
            boundary=self.mesh.boundary_proper_code,
            width=np.diff(self.mesh.boundary_proper_code),
            area=self.mesh.area_proper_code,
            volume=self.mesh.volume_proper_code,
        )
        self.fluid.runtime_fields = PROPER_RUNTIME_FIELDS
        self.fluid.runtime_state = FluidRuntimeState.from_arrays(
            PROPER_RUNTIME_FIELDS,
            density=self.fluid.rho_proper_code,
            velocity=self.fluid.vel_proper_code,
            pressure=self.fluid.pre_proper_code,
            temperature=self.fluid.temp_proper_code,
            time=self.fluid.time_proper_code,
            mu=self.fluid.mu,
        )


def analytic_compton_temperature(
    cosmic_times_s,
    initial_temperature_cgs_K,
    initial_cosmic_time,
    cosmology,
    time_unit_s,
    hydrogen_density_cgs_cm3,
    hydrogen_mass_fraction,
    xHI,
    gamma,
    cmb_temperature_0_cgs_K,
    mu,
):
    """Return the EdS Compton-only solution using the linear ODE integral."""
    from scipy.integrate import solve_ivp

    from radhydropy.thermo_networks.compton import cmb_compton_rate

    rho = (
        hydrogen_density_cgs_cm3 * PROTON_MASS_CGS / hydrogen_mass_fraction
    )
    ne = hydrogen_density_cgs_cm3 * (1.0 - xHI)
    source_slope = float(
        cmb_compton_rate(
            np.asarray([0.0]), np.asarray([ne]), enabled=True, redshift=0.0
        )[0]
    )
    # The source is C * ne * Tcmb^4 * (Tcmb - T), so the EdS factor is
    # a^-4.  At z=0, source_slope is C * ne * T0^5.
    coefficient = source_slope / cmb_temperature_0_cgs_K
    temperature_coefficient = (
        (gamma - 1.0) * mu * PROTON_MASS_CGS / rho
        / float(unyt.kb.to_value("erg/K"))
        * coefficient
    )

    initial_time_s = float(initial_cosmic_time) * time_unit_s
    final_time_s = float(np.max(cosmic_times_s))

    def rhs(time_s, values):
        time_proper_code = time_s / time_unit_s
        scale_factor = float(cosmology.scale_factor(time_proper_code))
        cmb_temperature = cmb_temperature_0_cgs_K / scale_factor
        return [temperature_coefficient * scale_factor ** -4 * (cmb_temperature - values[0])]

    solution = solve_ivp(
        rhs,
        (initial_time_s, final_time_s),
        [initial_temperature_cgs_K],
        t_eval=np.asarray(cosmic_times_s, dtype=float),
        rtol=1.0e-10,
        atol=1.0e-8,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[0]
