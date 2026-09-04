"""Initial conditions and analytic reference for the uniform EdS source test."""

from types import SimpleNamespace

import numpy as np
import unyt

from radhydropy.constants import PROTON_MASS_CGS
from radhydropy.units import quantity_to_value


class UniformEdSInitialCondition:
    """Build a few-cell uniform supercomoving initial condition."""

    def __init__(self, icparams, mesh_config, units, cosmology):
        self.par = SimpleNamespace()
        self.mesh = SimpleNamespace()
        self.fluid = SimpleNamespace()

        count = int(mesh_config["grid_cells"])
        rmin = quantity_to_value(icparams["inner_radius"], units.length_unit)
        rmax = quantity_to_value(icparams["outer_radius"], units.length_unit)
        initial_time = float(icparams["initial_cosmic_time"])

        self.par.CodeUnits = units
        self.par.units = SimpleNamespace(CodeUnits=units)
        self.par.unit_system = units.unit_system
        self.par.nogrid = count
        self.par.coordsys = "spherical"
        self.par.boxsize = np.asarray([rmax])
        self.par.time = np.asarray([initial_time], dtype=float)
        self.par.cosmological_expansion = True
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
            current_time=initial_time,
            box_size=np.asarray([rmax]),
            coordinate_system="spherical",
        )
        self.par.mesh = SimpleNamespace(grid_cells=count, ghost_cells=0)

        self.mesh.boundary = np.linspace(rmin, rmax, count + 1)
        self.mesh.coordinate = 0.75 * (
            self.mesh.boundary[1:] ** 4 - self.mesh.boundary[:-1] ** 4
        ) / np.maximum(
            self.mesh.boundary[1:] ** 3 - self.mesh.boundary[:-1] ** 3,
            1.0e-300,
        )
        self.mesh.area = 4.0 * np.pi * self.mesh.boundary[:-1] ** 2
        self.mesh.vol = 4.0 * np.pi / 3.0 * np.diff(self.mesh.boundary ** 3)

        nH = float(icparams["hydrogen_density_cgs_cm3"])
        hydrogen_fraction = float(icparams["hydrogen_mass_fraction"])
        rho_physical = nH * PROTON_MASS_CGS / hydrogen_fraction
        rho_code = rho_physical / float(units.density_unit.to_value("g/cm**3"))

        temperature = float(icparams["temperature_cgs_K"])
        xHI = float(icparams["xHI"])
        mu = 1.0 / (hydrogen_fraction * (2.0 - xHI))

        self.fluid.rho_code = np.full(count, rho_code)
        self.fluid.vel_code = np.zeros(count)
        temperature_unit_cgs_K = float(units.temperature_unit.to_value("K"))
        self.fluid.temp_code = np.full(count, temperature / temperature_unit_cgs_K)
        self.fluid.xHI = np.full(count, xHI)
        self.fluid.mu = np.full(count, mu)


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
        time_code = time_s / time_unit_s
        scale_factor = float(cosmology.scale_factor(time_code))
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
