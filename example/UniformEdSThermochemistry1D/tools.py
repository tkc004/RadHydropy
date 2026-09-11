"""Initial conditions and analytic reference for the uniform EdS source test."""

import numpy as np
import unyt

from radhydropy.constants import PROTON_MASS_CGS
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
)
from radhydropy.rsim import Rsim
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.units import CodeUnits, quantity_to_value


def build_initial_condition(config):
    """Build a typed few-cell proper-code initial condition."""
    code_unit_system = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    cosmology_config = config["par"]["cosmology"]
    cosmology = EinsteinDeSitter.from_code_units(
        code_unit_system,
        t_ref=quantity_to_value(cosmology_config["cosmology_t_ref"], code_unit_system.time_unit),
        a_ref=float(cosmology_config["cosmology_a_ref"]),
    )
    result = Rsim(config["par"])
    initial_condition = config["initial_condition"]
    count = int(config["par"]["mesh"]["grid_cells"])
    radius_inner_proper_code = quantity_to_value(
        initial_condition["radius_inner_proper"], code_unit_system.length_unit
    )
    radius_outer_proper_code = quantity_to_value(
        initial_condition["radius_outer_proper"], code_unit_system.length_unit
    )
    initial_time_proper_code = quantity_to_value(
        initial_condition["time_cosmic"], code_unit_system.time_unit
    )

    result.par.mesh.ghost_cells = 0
    result.par.simulation.coordinate_system = "spherical"
    result.par.simulation.box_size_comoving_code = np.asarray([radius_outer_proper_code])
    result.par.simulation.time_proper_code = initial_time_proper_code
    result.par.cosmology = cosmology
    result.par.time_proper_code = np.asarray([initial_time_proper_code])

    result.mesh.boundary_proper_code = np.linspace(
        radius_inner_proper_code, radius_outer_proper_code, count + 1
    )
    result.mesh.x_proper_code = 0.75 * (
        result.mesh.boundary_proper_code[1:] ** 4
        - result.mesh.boundary_proper_code[:-1] ** 4
    ) / np.maximum(
        result.mesh.boundary_proper_code[1:] ** 3
        - result.mesh.boundary_proper_code[:-1] ** 3,
        1.0e-300,
    )
    result.mesh.area_proper_code = 4.0 * np.pi * result.mesh.boundary_proper_code[:-1] ** 2
    result.mesh.volume_proper_code = 4.0 * np.pi / 3.0 * np.diff(
        result.mesh.boundary_proper_code ** 3
    )

    hydrogen_number_density_cgs_cm3 = float(initial_condition["hydrogen_number_density"].to_value("1/cm**3"))
    hydrogen_mass_fraction = float(initial_condition["hydrogen_mass_fraction"])
    rho_cgs_g_cm3 = hydrogen_number_density_cgs_cm3 * PROTON_MASS_CGS / hydrogen_mass_fraction
    rho_proper_code = rho_cgs_g_cm3 / float(code_unit_system.density_unit.to_value("g/cm**3"))
    temperature_cgs_K = float(initial_condition["temperature_proper"].to_value("K"))
    temperature_proper_code = temperature_cgs_K / float(code_unit_system.temperature_unit.to_value("K"))
    xHI_dimensionless = float(initial_condition["xHI"])
    mu_dimensionless = 1.0 / (hydrogen_mass_fraction * (2.0 - xHI_dimensionless))

    result.fluid.rho_proper_code = np.full(count, rho_proper_code)
    result.fluid.vel_proper_code = np.zeros(count)
    result.fluid.temp_proper_code = np.full(count, temperature_proper_code)
    result.fluid.xHI = np.full(count, xHI_dimensionless)
    result.fluid.mu = np.full(count, mu_dimensionless)
    result.fluid.time_proper_code = initial_time_proper_code
    result.fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    result.fluid.SetPressure()
    result.fluid.SetEnergyDensity()
    result.mesh.geometry_state = MeshGeometryState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        x_proper_code=result.mesh.x_proper_code,
        boundary_proper_code=result.mesh.boundary_proper_code,
        width_proper_code=np.diff(result.mesh.boundary_proper_code),
        area_proper_code=result.mesh.area_proper_code,
        volume_proper_code=result.mesh.volume_proper_code,
    )
    result.fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        rho_proper_code=result.fluid.rho_proper_code,
        vel_proper_code=result.fluid.vel_proper_code,
        pre_proper_code=result.fluid.pre_proper_code,
        temp_proper_code=result.fluid.temp_proper_code,
        time_proper_code=result.fluid.time_proper_code,
        mu_dimensionless=result.fluid.mu,
        xHI_dimensionless=result.fluid.xHI,
    )
    return result


def analytic_compton_temperature(
    cosmic_times_s,
    temperature_proper_cgs_K,
    time_cosmic_cgs_s,
    cosmology,
    time_unit_s,
    hydrogen_number_density_cgs_cm3,
    hydrogen_mass_fraction,
    xHI,
    gamma,
    cmb_temperature_0_cgs_K,
    mu,
):
    """Return the EdS Compton-only solution using the linear ODE integral."""
    from scipy.integrate import solve_ivp

    from radhydropy.thermo_networks.compton import cmb_compton_rate

    rho_proper_cgs_g_cm3 = (
        hydrogen_number_density_cgs_cm3 * PROTON_MASS_CGS / hydrogen_mass_fraction
    )
    ne = hydrogen_number_density_cgs_cm3 * (1.0 - xHI)
    source_slope = float(
        cmb_compton_rate(
            np.asarray([0.0]), np.asarray([ne]), enabled=True, redshift=0.0
        )[0]
    )
    # The source is C * ne * Tcmb^4 * (Tcmb - T), so the EdS factor is
    # a^-4.  At z=0, source_slope is C * ne * T0^5.
    coefficient = source_slope / cmb_temperature_0_cgs_K
    temperature_coefficient = (
        (gamma - 1.0) * mu * PROTON_MASS_CGS / rho_proper_cgs_g_cm3
        / float(unyt.kb.to_value("erg/K"))
        * coefficient
    )

    initial_time_cosmic_cgs_s = float(time_cosmic_cgs_s) * time_unit_s
    final_time_cosmic_cgs_s = float(np.max(cosmic_times_s))

    def rhs(time_cosmic_cgs_s_value, values):
        time_proper_code = time_cosmic_cgs_s_value / time_unit_s
        scale_factor = float(cosmology.scale_factor(time_proper_code))
        cmb_temperature = cmb_temperature_0_cgs_K / scale_factor
        return [temperature_coefficient * scale_factor ** -4 * (cmb_temperature - values[0])]

    solution = solve_ivp(
        rhs,
        (initial_time_cosmic_cgs_s, final_time_cosmic_cgs_s),
        [temperature_proper_cgs_K],
        t_eval=np.asarray(cosmic_times_s, dtype=float),
        rtol=1.0e-10,
        atol=1.0e-8,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[0]
