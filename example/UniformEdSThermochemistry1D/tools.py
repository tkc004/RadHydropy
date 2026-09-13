"""Initial conditions and analytic reference for the uniform EdS source test."""

import numpy as np
import unyt

from radhydropy.constants import PROTON_MASS_CGS
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits, quantity_to_value


def build_initial_condition(config):
    """Build the proper-coordinate IC through the shared writer boundary."""
    code_unit_system = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    initial_condition = config["initial_condition"]
    count = int(config["par"]["mesh"]["grid_cells"])
    writer_ic_config = dict(initial_condition)
    writer_ic_config["time_proper"] = initial_condition["time_cosmic"]
    writer = InitialConditionWriter(
        par_config=config["par"],
        code_units=code_unit_system,
        ic_config=writer_ic_config,
    )
    boundary_proper_unyt = np.linspace(
        0.0, 1.0, count + 1
    ) * (
        initial_condition["radius_outer_proper"]
        - initial_condition["radius_inner_proper"]
    ) + initial_condition["radius_inner_proper"]
    writer.box_size = writer.radquantity(initial_condition["radius_outer_proper"])
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)

    hydrogen_number_density_cgs_cm3 = float(initial_condition["hydrogen_number_density"].to_value("1/cm**3"))
    hydrogen_mass_fraction = float(initial_condition["hydrogen_mass_fraction"])
    rho_proper_cgs_g_cm3_unyt = (
        hydrogen_number_density_cgs_cm3 * PROTON_MASS_CGS / hydrogen_mass_fraction
    ) * unyt.g / unyt.cm**3
    xHI_dimensionless = float(initial_condition["xHI"])
    mu_dimensionless = 1.0 / (hydrogen_mass_fraction * (2.0 - xHI_dimensionless))
    writer.fluid.rho_radarray = writer.radarray(
        np.ones(count) * rho_proper_cgs_g_cm3_unyt
    )
    writer.fluid.vel_radarray = writer.radarray(
        np.zeros(count) * code_unit_system.velocity_unit
    )
    writer.fluid.temp_radarray = writer.radarray(
        np.ones(count) * initial_condition["temperature_proper"]
    )
    writer.fluid.xHI = np.full(count, xHI_dimensionless)
    writer.fluid.mu = np.full(count, mu_dimensionless)
    return writer


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
