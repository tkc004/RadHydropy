"""Typed code-to-cgs source-state construction for hydrogen chemistry."""

import numpy as np
import unyt

from radhydropy.arrays import as_named_array
from radhydropy.constants import (
    BOLTZMANN_CONSTANT_CGS,
    DEFAULT_EPSILON_GAMMA_CGS_ERG,
    DEFAULT_SIGMA_GAMMA_CGS_CM2,
    PROTON_MASS_CGS,
)
from radhydropy.diagnostics import thermochemistry_active_mask
from radhydropy.runtime_fields import runtime_fields
from radhydropy.state_boundaries import (
    CgsSourceState,
    ProperCodeState,
    SupercomovingCodeState,
)


def build_source_state(
    mesh,
    fluid,
    par,
    *,
    code_units,
    cgs_source_state_from_code,
    canonical_fluid_primitive_arrays,
    canonical_mesh_geometry_arrays,
    fast_source_scaling,
    optional_numeric_value,
    parameter_value,
    interior_slice,
):
    """Build the hydrogen source state at the explicit cgs boundary."""
    if code_units is None:
        raise ValueError("hydrogen thermo-chemistry requires configured code units")
    kpc_in_cm = float((1.0 * unyt.kpc).to_value(unyt.cm))
    interior = interior_slice(par)
    fields = runtime_fields(par)
    runtime = fluid.runtime_state
    xHI = as_named_array(runtime.xHI_dimensionless[interior].copy())
    gamma = getattr(
        getattr(fluid, "eos", None),
        "gamma",
        getattr(par, "gamma", 5.0 / 3.0),
    )
    scaling = fast_source_scaling(fluid, par, gamma)
    mu = 1.0 / (2.0 - np.clip(xHI, 1.0e-12, 1.0 - 1.0e-12))

    _, _, _, temp_runtime_code, _ = canonical_fluid_primitive_arrays(fluid, par)
    temperature_code = temp_runtime_code[interior]
    temperature_cgs_K = temperature_code * code_units.unit_conversion[
        "temperature_cgs_K"
    ]
    specific_energy_cgs_erg_g = (
        BOLTZMANN_CONSTANT_CGS
        * temperature_cgs_K
        / ((gamma - 1.0) * mu * PROTON_MASS_CGS)
    )
    density_code, velocity_code, _, _, _ = canonical_fluid_primitive_arrays(
        fluid, par,
    )
    state_type = (
        ProperCodeState
        if fields.time == "time_proper_code"
        else SupercomovingCodeState
    )
    state_kwargs = {
        "specific_energy_proper_code" if state_type is ProperCodeState
        else "specific_energy_supercomoving_code": (
            specific_energy_cgs_erg_g
            / code_units.unit_conversion["specific_energy_cgs_erg_g"]
        ),
        "xHI_dimensionless": xHI,
    }
    if state_type is ProperCodeState:
        state_kwargs.update(
            rho_proper_code=density_code[interior],
            vel_proper_code=velocity_code[interior],
            temp_proper_code=temperature_code,
            time_proper_code=None,
        )
    else:
        state_kwargs.update(
            rho_comoving_code=density_code[interior],
            vel_supercomoving_code=velocity_code[interior],
            temp_supercomoving_code=temperature_code,
            tau_supercomoving_code=None,
        )
    interior_code = state_type(**state_kwargs)

    _, boundary_runtime_code, _, _, volume_runtime_code = (
        canonical_mesh_geometry_arrays(mesh, par)
    )
    primitive_cgs = cgs_source_state_from_code(
        code_units=code_units,
        fluid=interior_code,
        boundary_code=boundary_runtime_code[interior.start : interior.stop + 1],
        volume_code=volume_runtime_code[interior],
    )
    source = CgsSourceState(
        boundary_cgs_cm=primitive_cgs.boundary_cgs_cm * scaling["scale_factor"],
        volume_cgs_cm3=primitive_cgs.volume_cgs_cm3 * scaling["density_factor"],
        rho_cgs_g_cm3=primitive_cgs.rho_cgs_g_cm3 / scaling["density_factor"],
        velocity_cgs_cm_s=primitive_cgs.velocity_cgs_cm_s,
        temperature_cgs_K=(
            primitive_cgs.temperature_cgs_K / scaling["temperature_factor"]
        ),
        specific_energy_cgs_erg_g=specific_energy_cgs_erg_g,
        xHI_dimensionless=xHI,
    )
    rho_physical = source.rho_cgs_g_cm3
    sigma_parameter = (
        getattr(par, "radiation_group_sigma_gamma", None)
        if getattr(par, "radiation_group_edges_eV", None) is not None
        else getattr(par, "hydrogen_sigma_gamma", None)
    )
    sigma_gamma = optional_numeric_value(
        sigma_parameter,
        code_units.area_unit,
        default=DEFAULT_SIGMA_GAMMA_CGS_CM2,
    )
    source_rate = optional_numeric_value(
        parameter_value(par, "source_photon_rate"),
        code_units.time_unit ** -1,
        default=0.0,
    )
    epsilon_parameter = (
        getattr(par, "radiation_group_epsilon_gamma", None)
        if getattr(par, "radiation_group_edges_eV", None) is not None
        else getattr(par, "hydrogen_epsilon_gamma", None)
    )
    epsilon_gamma = optional_numeric_value(
        epsilon_parameter,
        code_units.energy_unit,
        default=DEFAULT_EPSILON_GAMMA_CGS_ERG,
    )
    alpha_B = getattr(par, "hydrogen_alpha_B", None)
    if alpha_B is not None:
        alpha_B = alpha_B.to_value(code_units.volume_unit / code_units.time_unit)
    beta = getattr(par, "hydrogen_beta", None)
    if beta is not None:
        beta = beta.to_value(code_units.volume_unit / code_units.time_unit)
    radius_code, _, _, _, _ = canonical_mesh_geometry_arrays(mesh, par)
    radius_length = optional_numeric_value(radius_code[interior], code_units.length_unit)
    return {
        "interior": interior,
        "boundary_cgs_cm": source.boundary_cgs_cm,
        "width_cgs_cm": np.diff(source.boundary_cgs_cm),
        "volume_cgs_cm3": source.volume_cgs_cm3,
        "radius_cgs_cm": as_named_array(radius_length * scaling["scale_factor"]),
        "radius_kpc": np.asarray(
            radius_length * scaling["scale_factor"] / kpc_in_cm,
            dtype=float,
        ),
        "xHI": source.xHI_dimensionless,
        "temperature_cgs_K": source.temperature_cgs_K,
        "specific_energy_cgs_erg_g": source.specific_energy_cgs_erg_g,
        "rho_cgs_g_cm3": source.rho_cgs_g_cm3,
        "active": thermochemistry_active_mask(
            rho_physical, par, scaling["density_factor"],
        ),
        "nH_cgs_cm3": rho_physical
        * getattr(par, "hydrogen_mass_fraction", 1.0)
        / PROTON_MASS_CGS,
        "gamma": gamma,
        "hydrogen_mass_fraction": getattr(par, "hydrogen_mass_fraction", 1.0),
        "sigma_gamma_cgs_cm2": sigma_gamma,
        "source_rate_s": source_rate,
        "epsilon_gamma_cgs_erg": epsilon_gamma,
        "source_temperature_factor": scaling["temperature_factor"],
        "source_scale_factor": scaling["scale_factor"],
        "source_CFL": getattr(par, "hydrogen_source_CFL", 0.1),
        "dtmin_s": optional_numeric_value(
            getattr(par, "hydrogen_source_dtmin", None),
            code_units.time_unit,
            default=0.0,
        ),
        "recombination": getattr(par, "hydrogen_recombination", True),
        "collisional_ionization": getattr(
            par, "hydrogen_collisional_ionization", True,
        ),
        "thermal_coupling": getattr(par, "hydrogen_thermal_coupling", True),
        "compton_cmb_enabled": getattr(par, "compton_cmb_enabled", False),
        "compton_cmb_redshift": getattr(par, "compton_cmb_redshift", 0.0),
        "cmb_temperature_0_cgs_K": optional_numeric_value(
            getattr(par, "cmb_temperature_0", None),
            code_units.temperature_unit,
            default=2.7255 * unyt.K,
        ),
        "alpha_B_cgs_cm3_s": alpha_B,
        "beta_cgs_cm3_s": beta,
    }
