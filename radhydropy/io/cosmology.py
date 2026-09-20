"""Cosmology-specific HDF5 headers and field metadata."""

import numpy as np
import unyt

from radhydropy.cosmology import EinsteinDeSitter, LambdaCDM
from radhydropy.cosmology.context import CosmologyContext
from radhydropy.field_metadata import field_spec
from radhydropy.io.metadata import _restore_header_attr_value


def write_cosmology_header(header, par, output_time, code_units):
    """Write the canonical cosmology metadata contract to ``Header``."""
    if not getattr(par, "cosmological_expansion", False):
        return
    cosmology_parameters = getattr(par, "cosmology", None)
    cosmology = getattr(cosmology_parameters, "model", cosmology_parameters)
    if cosmology is None:
        raise ValueError("cosmological_expansion requires par.cosmology.model")
    if getattr(par, "supercomoving_coordinates", False):
        tau = float(np.asarray(output_time, dtype=float))
        cosmic_time = float(cosmology.cosmic_time_from_supercomoving(tau))
    else:
        cosmic_time = float(np.asarray(output_time, dtype=float))
        tau = float(cosmology.supercomoving_time(cosmic_time))
    header.attrs["CosmologyType"] = cosmology.type_name
    header.attrs["CosmologyTRef"] = float(cosmology.t_ref)
    header.attrs["CosmologyARef"] = float(cosmology.a_ref)
    if cosmology.type_name == "lambda_cdm":
        header.attrs["CosmologyOmegaM"] = float(cosmology.omega_m)
        header.attrs["CosmologyOmegaLambda"] = float(cosmology.omega_lambda)
        header.attrs["CosmologyHubbleRef"] = float(cosmology._hubble_ref)
    header.attrs["CoordinateFrame"] = getattr(par, "coordinate_frame", "physical")
    header.attrs["TimeCoordinate"] = getattr(par, "time_coordinate", "cosmic")
    header.attrs["VelocityRepresentation"] = getattr(par, "velocity_representation", "physical")
    header.attrs["DensityRepresentation"] = getattr(par, "density_representation", "physical")
    header.attrs["PressureRepresentation"] = getattr(par, "pressure_representation", "physical")
    header.attrs["TemperatureRepresentation"] = getattr(
        par, "temperature_representation", "physical"
    )
    header.attrs["ScaleFactor"] = float(cosmology.scale_factor(cosmic_time))
    header.attrs["CosmicTime"] = cosmic_time
    header.attrs["time_cosmic_code"] = cosmic_time
    header.attrs["CosmicTimeUnits"] = str(code_units.time_unit)
    header.attrs["SupercomovingTime"] = tau
    header.attrs["tau_supercomoving_code"] = tau
    header.attrs["SupercomovingTimeUnits"] = str(code_units.time_unit)
    header.attrs["HubbleParameter"] = float(cosmology.hubble(cosmic_time))
    header.attrs["HubbleParameterUnits"] = str(1.0 / code_units.time_unit)
    hubble_unit_km_s_Mpc = code_units.velocity_unit.to_value(
        unyt.km / unyt.s
    ) / code_units.length_unit.to_value(unyt.Mpc)
    header.attrs["HubbleParameterKmS_Mpc"] = (
        float(cosmology.hubble(cosmic_time)) * hubble_unit_km_s_Mpc
    )
    header.attrs["Gamma"] = float(par.hydrodynamics.gamma)


def restore_cosmology_from_header(par, header, code_units):
    """Restore and validate cosmology metadata from an HDF5 ``Header``."""
    enabled = bool(getattr(par, "cosmological_expansion", False))
    cosmology_type = _restore_header_attr_value(header.attrs.get("CosmologyType", None))
    if not enabled and cosmology_type is None:
        return
    if cosmology_type not in (None, "einstein_de_sitter", "lambda_cdm"):
        raise ValueError("unsupported CosmologyType in HDF5 header: %s" % cosmology_type)
    t_ref = float(_restore_header_attr_value(header.attrs.get("CosmologyTRef", 1.0)))
    a_ref = float(_restore_header_attr_value(header.attrs.get("CosmologyARef", 1.0)))
    par.cosmological_expansion = True
    is_lcdm = cosmology_type == "lambda_cdm"
    par.cosmology_type = "lambda_cdm" if is_lcdm else "einstein_de_sitter"
    par.cosmology_t_ref = t_ref
    par.cosmology_a_ref = a_ref
    if is_lcdm:
        omega_m = float(_restore_header_attr_value(header.attrs.get("CosmologyOmegaM", 0.3)))
        omega_lambda = float(
            _restore_header_attr_value(header.attrs.get("CosmologyOmegaLambda", 0.7))
        )
        hubble_ref = float(_restore_header_attr_value(header.attrs.get("CosmologyHubbleRef", 0.0)))
        if hubble_ref <= 0.0:
            hubble_ref = None
        par.cosmology_omega_m = omega_m
        par.cosmology_omega_lambda = omega_lambda
        par.cosmology_hubble_ref = hubble_ref
        cosmology = LambdaCDM.from_code_units(
            code_units,
            t_ref=t_ref,
            a_ref=a_ref,
            omega_m=omega_m,
            omega_lambda=omega_lambda,
            hubble_ref=hubble_ref,
        )
    else:
        cosmology = EinsteinDeSitter.from_code_units(
            code_units,
            t_ref=t_ref,
            a_ref=a_ref,
        )
    if hasattr(par, "set_cosmology_model"):
        par.set_cosmology_model(cosmology)
    else:
        par.cosmology = cosmology


def restore_cosmology_context_from_header(par, header):
    """Restore the immutable snapshot conversion context from ``Header``."""
    gamma_value = header.attrs.get("Gamma")
    scale_factor_value = header.attrs.get("ScaleFactor")
    hubble_value = header.attrs.get("HubbleParameterKmS_Mpc")
    if gamma_value is None or scale_factor_value is None or hubble_value is None:
        return None
    cosmology_name = _restore_header_attr_value(header.attrs.get("CosmologyType", "proper"))
    try:
        context = CosmologyContext(
            gamma=float(_restore_header_attr_value(gamma_value)),
            cosmology=str(cosmology_name),
            isothermal=(
                getattr(getattr(par, "hydrodynamics", None), "eos_type", "") == "isothermal"
            ),
            scale_factor=float(_restore_header_attr_value(scale_factor_value)),
            hubble_parameter_km_s_Mpc=float(_restore_header_attr_value(hubble_value)),
        )
    except (TypeError, ValueError):
        return None
    par.cosmology_context = context
    return context


def runtime_field_spec(field_name, par, code_units, output_time):
    """Build metadata for a canonical runtime field at write time."""
    cosmology_parameters = getattr(par, "cosmology", None)
    cosmology = getattr(cosmology_parameters, "model", cosmology_parameters)
    if cosmology is None or not getattr(par, "cosmological_expansion", False):
        return field_spec(field_name, code_units)
    if getattr(par, "supercomoving_coordinates", False):
        cosmic_time = float(cosmology.cosmic_time_from_supercomoving(output_time))
    else:
        cosmic_time = float(output_time)
    scale_factor = float(cosmology.scale_factor(cosmic_time))
    hubble_parameter_km_s_Mpc = None
    if field_name == "vel_supercomoving_code":
        hubble_code = float(cosmology.hubble(cosmic_time))
        hubble_unit_km_s_Mpc = code_units.velocity_unit.to_value(
            unyt.km / unyt.s
        ) / code_units.length_unit.to_value(unyt.Mpc)
        hubble_parameter_km_s_Mpc = hubble_code * hubble_unit_km_s_Mpc
    return field_spec(
        field_name,
        code_units,
        cosmology=cosmology.type_name,
        scale_factor=scale_factor,
        hubble_parameter_km_s_Mpc=hubble_parameter_km_s_Mpc,
    )
