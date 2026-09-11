"""HDF5 input and output helpers for simulations."""

from pathlib import Path
import hashlib

import h5py
import os
import unyt
import numpy as np
import yaml
import radhydropy.utils as ru
from radhydropy.units import CodeUnits, code_unit_scales, code_quantity_to_cgs, _code_units
from radhydropy.arrays import as_named_array
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    PROPER_RUNTIME_FIELDS,
    SUPERCOMOVING_RUNTIME_FIELDS,
)
from radhydropy.cosmology import EinsteinDeSitter, LambdaCDM
from radhydropy.cosmology_context import CosmologyContext
from radhydropy.field_metadata import FieldSpec, field_spec
from radhydropy.radarray import RadArray
try:
    from sympy.core.basic import Basic as SympyBasic
except Exception:  # pragma: no cover - optional dependency shape
    SympyBasic = None


class SnapshotConfigurationError(ValueError):
    """Raised when a snapshot is incompatible with the supplied runtime."""


def _provenance_yaml_value(value):
    """Convert nested configuration values into YAML-safe values."""
    if isinstance(value, unyt.array.unyt_array):
        numeric_value = np.asarray(value.value)
        if numeric_value.ndim == 0:
            numeric_value = numeric_value.item()
        else:
            numeric_value = numeric_value.tolist()
        return {"value": numeric_value, "unit": str(value.units)}
    if isinstance(value, dict):
        return {
            str(key): _provenance_yaml_value(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_provenance_yaml_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _provenance_yaml_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return yaml.safe_dump(
        _provenance_yaml_value(value),
        sort_keys=True,
        default_flow_style=False,
    )


def _write_provenance(header, provenance):
    """Write reproducibility metadata under ``Header/Provenance``."""
    if provenance is None:
        return
    if not hasattr(provenance, "get"):
        raise TypeError("HDF5 provenance must be supplied as a mapping")
    source_yaml = _provenance_yaml_text(
        provenance.get("source_config_yaml", "")
    )
    effective_value = provenance.get("effective_config_yaml")
    if effective_value is None:
        effective_value = provenance.get("effective_config", {})
    effective_yaml = _provenance_yaml_text(effective_value)
    provenance_group = header.create_group("Provenance")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    provenance_group.create_dataset(
        "source_config_yaml", data=source_yaml, dtype=string_dtype
    )
    provenance_group.create_dataset(
        "effective_config_yaml", data=effective_yaml, dtype=string_dtype
    )
    provenance_group.attrs["source_config_sha256"] = hashlib.sha256(
        source_yaml.encode("utf-8")
    ).hexdigest()
    provenance_group.attrs["effective_config_sha256"] = hashlib.sha256(
        effective_yaml.encode("utf-8")
    ).hexdigest()
    for key in (
        "schema_version",
        "source_config_filename",
        "git_commit",
        "git_dirty",
        "initial_condition_sha256",
    ):
        if key in provenance and provenance[key] is not None:
            provenance_group.attrs[key] = _header_attr_value(provenance[key])


def _read_provenance(header):
    if "Provenance" not in header:
        return None
    group = header["Provenance"]
    decode = lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
    provenance = {
        "source_config_yaml": decode(group["source_config_yaml"][()]),
        "effective_config_yaml": decode(group["effective_config_yaml"][()]),
    }
    for key, value in group.attrs.items():
        restored = _restore_header_attr_value(value)
        provenance[key] = restored
    return provenance


def _code_units_from_parameter(par):
    """Return pre-existing runtime code units, if the runtime declares them."""
    code_units = getattr(par, "CodeUnits", None)
    if code_units is not None:
        return code_units
    return getattr(getattr(par, "units", None), "CodeUnits", None)


def _validate_snapshot_configuration(par, header, header_code_units):
    """Validate header compatibility before mutating the runtime objects."""
    expected_units = _code_units_from_parameter(par)
    if expected_units is not None:
        if not isinstance(expected_units, CodeUnits):
            expected_units = CodeUnits.from_mapping(expected_units)
        scales = (
            ("mass", expected_units.mass_in_cgs, header_code_units.mass_in_cgs),
            ("length", expected_units.length_in_cgs, header_code_units.length_in_cgs),
            ("velocity", expected_units.velocity_in_cgs, header_code_units.velocity_in_cgs),
            ("current", expected_units.current_in_cgs, header_code_units.current_in_cgs),
            ("temperature", expected_units.temperature_in_cgs, header_code_units.temperature_in_cgs),
        )
        for name, expected, actual in scales:
            if not np.isclose(expected, actual, rtol=1e-12, atol=0.0):
                raise SnapshotConfigurationError(
                    f"snapshot CodeUnits {name} scale ({actual}) does not "
                    f"match runtime scale ({expected})"
                )

    header_coordsys = _restore_header_attr_value(
        header.attrs.get("CoordinateSystem", None)
    )
    expected_coordsys = getattr(
        getattr(par, "simulation", None), "coordinate_system", None
    )
    if (
        header_coordsys is not None
        and expected_coordsys is not None
        and str(header_coordsys) != str(expected_coordsys)
    ):
        raise SnapshotConfigurationError(
            f"snapshot coordinate system {header_coordsys!r} does not match "
            f"runtime coordinate system {expected_coordsys!r}"
        )

    header_grid = header.attrs.get("GridCells", None)
    expected_grid = getattr(getattr(par, "mesh", None), "grid_cells", None)
    if (
        header_grid is not None
        and expected_grid is not None
        and int(_restore_header_attr_value(header_grid)) != int(expected_grid)
    ):
        raise SnapshotConfigurationError(
            f"snapshot grid size {int(_restore_header_attr_value(header_grid))} "
            f"does not match runtime grid size {int(expected_grid)}"
        )

    header_cosmology = _restore_header_attr_value(
        header.attrs.get("CosmologyType", None)
    )
    expected_expansion = getattr(par, "cosmological_expansion", None)
    if (
        header_cosmology is not None
        and expected_expansion is False
    ):
        raise SnapshotConfigurationError(
            "snapshot is cosmological but runtime has cosmological_expansion=False"
        )
    if (
        header_cosmology is None
        and expected_expansion is True
    ):
        raise SnapshotConfigurationError(
            "snapshot is non-cosmological but runtime has cosmological_expansion=True"
        )
    expected_cosmology = getattr(par, "cosmology_type", None)
    if expected_cosmology is None:
        expected_model = getattr(par, "cosmology", None)
        expected_cosmology = getattr(expected_model, "type_name", None)
    if header_cosmology is not None and expected_cosmology is not None:
        aliases = {
            "LambdaCDM": "lambda_cdm",
            "lcdm": "lambda_cdm",
            "EinsteinDeSitter": "einstein_de_sitter",
        }
        header_cosmology = aliases.get(str(header_cosmology), str(header_cosmology))
        expected_cosmology = aliases.get(str(expected_cosmology), str(expected_cosmology))
        if header_cosmology != expected_cosmology:
            raise SnapshotConfigurationError(
                f"snapshot cosmology {header_cosmology!r} does not match "
                f"runtime cosmology {expected_cosmology!r}"
            )

    for header_key, parameter_key in (
        ("CoordinateFrame", "coordinate_frame"),
        ("TimeCoordinate", "time_coordinate"),
        ("VelocityRepresentation", "velocity_representation"),
        ("DensityRepresentation", "density_representation"),
        ("PressureRepresentation", "pressure_representation"),
        ("TemperatureRepresentation", "temperature_representation"),
    ):
        header_value = _restore_header_attr_value(
            header.attrs.get(header_key, None)
        )
        expected_value = getattr(par, parameter_key, None)
        if (
            header_value is not None
            and expected_value is not None
            and str(header_value) != str(expected_value)
        ):
            raise SnapshotConfigurationError(
                f"snapshot {parameter_key} {header_value!r} does not match "
                f"runtime {parameter_key} {expected_value!r}"
            )


def _scale_unit_for_key(scale_key):
    return {
        "length_cgs_cm": unyt.cm,
        "mass_g": unyt.g,
        "velocity_cgs_cm_s": unyt.cm / unyt.s,
        "time_s": unyt.s,
        "temperature_cgs_K": unyt.K,
        "area_cgs_cm2": unyt.cm**2,
        "volume_cgs_cm3": unyt.cm**3,
        "density_cgs_g_cm3": unyt.g / unyt.cm**3,
        "pressure_cgs_erg_cm3": unyt.erg / unyt.cm**3,
        "energy_cgs_erg": unyt.erg,
        "specific_energy_cgs_erg_g": unyt.erg / unyt.g,
        "momentum_g_cgs_cm_s": unyt.g * unyt.cm / unyt.s,
        "mass_flux_g_cgs_cm2_s": unyt.g / (unyt.cm**2 * unyt.s),
        "energy_flux_cgs_erg_cm2_s": unyt.erg / (unyt.cm**2 * unyt.s),
        "number_density_cgs_cm3": 1.0 / unyt.cm**3,
        "photon_flux_per_cgs_cm2_s": 1.0 / (unyt.cm**2 * unyt.s),
        "photon_rate_per_s": 1.0 / unyt.s,
        "alpha_cgs_cm3_s": unyt.cm**3 / unyt.s,
        "acceleration_cgs_cm_s2": unyt.cm / unyt.s**2,
        "specific_angular_momentum": unyt.cm**2 / unyt.s,
        "angular_momentum": unyt.g * unyt.cm**2 / unyt.s,
    }.get(scale_key, None)


def _normalize_attr_name(name):
    """Return a safe Python attribute name for an HDF5 dataset name."""
    normalized = []
    for char in str(name):
        if char.isalnum() or char == "_":
            normalized.append(char)
        else:
            normalized.append("_")
    result = "".join(normalized).strip("_")
    return result or "field"


def _read_any_dataset(dataset, code_units=None, scale_key=None):
    """Read a dataset and normalize it into code-unit numeric arrays.

    When ``code_units`` and ``scale_key`` are provided, the stored dataset is
    interpreted in its declared unit, converted to the corresponding cgs scale
    for that physical quantity, and then divided by the runtime code-unit
    scale.
    """
    data = np.asarray(dataset[()], dtype=float)
    storage_unit = dataset.attrs.get("storage_unit", None)
    if isinstance(storage_unit, bytes):
        storage_unit = storage_unit.decode("utf-8")
    if storage_unit == "code":
        metadata = {
            key: _restore_header_attr_value(value)
            for key, value in dataset.attrs.items()
            if key not in {"units", "storage_unit"}
        }
        metadata["storage_unit"] = storage_unit
        try:
            FieldSpec.from_metadata(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Dataset {dataset.name!r} has invalid FieldSpec metadata"
            ) from exc
        return as_named_array(data)
    if storage_unit not in {None, "cgs"}:
        raise ValueError(
            f"Dataset {dataset.name!r} has unsupported storage_unit "
            f"{storage_unit!r}"
        )
    unit_name = dataset.attrs.get("units", None)
    if code_units is not None and scale_key is not None:
        scales = code_unit_scales(code_units)
        if unit_name:
            stored_unit = unyt.Unit(unit_name)
            cgs_unit = _scale_unit_for_key(scale_key)
            if cgs_unit is not None:
                data = unyt.unyt_array(data, stored_unit).to_value(cgs_unit)
        return as_named_array(data / scales[scale_key])
    if unit_name:
        raise ValueError(
            f"Cannot read dataset {dataset.name!r} with units {unit_name!r} without a code-unit mapping."
        )
    return as_named_array(data)


def _populate_group_targets(group, targets, code_units=None, scale_map=None):
    scale_map = scale_map or {}
    for name, dataset in group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        value = _read_any_dataset(
            dataset,
            code_units=code_units,
            scale_key=scale_map.get(name),
        )
        attr_name = _normalize_attr_name(name)
        for target in targets:
            setattr(target, attr_name, value)


def _radarray_field_spec(dataset, canonical_name, code_units, cosmology):
    """Restore a dataset's ``FieldSpec`` or build its canonical fallback."""
    metadata = {
        key: _restore_header_attr_value(value)
        for key, value in dataset.attrs.items()
        if key != "units"
    }
    if "storage_unit" not in metadata:
        metadata["storage_unit"] = "code"
    try:
        restored = FieldSpec.from_metadata(metadata)
        # ``boundary`` is an older physical/proper alias.  RadArray uses the
        # explicit ``proper`` representation so that conversion dispatch is
        # unambiguous on readback.
        if canonical_name == "boundary_proper_code" and restored.representation == "physical":
            raise ValueError("legacy physical boundary alias")
        return restored
    except (TypeError, ValueError):
        hubble = (
            cosmology.hubble_parameter_km_s_Mpc
            if canonical_name == "vel_supercomoving_code"
            else None
        )
        return field_spec(
            canonical_name,
            code_units,
            cosmology=cosmology.cosmology,
            scale_factor=cosmology.scale_factor,
            hubble_parameter_km_s_Mpc=hubble,
        )


def _attach_radarray_views(group, target, dataset_names, canonical_schema,
                           code_units, cosmology, allowed_names=None):
    """Expose loaded dimensional fields as neutral ``*_radarray`` views.

    The ordinary attributes populated by ``_populate_group_targets`` remain
    plain code-unit arrays for solver compatibility.  These parallel views
    are the typed, representation-aware interface for analysis and restart
    preparation, similar to SWIFTsimIO's unit-bearing data objects.
    """
    if cosmology is None:
        return
    if canonical_schema == "cosmological":
        mapping = {
            "boundary_comoving_code": ("boundary_radarray", "boundary_comoving_code"),
            "rho_comoving_code": ("rho_radarray", "rho_comoving_code"),
            "vel_supercomoving_code": ("vel_radarray", "vel_supercomoving_code"),
            "temp_supercomoving_code": ("temp_radarray", "temp_supercomoving_code"),
            "pre_supercomoving_code": ("pre_radarray", "pre_supercomoving_code"),
        }
    else:
        mapping = {
            "boundary": ("boundary_radarray", "boundary_proper_code"),
            "boundary_proper_code": ("boundary_radarray", "boundary_proper_code"),
            "boundary_comoving_code": ("boundary_radarray", "boundary_comoving_code"),
            "rho_code": ("rho_radarray", "rho_proper_code"),
            "rho_proper_code": ("rho_radarray", "rho_proper_code"),
            "vel_code": ("vel_radarray", "vel_proper_code"),
            "vel_proper_code": ("vel_radarray", "vel_proper_code"),
            "temp_code": ("temp_radarray", "temp_proper_code"),
            "temp_proper_code": ("temp_radarray", "temp_proper_code"),
            "pre_proper_code": ("pre_radarray", "pre_proper_code"),
        }
    for dataset_name, (view_name, canonical_name) in mapping.items():
        if allowed_names is not None and dataset_name not in allowed_names:
            continue
        if dataset_name not in dataset_names:
            continue
        attr_name = _normalize_attr_name(dataset_name)
        if not hasattr(target, attr_name):
            continue
        spec = _radarray_field_spec(
            dataset_names[dataset_name], canonical_name, code_units, cosmology
        )
        setattr(
            target,
            view_name,
            RadArray(
                np.asarray(getattr(target, attr_name), dtype=float),
                code_units=code_units,
                field_spec=spec,
                cosmology=cosmology,
            ),
        )
    # Keep auxiliary dimensional fields discoverable without inventing
    # semantic aliases.  Unknown/dimensionless datasets (mu, xHI, etc.) are
    # intentionally left as ordinary numeric arrays.
    for dataset_name, dataset in dataset_names.items():
        if dataset_name in mapping or (
            allowed_names is not None and dataset_name not in allowed_names
        ):
            continue
        attr_name = _normalize_attr_name(dataset_name)
        if not hasattr(target, attr_name):
            continue
        try:
            spec = _radarray_field_spec(
                dataset, dataset_name, code_units, cosmology
            )
        except (TypeError, ValueError):
            continue
        radarray_name = attr_name
        if radarray_name.endswith("_code"):
            radarray_name = radarray_name[:-5]
        setattr(
            target,
            f"{radarray_name}_radarray",
            RadArray(
                np.asarray(getattr(target, attr_name), dtype=float),
                code_units=code_units,
                field_spec=spec,
                cosmology=cosmology,
            ),
        )


def _yaml_config_value(value):
    """Convert a value to a YAML config friendly representation."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, unyt.unit_object.Unit):
        return str(value)
    if SympyBasic is not None and isinstance(value, SympyBasic):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _yaml_config_value(value.to_dict())
    if hasattr(value, "units"):
        raw_value = np.asarray(value.to_value(value.units))
        if raw_value.shape == () or raw_value.size == 1:
            return {
                "value": float(raw_value.reshape(-1)[0]),
                "unit": str(value.units),
            }
        return {
            "value": raw_value.tolist(),
            "unit": str(value.units),
        }
    if isinstance(value, dict):
        return {str(key): _yaml_config_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_config_value(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.shape == () or value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if callable(value):
        return getattr(value, "__name__", value.__class__.__name__)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {
            key: _yaml_config_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    if isinstance(value, Path):
        return str(value)
    return value


def parameter_tree(value):
    """Convert an arbitrary value into a YAML-safe, human-readable object."""
    return _yaml_config_value(value)


def _write_quantity(
    group, name, value, code_units=None, scale_key=None, default_unit=None,
    metadata=None, field_spec_obj=None,
):
    def _unit_label(unit_obj):
        return str(getattr(unit_obj, "units", unit_obj))

    if code_units is None and scale_key is not None:
        raise ValueError(f"{name} requires code_units for HDF5 serialization")

    storage_unit = (
        field_spec_obj.storage_unit if field_spec_obj is not None else "cgs"
    )
    if field_spec_obj is not None and storage_unit == "code":
        if code_units is None or scale_key is None:
            raise ValueError(
                f"{name} requires code_units and scale_key for code storage"
            )
        code_unit = _scale_unit_for_key(scale_key)
        if code_unit is None:
            raise ValueError(f"no code-unit scale is defined for {scale_key!r}")
        if hasattr(value, "to_value"):
            data = np.asarray(value.to_value(code_unit))
        else:
            data = np.asarray(value, dtype=float)
        unit = str(code_unit)
    elif hasattr(value, "to_value"):
        if code_units is not None and scale_key is not None:
            unit_obj = _scale_unit_for_key(scale_key)
            if unit_obj is None:
                unit_obj = value.units
            data = np.asarray(value.to_value(unit_obj))
            unit = _unit_label(unit_obj)
        else:
            data = np.asarray(value.to_value(value.units))
            unit = str(value.units)
    elif code_units is not None and scale_key is not None:
        unit_obj = _scale_unit_for_key(scale_key)
        if unit_obj is None:
            unit_obj = unyt.Unit(default_unit) if default_unit is not None else None
        data = code_quantity_to_cgs(value, code_units, scale_key)
        unit = _unit_label(unit_obj) if unit_obj is not None else "dimensionless"
    else:
        data = np.asarray(value)
        unit = str(unyt.Unit(default_unit)) if default_unit is not None else "dimensionless"
    dataset = group.create_dataset(name, data=data)
    dataset.attrs["units"] = unit
    dataset.attrs["storage_unit"] = storage_unit
    if field_spec_obj is not None:
        for key, metadata_value in field_spec_obj.to_metadata().items():
            if metadata_value is not None:
                dataset.attrs[key] = metadata_value
    for key, metadata_value in (metadata or {}).items():
        dataset.attrs[key] = metadata_value
    return dataset


def _write_cosmology_header(header, par, output_time, code_units):
    """Write the canonical cosmology metadata contract to ``Header``."""
    if not getattr(par, "cosmological_expansion", False):
        return
    cosmology = getattr(par, "cosmology", None)
    if cosmology is None:
        raise ValueError("cosmological_expansion requires par.cosmology")
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
    header.attrs["TemperatureRepresentation"] = getattr(par, "temperature_representation", "physical")
    header.attrs["ScaleFactor"] = float(cosmology.scale_factor(cosmic_time))
    header.attrs["CosmicTime"] = cosmic_time
    header.attrs["time_cosmic_code"] = cosmic_time
    header.attrs["CosmicTimeUnits"] = str(code_units.time_unit)
    header.attrs["SupercomovingTime"] = tau
    header.attrs["tau_supercomoving_code"] = tau
    header.attrs["SupercomovingTimeUnits"] = str(code_units.time_unit)
    header.attrs["HubbleParameter"] = float(cosmology.hubble(cosmic_time))
    header.attrs["HubbleParameterUnits"] = str(1.0 / code_units.time_unit)
    hubble_unit_km_s_Mpc = (
        code_units.velocity_unit.to_value(unyt.km / unyt.s)
        / code_units.length_unit.to_value(unyt.Mpc)
    )
    header.attrs["HubbleParameterKmS_Mpc"] = (
        float(cosmology.hubble(cosmic_time)) * hubble_unit_km_s_Mpc
    )
    header.attrs["Gamma"] = float(par.hydrodynamics.gamma)


def _restore_cosmology_from_header(par, header, code_units):
    """Restore and validate cosmology metadata from an HDF5 ``Header``."""
    enabled = bool(getattr(par, "cosmological_expansion", False))
    cosmology_type = _restore_header_attr_value(header.attrs.get("CosmologyType", None))
    if not enabled and cosmology_type is None:
        return
    if cosmology_type not in (
        None, "einstein_de_sitter", "EinsteinDeSitter",
        "lambda_cdm", "LambdaCDM", "lcdm",
    ):
        raise ValueError("unsupported CosmologyType in HDF5 header: %s" % cosmology_type)
    t_ref = float(_restore_header_attr_value(header.attrs.get("CosmologyTRef", 1.0)))
    a_ref = float(_restore_header_attr_value(header.attrs.get("CosmologyARef", 1.0)))
    par.cosmological_expansion = True
    is_lcdm = cosmology_type in ("lambda_cdm", "LambdaCDM", "lcdm")
    par.cosmology_type = "lambda_cdm" if is_lcdm else "einstein_de_sitter"
    par.cosmology_t_ref = t_ref
    par.cosmology_a_ref = a_ref
    if is_lcdm:
        omega_m = float(_restore_header_attr_value(
            header.attrs.get("CosmologyOmegaM", 0.3)))
        omega_lambda = float(_restore_header_attr_value(
            header.attrs.get("CosmologyOmegaLambda", 0.7)))
        hubble_ref = float(_restore_header_attr_value(
            header.attrs.get("CosmologyHubbleRef", 0.0)))
        if hubble_ref <= 0.0:
            hubble_ref = None
        par.cosmology_omega_m = omega_m
        par.cosmology_omega_lambda = omega_lambda
        par.cosmology_hubble_ref = hubble_ref
        cosmology = LambdaCDM.from_code_units(
            code_units, t_ref=t_ref, a_ref=a_ref,
            omega_m=omega_m, omega_lambda=omega_lambda,
            hubble_ref=hubble_ref,
        )
    else:
        cosmology = EinsteinDeSitter.from_code_units(
            code_units, t_ref=t_ref, a_ref=a_ref
        )
    if hasattr(par, "set_cosmology_model"):
        par.set_cosmology_model(cosmology)
    else:
        par.cosmology = cosmology


def _restore_cosmology_context_from_header(par, header):
    """Restore the immutable snapshot conversion context from ``Header``."""
    gamma_value = header.attrs.get("Gamma")
    scale_factor_value = header.attrs.get("ScaleFactor")
    hubble_value = header.attrs.get("HubbleParameterKmS_Mpc")
    if gamma_value is None or scale_factor_value is None or hubble_value is None:
        return None
    cosmology_name = _restore_header_attr_value(
        header.attrs.get("CosmologyType", "proper")
    )
    try:
        context = CosmologyContext(
            gamma=float(_restore_header_attr_value(gamma_value)),
            cosmology=str(cosmology_name),
            scale_factor=float(_restore_header_attr_value(scale_factor_value)),
            hubble_parameter_km_s_Mpc=float(
                _restore_header_attr_value(hubble_value)
            ),
        )
    except (TypeError, ValueError):
        # Older examples may record an isothermal gamma=1.0.  That is a
        # valid solver setting, but not a valid adiabatic conversion context.
        return None
    par.cosmology_context = context
    return context


def _runtime_field_spec(field_name, par, code_units, output_time):
    """Build metadata for a canonical runtime field at write time."""
    cosmology = getattr(par, "cosmology", None)
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
        hubble_unit_km_s_Mpc = (
            code_units.velocity_unit.to_value(unyt.km / unyt.s)
            / code_units.length_unit.to_value(unyt.Mpc)
        )
        hubble_parameter_km_s_Mpc = hubble_code * hubble_unit_km_s_Mpc
    return field_spec(
        field_name,
        code_units,
        cosmology=cosmology.type_name,
        scale_factor=scale_factor,
        hubble_parameter_km_s_Mpc=hubble_parameter_km_s_Mpc,
    )


def _used_parameters_payload(par_config=None, initial_condition=None, existing=None):
    payload = {}
    if isinstance(existing, dict):
        payload.update(existing)
    if par_config is not None:
        payload["par"] = {
            key: _yaml_config_value(value)
            for key, value in sorted(par_config.items())
            if not str(key).startswith("_")
        }
    elif "par" not in payload:
        payload["par"] = {}
    if initial_condition is not None:
        payload["initial_condition"] = {
            key: _yaml_config_value(value)
            for key, value in sorted(initial_condition.items())
            if not str(key).startswith("_")
        }
    elif "initial_condition" not in payload:
        payload["initial_condition"] = {}
    return payload


def update_used_parameters_yaml(path, par_config=None, initial_condition=None):
    """Create or update a config-style ``used_parameters.yaml`` file."""
    path = Path(path)
    existing = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
        except yaml.YAMLError:
            loaded = None
        if isinstance(loaded, dict):
            existing = loaded
    payload = _used_parameters_payload(
        par_config=par_config,
        initial_condition=initial_condition,
        existing=existing,
    )
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    return path


def write_used_parameters(path, par):
    """Write the active runtime parameters to a YAML file."""
    path = Path(path)
    nested_par_config = getattr(par, "nested_par_config", None)
    if isinstance(nested_par_config, dict):
        runtime_parameters = {
            group: _yaml_config_value(nested_par_config.get(group, {}))
            for group in (
                "simulation", "mesh", "hydrodynamics", "boundary", "timestep",
                "output", "diagnostics", "units", "thermochemistry", "chemistry",
                "gravity", "dark_matter", "radiation",
            )
        }
        runtime_parameters.update(
            {
                key: _yaml_config_value(value)
                for key, value in nested_par_config.items()
                if key not in runtime_parameters
            }
        )
    else:
        # Retain serialization for programmatically constructed legacy
        # namespaces, while all YAML-driven ``Par`` instances use the nested
        # configuration preserved by ``Par``.
        runtime_parameters = {
            key: parameter_tree(value)
            for key, value in sorted(vars(par).items())
            if not key.startswith("_")
            and key not in {"par_config", "initial_condition"}
        }
    payload = {
        "par": runtime_parameters,
        "initial_condition": parameter_tree(
            getattr(par, "initial_condition", None)
        ),
        "example": {},
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    return path


def _header_attr_value(value):
    """Convert a runtime parameter into an HDF5-attribute-friendly value."""
    tree = parameter_tree(value)
    if tree is None:
        return yaml.safe_dump(None, sort_keys=True, default_flow_style=False)
    if isinstance(tree, (str, bytes, int, float, bool)):
        return tree
    if isinstance(tree, np.generic):
        return tree.item()
    if isinstance(tree, np.ndarray):
        if tree.dtype == object or tree.dtype.kind == "U":
            return yaml.safe_dump(tree.tolist(), sort_keys=True, default_flow_style=False)
        return tree
    if isinstance(tree, (list, tuple)) and all(
        isinstance(item, (str, bytes, int, float, bool, np.generic))
        for item in tree
    ):
        array = np.asarray(tree)
        if array.dtype == object or array.dtype.kind == "U":
            return yaml.safe_dump(tree, sort_keys=True, default_flow_style=False)
        return array
    return yaml.safe_dump(tree, sort_keys=True, default_flow_style=False)


def _restore_header_attr_value(value):
    """Convert a stored HDF5 header attribute back into a Python value."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _restore_header_attr_value(value.item())
        return np.asarray([_restore_header_attr_value(item) for item in value.tolist()])
    if isinstance(value, str):
        try:
            loaded = yaml.safe_load(value)
            if isinstance(loaded, str):
                return loaded
            return _restore_header_attr_value(loaded)
        except yaml.YAMLError:
            return value
    if isinstance(value, dict):
        if {'value', 'unit'} <= value.keys():
            restored_value = _restore_header_attr_value(value['value'])
            unit = unyt.Unit(value['unit'])
            if isinstance(restored_value, list):
                restored_value = np.asarray(restored_value)
            return np.asarray(restored_value) * unit
        return {key: _restore_header_attr_value(item) for key, item in value.items()}
    return value


def load_output_time_list(filename):
    """Load explicit output times from a text file."""
    if not filename:
        return None

    outputtimepath = Path(filename)
    if not outputtimepath.exists():
        raise FileNotFoundError(f"Output-time file not found: {outputtimepath}")

    unit = None
    output_times = []
    with outputtimepath.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            if unit is None:
                unit = tokens[0]
                for token in tokens[1:]:
                    output_times.append(float(token))
                continue
            for token in tokens:
                output_times.append(float(token))

    if unit is None:
        raise ValueError(f"Output-time file is empty: {outputtimepath}")

    return np.asarray(output_times, dtype=float) * unyt.Unit(unit)


def write_numbered_hdf5(sim, outindex):
    from .output import write_numbered_hdf5 as implementation

    return implementation(sim, outindex)


def hdf5_output_callback(sim, outputtime=0, output_state=None):
    from .output import hdf5_output_callback as implementation

    return implementation(sim, outputtime, output_state)


def run_with_output_times(
    sim,
    outputtime=0,
    mode="hydro_sources",
    advect_chemistry=True,
    stop_condition=None,
    step_backend=None,
    step_backend_kwargs=None,
):
    from .output import run_with_output_times as implementation

    return implementation(
        sim,
        outputtime=outputtime,
        mode=mode,
        advect_chemistry=advect_chemistry,
        stop_condition=stop_condition,
        step_backend=step_backend,
        step_backend_kwargs=step_backend_kwargs,
        output_writer=write_numbered_hdf5,
    )

def _writehdf5(ric, ICfilename, *, provenance=None):
    """Write simulation state to a RadHydropy HDF5 file.

    The output file contains a ``Header`` group for metadata and a ``Data``
    group for mesh and fluid arrays. Units are stored as HDF5 attributes.
    """
    ICfilename = str(ICfilename)
    print(f"--- writing {ICfilename} --- ")
    cosmological_schema = bool(
        getattr(ric.par, "cosmological_expansion", False)
        and getattr(ric.par, "supercomoving_coordinates", False)
    )
    geometry = ric.mesh.geometry_state
    runtime = ric.fluid.runtime_state
    if cosmological_schema:
        output_time = runtime.tau_supercomoving_code
        boundary_runtime_code = geometry.boundary_comoving_code
        density_runtime_code = runtime.rho_comoving_code
        velocity_runtime_code = runtime.vel_supercomoving_code
        temperature_runtime_code = runtime.temp_supercomoving_code
    else:
        output_time = runtime.time_proper_code
        boundary_runtime_code = geometry.boundary_proper_code
        density_runtime_code = runtime.rho_proper_code
        velocity_runtime_code = runtime.vel_proper_code
        temperature_runtime_code = runtime.temp_proper_code
    with h5py.File(ICfilename, 'w') as fic:
        code_units = getattr(getattr(ric.par, "units", None), "CodeUnits", None)
        # saving initial condition
        # first, save header:
        header = fic.create_group("Header")
        for key, value in sorted(vars(ric.par).items()):
            if key.startswith("_") or key in {
                "dark_matter", "dark_matter_snapshot", "cosmology",
                "hydrodynamics", "boundary", "timestep", "thermochemistry",
                "gravity", "output", "simulation", "diagnostics", "mesh",
                "chemistry", "angular_momentum", "dark_matter_config",
                "dual_energy_config", "positivity", "radiation", "units",
            }:
                continue
            if cosmological_schema and key in {"time_code", "box_size_code"}:
                continue
            if cosmological_schema is False and key in {
                "tau_supercomoving_code",
                "time_cosmic_code",
            }:
                continue
            header.attrs[key] = _header_attr_value(value)
        if code_units is not None:
            header.attrs["CodeUnits"] = _header_attr_value(code_units)
        gamma = getattr(getattr(ric.par, "hydrodynamics", None), "gamma", None)
        if gamma is not None:
            header.attrs["Gamma"] = float(gamma)
        if not cosmological_schema:
            header.attrs["ScaleFactor"] = 1.0
            header.attrs["HubbleParameterKmS_Mpc"] = 0.0
        header.attrs["GridCells"] = int(ric.par.mesh.grid_cells)
        header.attrs["GhostCells"] = int(ric.par.mesh.ghost_cells)
        header.attrs["CoordinateSystem"] = getattr(
            getattr(ric.par, "simulation", None), "coordinate_system", "cartesian"
        )
        _write_provenance(
            header,
            provenance
            if provenance is not None
            else getattr(ric, "provenance", None)
            or getattr(ric.par, "provenance", None),
        )
        if hasattr(ric, "cumulative_hydro_boundary_energy"):
            header.attrs["CumulativeHydroBoundaryEnergyCode"] = float(
                ric.cumulative_hydro_boundary_energy
            )
        if hasattr(ric, "cumulative_gravity_work"):
            header.attrs["CumulativeGravityWorkCode"] = float(
                ric.cumulative_gravity_work
            )
        _write_cosmology_header(header, ric.par, output_time, code_units)
        _write_quantity(
            header,
            "tau_supercomoving_code" if cosmological_schema else "time_proper_code",
            output_time,
            code_units=code_units,
            scale_key="time_s",
            default_unit=unyt.s,
        )
        _write_quantity(
            header,
            "box_size_comoving_code" if cosmological_schema else "box_size_proper_code",
            (
                ric.par.simulation.box_size_comoving_code
                if cosmological_schema
                else ric.par.simulation.box_size_proper_code
            ),
            code_units=code_units,
            scale_key="length_cgs_cm",
            default_unit=unyt.cm,
            metadata={
                "quantity": "radius",
                "coordinate_frame": getattr(ric.par, "coordinate_frame", "physical"),
                "representation": getattr(ric.par, "coordinate_frame", "physical"),
                "physical_relation": (
                    "physical = a * stored"
                    if getattr(ric.par, "supercomoving_coordinates", False)
                    else "physical = stored"
                ),
            },
        )

        #second, save mesh and fluid data:
        gdata = fic.create_group("Data")
        _write_quantity(
            gdata,
            "boundary_comoving_code" if cosmological_schema else "boundary_proper_code",
            boundary_runtime_code,
            code_units=code_units,
            scale_key="length_cgs_cm",
            default_unit=unyt.cm,
            metadata={
                "quantity": "radius",
                "coordinate_frame": getattr(ric.par, "coordinate_frame", "physical"),
                "representation": getattr(ric.par, "coordinate_frame", "physical"),
                "physical_relation": (
                    "physical = a * stored"
                    if getattr(ric.par, "supercomoving_coordinates", False)
                    else "physical = stored"
                ),
            },
            field_spec_obj=_runtime_field_spec(
                "boundary_comoving_code" if cosmological_schema else "boundary_proper_code",
                ric.par,
                code_units,
                output_time,
            ),
        )
        _write_quantity(
            gdata,
            "rho_comoving_code" if cosmological_schema else "rho_proper_code",
            density_runtime_code,
            code_units=code_units,
            scale_key="density_cgs_g_cm3",
            default_unit=unyt.g / unyt.cm**3,
            metadata={
                "quantity": "mass_density",
                "representation": getattr(ric.par, "density_representation", "physical"),
                "scale_factor_power": 3.0 if getattr(ric.par, "supercomoving_coordinates", False) else 0.0,
                "physical_relation": (
                    "physical = stored / a**3"
                    if getattr(ric.par, "supercomoving_coordinates", False)
                    else "physical = stored"
                ),
            },
            field_spec_obj=_runtime_field_spec(
                "rho_comoving_code" if cosmological_schema else "rho_proper_code",
                ric.par,
                code_units,
                output_time,
            ),
        )
        _write_quantity(
            gdata,
            "vel_supercomoving_code" if cosmological_schema else "vel_proper_code",
            velocity_runtime_code,
            code_units=code_units,
            scale_key="velocity_cgs_cm_s",
            default_unit=unyt.cm / unyt.s,
            metadata={
                "quantity": "velocity",
                "representation": getattr(ric.par, "velocity_representation", "physical"),
                "physical_relation": (
                    "physical = H*a*x + stored/a"
                    if getattr(ric.par, "supercomoving_coordinates", False)
                    else "physical = stored"
                ),
            },
            field_spec_obj=_runtime_field_spec(
                "vel_supercomoving_code" if cosmological_schema else "vel_proper_code",
                ric.par,
                code_units,
                output_time,
            ),
        )
        _write_quantity(
            gdata,
            "temp_supercomoving_code" if cosmological_schema else "temp_proper_code",
            temperature_runtime_code,
            code_units=code_units,
            scale_key="temperature_cgs_K",
            default_unit=unyt.K,
            metadata={
                "quantity": "temperature",
                "representation": (
                    getattr(ric.par, "temperature_representation", "physical")
                    if getattr(ric.par, "supercomoving_coordinates", False)
                    else "physical"
                ),
                "scale_factor_power": (
                    3.0 * (ric.par.hydrodynamics.gamma - 1.0)
                    if getattr(ric.par, "supercomoving_coordinates", False)
                    else 0.0
                ),
            },
            field_spec_obj=_runtime_field_spec(
                "temp_supercomoving_code" if cosmological_schema else "temp_proper_code",
                ric.par,
                code_units,
                output_time,
            ),
        )
        if hasattr(ric.fluid, "specific_angular_momentum_code"):
            _write_quantity(
                gdata,
                "specific_angular_momentum_code",
                ric.fluid.specific_angular_momentum_code,
                code_units=code_units,
                scale_key="specific_angular_momentum",
                default_unit=unyt.cm**2 / unyt.s,
            )
        for attr, dataset_name in (
            ("Mass_code", "Mass_code"),
            ("Energy_code", "Energy_code"),
            ("InternalEnergy_code", "InternalEnergy_code"),
            ("AngularMomentum_code", "AngularMomentum_code"),
            ("GravitationalPotentialEnergy_code", "GravitationalPotentialEnergy_code"),
        ):
            if hasattr(ric.fluid, attr):
                scale_key = (
                    "mass_g" if attr == "Mass_code"
                    else "angular_momentum" if attr == "AngularMomentum_code"
                    else "energy_cgs_erg"
                )
                _write_quantity(
                    gdata,
                    dataset_name,
                    getattr(ric.fluid, attr),
                    code_units=code_units,
                    scale_key=scale_key,
                    default_unit=(
                        unyt.g if attr == "Mass_code"
                        else unyt.g * unyt.cm**2 / unyt.s
                        if attr == "AngularMomentum_code" else unyt.erg
                    ),
                )
        gdata.create_dataset("mu", data=np.asarray(ric.fluid.mu))
        if hasattr(ric.fluid, "xHI"):
            gdata.create_dataset("xHI", data=np.asarray(ric.fluid.xHI))
        for attr, dataset in (("xHeI", "xHeI"), ("xHeII", "xHeII"), ("xHeIII", "xHeIII")):
            if hasattr(ric.fluid, attr):
                gdata.create_dataset(dataset, data=np.asarray(getattr(ric.fluid, attr)))
        if hasattr(ric.fluid, "ngamma_code"):
            ngamma_cgs_cm3 = ric.fluid.ngamma_code
            # Runtime fluid fields are stored as code-unit arrays.  Some
            # chemistry paths may temporarily attach units to ngamma_cgs_cm3; strip
            # those units in the configured code system before the generic
            # serializer converts the field to cgs for HDF5.
            if hasattr(ngamma_cgs_cm3, "to_value") and code_units is not None:
                ngamma_cgs_cm3 = np.asarray(ngamma_cgs_cm3.to_value(code_units.number_density_unit))
            _write_quantity(
                gdata,
                "ngamma_code",
                ngamma_cgs_cm3,
                code_units=code_units,
                scale_key="number_density_cgs_cm3",
                default_unit=1.0 / unyt.cm**3,
            )
        if getattr(ric.par, "cosmological_expansion", False):
            for dataset_name, dataset in gdata.items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                if dataset_name in {
                    "boundary", "rho_code", "vel_code", "temp_code",
                    "boundary_comoving_code", "rho_comoving_code",
                    "vel_supercomoving_code", "temp_supercomoving_code",
                }:
                    continue
                dataset.attrs["representation"] = "physical"
        dark_matter = getattr(ric.par, "dark_matter", None)
        if dark_matter is None:
            gravity = getattr(ric.par, "gravity", None)
            dark_matter = getattr(gravity, "dark_matter", None)
        if dark_matter is not None:
            dmdata = fic.create_group("DarkMatter")
            _write_quantity(dmdata, "Radius", dark_matter.radius,
                            code_units=code_units, scale_key="length_cgs_cm",
                            default_unit=unyt.cm)
            _write_quantity(dmdata, "RadialVelocity", dark_matter.velocity,
                            code_units=code_units, scale_key="velocity_cgs_cm_s",
                            default_unit=unyt.cm / unyt.s)
            _write_quantity(dmdata, "Mass", dark_matter.mass,
                            code_units=code_units, scale_key="mass_g",
                            default_unit=unyt.g)
            _write_quantity(dmdata, "SpecificAngularMomentum", dark_matter.angular_momentum,
                            code_units=code_units, scale_key="specific_angular_momentum",
                            default_unit=unyt.cm**2 / unyt.s)
            dmdata.attrs["Softening"] = _header_attr_value(
                dark_matter.softening * code_units.length_unit
            )

    if (
        not hasattr(ric, "solver")
        and Path(ICfilename).stem.lower() == "initialcondition"
    ):
        update_used_parameters_yaml(
            Path.cwd() / "used_parameters.yaml",
            initial_condition=vars(ric.par),
        )

def writehdf5(ric, ICfilename, *, provenance=None):
    """Write an initial-condition or snapshot HDF5 file.

    Representation-aware IC values are prepared by the shared
    :class:`InitialConditionWriter` boundary before serialization.
    """
    from radhydropy.initial_condition_writer import InitialConditionWriter

    return InitialConditionWriter.from_rsim(
        ric,
        provenance=provenance,
    ).write(ICfilename)


def readhdf5(par, mesh, fluid, ICfilename): 
    """Read a RadHydropy HDF5 file into parameter, mesh, and fluid objects.

    Canonical ``*_code`` datasets are restored into the runtime code-unit
    system when ``CodeUnits`` is available in the file header, so fields such
    as ``fluid.rho_code`` come back as plain numeric arrays in code units.
    """
    ICfilename = str(ICfilename)
    print(f"--- reading {ICfilename} --- ")
    with h5py.File(ICfilename, 'r') as fic:
        expected_coordsys = par.simulation.coordinate_system
        expected_nogrid = getattr(
            getattr(par, 'mesh', None), 'grid_cells', None
        )
        # saving initial condition
        # first, save header:
        header = fic["Header"]
        if "CodeUnits" not in header.attrs:
            raise ValueError(
                "IC file is missing Header.attrs['CodeUnits']; cannot read datasets without a code-unit mapping."
            )
        code_units = _restore_header_attr_value(header.attrs["CodeUnits"])
        if isinstance(code_units, dict):
            code_units = CodeUnits.from_mapping(code_units)
        if not isinstance(code_units, CodeUnits):
            raise ValueError(
                "IC file Header.attrs['CodeUnits'] is not a valid CodeUnits mapping."
            )
        _validate_snapshot_configuration(par, header, code_units)
        file_provenance = _read_provenance(header)
        if file_provenance is not None:
            par.provenance = file_provenance
        for key, value in header.attrs.items():
            restored = _restore_header_attr_value(value)
            if key == "CodeUnits":
                if isinstance(restored, dict):
                    restored = CodeUnits.from_mapping(restored)
                if hasattr(par, "set_code_units"):
                    par.set_code_units(restored)
                else:
                    setattr(par, "CodeUnits", restored)
                continue
            setattr(par, key, restored)
        _restore_cosmology_from_header(par, header, code_units)
        _restore_cosmology_context_from_header(par, header)
        if hasattr(par, 'mesh'):
            if 'GridCells' in header.attrs:
                par.mesh.grid_cells = int(header.attrs['GridCells'])
            if 'GhostCells' in header.attrs:
                par.mesh.ghost_cells = int(header.attrs['GhostCells'])
        if hasattr(par, 'simulation') and 'CoordinateSystem' in header.attrs:
            par.simulation.coordinate_system = header.attrs['CoordinateSystem']
        coordinate_system = par.simulation.coordinate_system
        if expected_coordsys is not None and coordinate_system != expected_coordsys:
            raise Exception(
                "Coordinate systems in IC (%s) and run (%s) do not agree!"
                % (coordinate_system, expected_coordsys)
            )
        grid_cells = par.mesh.grid_cells
        if expected_nogrid is not None and grid_cells != expected_nogrid:
            raise Exception(
                "Number of grids in IC (%s) and run (%s) do not agree!"
                % (grid_cells, expected_nogrid)
            )
        # The canonical representation is encoded by the typed header
        # datasets.  Parameter attributes may contain stale fields from an
        # input namespace, so they must not decide the restart schema.
        canonical_cosmological_schema = (
            "tau_supercomoving_code" in header
            or "box_size_comoving_code" in header
        )
        canonical_proper_schema = (
            "time_proper_code" in header
            or "box_size_proper_code" in header
        )
        if canonical_cosmological_schema and canonical_proper_schema:
            raise ValueError("HDF5 header mixes cosmological and proper schemas")
        if canonical_cosmological_schema and (
            "time_code" in header or "box_size_code" in header
        ):
            raise ValueError("cosmological HDF5 headers must not use generic time/box names")
        header_scale_map = {
            (
                "tau_supercomoving_code"
                if canonical_cosmological_schema
                else "time_proper_code" if canonical_proper_schema else "time_code"
            ): "time_s",
            (
                "box_size_comoving_code"
                if canonical_cosmological_schema
                else "box_size_proper_code" if canonical_proper_schema else "box_size_code"
            ): "length_cgs_cm",
        }
        _populate_group_targets(
            header,
            (par,),
            code_units=code_units,
            scale_map=header_scale_map,
        )
        metadata_fields = {
            "CoordinateFrame": "coordinate_frame",
            "TimeCoordinate": "time_coordinate",
            "VelocityRepresentation": "velocity_representation",
            "DensityRepresentation": "density_representation",
            "PressureRepresentation": "pressure_representation",
            "TemperatureRepresentation": "temperature_representation",
        }
        for header_name, parameter_name in metadata_fields.items():
            if header_name in header.attrs:
                setattr(par, parameter_name, _restore_header_attr_value(header.attrs[header_name]))
        if canonical_cosmological_schema:
            par.tau_supercomoving_code = np.asarray(
                getattr(par, "tau_supercomoving_code"), dtype=float
            )
            cosmic_time = header.attrs.get("time_cosmic_code", header.attrs.get("CosmicTime"))
            if cosmic_time is not None:
                par.time_cosmic_code = float(_restore_header_attr_value(cosmic_time))
        if hasattr(par, "_sync_simulation_parameters"):
            par._sync_simulation_parameters()
        if hasattr(par, "_sync_mesh_parameters"):
            par._sync_mesh_parameters()
        if hasattr(par, 'load_radiation_spectrum'):
            par.load_radiation_spectrum(
                par.output.directory
            )
        if hasattr(par, 'simulation'):
            time_field = (
                "tau_supercomoving_code"
                if canonical_cosmological_schema
                else "time_proper_code" if canonical_proper_schema else "time_code"
            )
            box_field = (
                "box_size_comoving_code"
                if canonical_cosmological_schema
                else "box_size_proper_code" if canonical_proper_schema else "box_size_code"
            )
            runtime_time = getattr(par, time_field)
            if hasattr(runtime_time, "to_value"):
                runtime_time = float(
                    np.asarray(runtime_time.to_value(code_units.time_unit))
                )
            else:
                runtime_time = float(
                    np.asarray(runtime_time, dtype=float).reshape(-1)[0]
                )
            par.simulation.time_code = runtime_time
            if canonical_cosmological_schema:
                par.simulation.box_size_comoving_code = getattr(par, box_field)
            else:
                par.simulation.box_size_proper_code = getattr(par, box_field)
            if canonical_cosmological_schema:
                fluid.tau_supercomoving_code = par.simulation.time_code
            elif canonical_proper_schema:
                fluid.time_proper_code = par.simulation.time_code
            else:
                fluid.time_code = par.simulation.time_code.copy() if hasattr(
                    par.simulation.time_code, "copy"
                ) else float(par.simulation.time_code)
        else:
            # Plain parameter namespaces are accepted only as an I/O boundary
            # for callers that do not construct a full Par object.
            time_field = (
                "tau_supercomoving_code"
                if canonical_cosmological_schema
                else "time_proper_code" if canonical_proper_schema else "time_code"
            )
            runtime_time = getattr(par, time_field)
            if hasattr(runtime_time, "to_value"):
                runtime_time = float(
                    np.asarray(runtime_time.to_value(code_units.time_unit))
                )
            else:
                runtime_time = float(
                    np.asarray(runtime_time, dtype=float).reshape(-1)[0]
                )
            if canonical_cosmological_schema:
                fluid.tau_supercomoving_code = runtime_time
            elif canonical_proper_schema:
                fluid.time_proper_code = runtime_time
            else:
                fluid.time_code = runtime_time

        #second, save mesh and fluid data:
        gdata = fic["Data"]
        data_scale_map = {
            "boundary": "length_cgs_cm",
            "boundary_proper_code": "length_cgs_cm",
            "boundary_comoving_code": "length_cgs_cm",
            "rho_code": "density_cgs_g_cm3",
            "rho_proper_code": "density_cgs_g_cm3",
            "rho_comoving_code": "density_cgs_g_cm3",
            "vel_code": "velocity_cgs_cm_s",
            "vel_proper_code": "velocity_cgs_cm_s",
            "vel_supercomoving_code": "velocity_cgs_cm_s",
            "temp_code": "temperature_cgs_K",
            "temp_proper_code": "temperature_cgs_K",
            "temp_supercomoving_code": "temperature_cgs_K",
            "ngamma_code": "number_density_cgs_cm3",
            "Mass_code": "mass_g",
            "Energy_code": "energy_cgs_erg",
            "InternalEnergy_code": "energy_cgs_erg",
            "specific_angular_momentum_code": "specific_angular_momentum",
            "AngularMomentum_code": "angular_momentum",
            "GravitationalPotentialEnergy_code": "energy_cgs_erg",
        }
        _populate_group_targets(
            gdata,
            (mesh, fluid),
            code_units=code_units,
            scale_map=data_scale_map,
        )
        snapshot_cosmology = getattr(par, "cosmology_context", None)
        _attach_radarray_views(
            gdata,
            mesh,
            gdata,
            "cosmological" if canonical_cosmological_schema else "proper",
            code_units,
            snapshot_cosmology,
            allowed_names={
                "boundary",
                "boundary_proper_code",
                "boundary_comoving_code",
            },
        )
        _attach_radarray_views(
            gdata,
            fluid,
            gdata,
            "cosmological" if canonical_cosmological_schema else "proper",
            code_units,
            snapshot_cosmology,
            allowed_names={
                "rho_code",
                "rho_proper_code",
                "rho_comoving_code",
                "vel_code",
                "vel_proper_code",
                "vel_supercomoving_code",
                "temp_code",
                "temp_proper_code",
                "temp_supercomoving_code",
                "pre_proper_code",
                "pre_supercomoving_code",
                "Mass_code",
                "Energy_code",
                "InternalEnergy_code",
                "GravitationalPotentialEnergy_code",
                "AngularMomentum_code",
                "specific_angular_momentum_code",
                "ngamma_code",
            },
        )
        if canonical_proper_schema:
            fluid.runtime_fields = PROPER_RUNTIME_FIELDS
            fluid.runtime_state = FluidRuntimeState.from_arrays(
                PROPER_RUNTIME_FIELDS,
                rho_proper_code=fluid.rho_proper_code,
                vel_proper_code=fluid.vel_proper_code,
                pre_proper_code=getattr(
                    fluid, "pre_proper_code", np.zeros_like(fluid.rho_proper_code)
                ),
                temp_proper_code=fluid.temp_proper_code,
                time_proper_code=getattr(fluid, "time_proper_code", 0.0),
                mu_dimensionless=getattr(fluid, "mu", None),
                xHI_dimensionless=getattr(fluid, "xHI", None),
            )
        elif canonical_cosmological_schema:
            fluid.runtime_fields = SUPERCOMOVING_RUNTIME_FIELDS
            fluid.runtime_state = FluidRuntimeState.from_arrays(
                SUPERCOMOVING_RUNTIME_FIELDS,
                rho_comoving_code=fluid.rho_comoving_code,
                vel_supercomoving_code=fluid.vel_supercomoving_code,
                pre_supercomoving_code=getattr(
                    fluid, "pre_supercomoving_code", np.zeros_like(fluid.rho_comoving_code)
                ),
                temp_supercomoving_code=fluid.temp_supercomoving_code,
                tau_supercomoving_code=getattr(fluid, "tau_supercomoving_code", 0.0),
                mu_dimensionless=getattr(fluid, "mu", None),
                xHI_dimensionless=getattr(fluid, "xHI", None),
            )
        if canonical_cosmological_schema:
            if "boundary_comoving_code" not in gdata:
                raise ValueError("canonical cosmological HDF5 file is missing Data/boundary_comoving_code")
            for name in (
                "rho_comoving_code",
                "vel_supercomoving_code",
                "temp_supercomoving_code",
            ):
                if name not in gdata:
                    raise ValueError(f"canonical cosmological HDF5 file is missing Data/{name}")
        if hasattr(fluid, "code_state"):
            # Force the canonical runtime boundary to validate the restored
            # arrays before a restart can enter solver code.
            _ = fluid.code_state
        par.field_metadata = {}
        for dataset_name, dataset in gdata.items():
            if isinstance(dataset, h5py.Dataset):
                par.field_metadata[dataset_name] = {
                    key: _restore_header_attr_value(value)
                    for key, value in dataset.attrs.items()
                    if key != "units"
                }
        if "DarkMatter" in fic:
            dmdata = fic["DarkMatter"]
            dm_scale_map = {
                "Radius": "length_cgs_cm",
                "RadialVelocity": "velocity_cgs_cm_s",
                "Mass": "mass_g",
                "SpecificAngularMomentum": "specific_angular_momentum",
            }
            _populate_group_targets(
                dmdata,
                (par,),
                code_units=code_units,
                scale_map=dm_scale_map,
            )
            snapshot = {
                "radius": getattr(par, "Radius"),
                "velocity": getattr(par, "RadialVelocity"),
                "mass": getattr(par, "Mass"),
                "angular_momentum": getattr(par, "SpecificAngularMomentum"),
                "softening": _restore_header_attr_value(dmdata.attrs.get("Softening", 0.0)),
            }
            par.dark_matter_snapshot = snapshot
            # A restart snapshot contains the complete live shell state. Build
            # the runtime object so the normal solver/gravity path can resume
            # immediately after ``Callreadhdf5``.
            par.dark_matter = DarkMatterShells(
                radius=snapshot["radius"],
                velocity=snapshot["velocity"],
                mass=snapshot["mass"],
                angular_momentum=snapshot["angular_momentum"],
                softening=snapshot["softening"],
                code_units=code_units,
            )
