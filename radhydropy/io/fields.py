# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Dataset scaling and canonical field restoration helpers."""

import h5py
import numpy as np
import unyt

from radhydropy.arrays import as_named_array
from radhydropy.dark_matter import DarkMatterSnapshot
from radhydropy.field_metadata import FieldSpec, field_spec
from radhydropy.io.metadata import _restore_header_attr_value
from radhydropy.radarray import RadArray, RadQuantity
from radhydropy.units import code_quantity_to_cgs, code_unit_scales


def scale_unit_for_key(scale_key):
    """Return the cgs unit associated with a canonical scale key."""
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
    }.get(scale_key)


def normalize_attr_name(name):
    """Return a safe Python attribute name for an HDF5 dataset name."""
    normalized = [char if char.isalnum() or char == "_" else "_" for char in str(name)]
    return "".join(normalized).strip("_") or "field"


def read_any_dataset(dataset, code_units=None, scale_key=None):
    """Read a dataset and normalize it into code-unit numeric arrays."""
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
                f"Dataset {dataset.name!r} has invalid FieldSpec metadata",
            ) from exc
        return as_named_array(data)
    if storage_unit not in {None, "cgs"}:
        raise ValueError(
            f"Dataset {dataset.name!r} has unsupported storage_unit {storage_unit!r}",
        )
    unit_name = dataset.attrs.get("units", None)
    if code_units is not None and scale_key is not None:
        scales = code_unit_scales(code_units)
        if unit_name:
            stored_unit = unyt.Unit(unit_name)
            cgs_unit = scale_unit_for_key(scale_key)
            if cgs_unit is not None:
                data = unyt.unyt_array(data, stored_unit).to_value(cgs_unit)
        return as_named_array(data / scales[scale_key])
    if unit_name:
        raise ValueError(
            f"Cannot read dataset {dataset.name!r} with units {unit_name!r} "
            "without a code-unit mapping.",
        )
    return as_named_array(data)


def populate_group_targets(group, targets, code_units=None, scale_map=None):
    """Restore HDF5 datasets as named numeric attributes on target objects."""
    scale_map = scale_map or {}
    for name, dataset in group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        value = read_any_dataset(
            dataset,
            code_units=code_units,
            scale_key=scale_map.get(name),
        )
        attr_name = normalize_attr_name(name)
        for target in targets:
            setattr(target, attr_name, value)


def _quantity_storage_data(
    name,
    value,
    code_units,
    scale_key,
    default_unit,
    field_spec_obj,
):
    if code_units is None and scale_key is not None:
        raise ValueError(f"{name} requires code_units for HDF5 serialization")
    storage_unit = field_spec_obj.storage_unit if field_spec_obj is not None else "cgs"
    if field_spec_obj is not None and storage_unit == "code":
        if code_units is None or scale_key is None:
            raise ValueError(f"{name} requires code_units and scale_key for code storage")
        code_unit = scale_unit_for_key(scale_key)
        if code_unit is None:
            raise ValueError(f"no code-unit scale is defined for {scale_key!r}")
        data = np.asarray(
            value.to_value(code_unit) if hasattr(value, "to_value") else value,
            dtype=float,
        )
        return data, str(code_unit), storage_unit
    if hasattr(value, "to_value"):
        if code_units is not None and scale_key is not None:
            unit_obj = scale_unit_for_key(scale_key) or value.units
            data = np.asarray(value.to_value(unit_obj))
            return data, str(getattr(unit_obj, "units", unit_obj)), storage_unit
        return np.asarray(value.to_value(value.units)), str(value.units), storage_unit
    if code_units is not None and scale_key is not None:
        unit_obj = scale_unit_for_key(scale_key)
        if unit_obj is None:
            unit_obj = unyt.Unit(default_unit) if default_unit is not None else None
        data = code_quantity_to_cgs(value, code_units, scale_key)
        unit = (
            str(getattr(unit_obj, "units", unit_obj)) if unit_obj is not None else "dimensionless"
        )
        return data, unit, storage_unit
    data = np.asarray(value)
    unit = str(unyt.Unit(default_unit)) if default_unit is not None else "dimensionless"
    return data, unit, storage_unit


def write_quantity(
    group,
    name,
    value,
    code_units=None,
    scale_key=None,
    default_unit=None,
    metadata=None,
    field_spec_obj=None,
):
    """Write one quantity with canonical units and field metadata."""

    data, unit, storage_unit = _quantity_storage_data(
        name,
        value,
        code_units,
        scale_key,
        default_unit,
        field_spec_obj,
    )
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


def radarray_field_spec(dataset, canonical_name, code_units, cosmology):
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
    expected_representation = None
    if canonical_name.endswith("_comoving_code"):
        expected_representation = "comoving"
    elif canonical_name.endswith("_supercomoving_code"):
        expected_representation = "supercomoving"
    elif canonical_name.endswith("_proper_code"):
        expected_representation = "proper"
    if expected_representation is not None and restored.representation != expected_representation:
        raise ValueError(
            f"{dataset.name!r} uses representation "
            f"{restored.representation!r}; expected "
            f"{expected_representation!r} for {canonical_name!r}",
        )
    return restored


def attach_radarray_views(
    group,
    target,
    dataset_names,
    canonical_schema,
    code_units,
    cosmology,
    allowed_names=None,
):
    """Expose loaded dimensional fields as typed ``*_radarray`` views."""
    del group  # The target and dataset mapping are sufficient for restoration.
    if cosmology is None:
        return
    if canonical_schema == "cosmological":
        mapping = {
            "boundary_comoving_code": ("boundary_radarray", "boundary_comoving_code"),
            "rho_comoving_code": ("rho_radarray", "rho_comoving_code"),
            "vel_supercomoving_code": ("vel_radarray", "vel_supercomoving_code"),
            "temp_supercomoving_code": ("temp_radarray", "temp_supercomoving_code"),
            "pre_supercomoving_code": ("pre_radarray", "pre_supercomoving_code"),
            "ngamma_code": ("ngamma_radarray", "ngamma_comoving_code"),
        }
    else:
        mapping = {
            "boundary_proper_code": ("boundary_radarray", "boundary_proper_code"),
            "rho_proper_code": ("rho_radarray", "rho_proper_code"),
            "vel_proper_code": ("vel_radarray", "vel_proper_code"),
            "temp_proper_code": ("temp_radarray", "temp_proper_code"),
            "pre_proper_code": ("pre_radarray", "pre_proper_code"),
            "ngamma_code": ("ngamma_radarray", "ngamma_proper_code"),
        }
    _attach_canonical_radarray_views(
        target,
        dataset_names,
        mapping,
        allowed_names,
        code_units,
        cosmology,
    )
    _attach_extra_radarray_views(
        target,
        dataset_names,
        mapping,
        allowed_names,
        code_units,
        cosmology,
    )


def _attach_canonical_radarray_views(
    target,
    dataset_names,
    mapping,
    allowed_names,
    code_units,
    cosmology,
):
    for dataset_name, (view_name, canonical_name) in mapping.items():
        if allowed_names is not None and dataset_name not in allowed_names:
            continue
        if dataset_name not in dataset_names:
            continue
        attr_name = normalize_attr_name(dataset_name)
        if not hasattr(target, attr_name):
            continue
        spec = radarray_field_spec(
            dataset_names[dataset_name],
            canonical_name,
            code_units,
            cosmology,
        )
        setattr(
            target,
            view_name,
            RadArray(
                np.asarray(getattr(target, attr_name), dtype=float),
                code_units=code_units,
                field_spec=spec,
                cosmology=cosmology,
                field_name=canonical_name,
            ),
        )


def _attach_extra_radarray_views(
    target,
    dataset_names,
    mapping,
    allowed_names,
    code_units,
    cosmology,
):
    for dataset_name, dataset in dataset_names.items():
        if dataset_name in mapping or (
            allowed_names is not None and dataset_name not in allowed_names
        ):
            continue
        attr_name = normalize_attr_name(dataset_name)
        if not hasattr(target, attr_name):
            continue
        try:
            spec = radarray_field_spec(dataset, dataset_name, code_units, cosmology)
        except (TypeError, ValueError):
            continue
        radarray_name = attr_name.removesuffix("_code")
        setattr(
            target,
            f"{radarray_name}_radarray",
            RadArray(
                np.asarray(getattr(target, attr_name), dtype=float),
                code_units=code_units,
                field_spec=spec,
                cosmology=cosmology,
                field_name=dataset_name,
            ),
        )


def attach_dark_matter_radarray_views(
    group,
    par,
    code_units,
    cosmology,
    canonical_schema,
):
    """Restore the typed analysis view for a ``DarkMatter`` HDF5 group."""
    if group is None or cosmology is None:
        return None
    if canonical_schema == "cosmological":
        field_names = {
            "Radius": "radius_comoving_code",
            "RadialVelocity": "vel_supercomoving_code",
            "Mass": "dark_matter_mass_code",
            "SpecificAngularMomentum": "specific_angular_momentum_supercomoving_code",
        }
    else:
        field_names = {
            "Radius": "radius_proper_code",
            "RadialVelocity": "vel_proper_code",
            "Mass": "dark_matter_mass_code",
            "SpecificAngularMomentum": "specific_angular_momentum_proper_code",
        }
    views = {}
    for dataset_name, canonical_name in field_names.items():
        if dataset_name not in group or not hasattr(par, dataset_name):
            raise ValueError(
                f"DarkMatter group is missing required dataset {dataset_name!r}",
            )
        dataset = group[dataset_name]
        spec = radarray_field_spec(dataset, canonical_name, code_units, cosmology)
        views[dataset_name] = RadArray(
            np.asarray(getattr(par, dataset_name), dtype=float),
            code_units=code_units,
            field_spec=spec,
            cosmology=cosmology,
            field_name=canonical_name,
        )

    softening_runtime_code = _restore_header_attr_value(
        group.attrs.get("Softening", 0.0),
    )
    if hasattr(softening_runtime_code, "to_value"):
        softening_runtime_code = float(
            np.asarray(softening_runtime_code.to_value(code_units.length_unit)),
        )
    else:
        softening_runtime_code = float(softening_runtime_code)
    softening_field_name = (
        "radius_comoving_code" if canonical_schema == "cosmological" else "radius_proper_code"
    )
    softening_spec = radarray_field_spec(
        group["Radius"],
        softening_field_name,
        code_units,
        cosmology,
    )
    return DarkMatterSnapshot(
        radius_radarray=views["Radius"],
        radial_velocity_radarray=views["RadialVelocity"],
        dark_matter_mass_radarray=views["Mass"],
        specific_angular_momentum_radarray=views["SpecificAngularMomentum"],
        softening_radquantity=RadQuantity(
            softening_runtime_code,
            code_units=code_units,
            field_spec=softening_spec,
            cosmology=cosmology,
            field_name=softening_field_name,
        ),
    )
