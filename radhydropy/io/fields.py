"""Dataset scaling and canonical field restoration helpers."""

import h5py
import numpy as np
import unyt

from radhydropy.arrays import as_named_array
from radhydropy.field_metadata import FieldSpec
from radhydropy.io.metadata import _restore_header_attr_value
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
    normalized = [
        char if char.isalnum() or char == "_" else "_"
        for char in str(name)
    ]
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
            cgs_unit = scale_unit_for_key(scale_key)
            if cgs_unit is not None:
                data = unyt.unyt_array(data, stored_unit).to_value(cgs_unit)
        return as_named_array(data / scales[scale_key])
    if unit_name:
        raise ValueError(
            f"Cannot read dataset {dataset.name!r} with units {unit_name!r} "
            "without a code-unit mapping."
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


def write_quantity(
    group, name, value, code_units=None, scale_key=None, default_unit=None,
    metadata=None, field_spec_obj=None,
):
    """Write one quantity with canonical units and field metadata."""
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
        code_unit = scale_unit_for_key(scale_key)
        if code_unit is None:
            raise ValueError(f"no code-unit scale is defined for {scale_key!r}")
        data = np.asarray(
            value.to_value(code_unit) if hasattr(value, "to_value") else value,
            dtype=float,
        )
        unit = str(code_unit)
    elif hasattr(value, "to_value"):
        if code_units is not None and scale_key is not None:
            unit_obj = scale_unit_for_key(scale_key) or value.units
            data = np.asarray(value.to_value(unit_obj))
            unit = _unit_label(unit_obj)
        else:
            data = np.asarray(value.to_value(value.units))
            unit = str(value.units)
    elif code_units is not None and scale_key is not None:
        unit_obj = scale_unit_for_key(scale_key)
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
