"""HDF5 metadata, provenance, and configuration serialization helpers."""

import hashlib
from pathlib import Path

import h5py
import numpy as np
import unyt
import yaml

try:
    from sympy.core.basic import Basic as SympyBasic
except Exception:  # pragma: no cover - optional dependency shape
    SympyBasic = None


def _provenance_yaml_value(value):
    """Convert nested configuration values into YAML-safe values."""
    if isinstance(value, unyt.array.unyt_array):
        numeric_value = np.asarray(value.value)
        numeric_value = numeric_value.item() if numeric_value.ndim == 0 else numeric_value.tolist()
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
        provenance.get("source_config_yaml", ""),
    )
    effective_value = provenance.get("effective_config_yaml")
    if effective_value is None:
        effective_value = provenance.get("effective_config", {})
    effective_yaml = _provenance_yaml_text(effective_value)
    provenance_group = header.create_group("Provenance")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    provenance_group.create_dataset(
        "source_config_yaml",
        data=source_yaml,
        dtype=string_dtype,
    )
    provenance_group.create_dataset(
        "effective_config_yaml",
        data=effective_yaml,
        dtype=string_dtype,
    )
    provenance_group.attrs["source_config_sha256"] = hashlib.sha256(
        source_yaml.encode("utf-8"),
    ).hexdigest()
    provenance_group.attrs["effective_config_sha256"] = hashlib.sha256(
        effective_yaml.encode("utf-8"),
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

    def decode(value):
        return value.decode("utf-8") if isinstance(value, bytes) else value

    provenance = {
        "source_config_yaml": decode(group["source_config_yaml"][()]),
        "effective_config_yaml": decode(group["effective_config_yaml"][()]),
    }
    for key, value in group.attrs.items():
        restored = _restore_header_attr_value(value)
        provenance[key] = restored
    return provenance


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
            return {"value": float(raw_value.reshape(-1)[0]), "unit": str(value.units)}
        return {"value": raw_value.tolist(), "unit": str(value.units)}
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
                "simulation",
                "mesh",
                "hydrodynamics",
                "boundary",
                "timestep",
                "output",
                "diagnostics",
                "units",
                "thermochemistry",
                "chemistry",
                "gravity",
                "dark_matter",
                "radiation",
            )
        }
        runtime_parameters.update(
            {
                key: _yaml_config_value(value)
                for key, value in nested_par_config.items()
                if key not in runtime_parameters
            },
        )
    else:
        runtime_parameters = {
            key: parameter_tree(value)
            for key, value in sorted(vars(par).items())
            if not key.startswith("_") and key not in {"par_config", "initial_condition"}
        }
    payload = {
        "par": runtime_parameters,
        "initial_condition": parameter_tree(
            getattr(par, "initial_condition", None),
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
        isinstance(item, (str, bytes, int, float, bool, np.generic)) for item in tree
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
        if {"value", "unit"} <= value.keys():
            restored_value = _restore_header_attr_value(value["value"])
            unit = unyt.Unit(value["unit"])
            if isinstance(restored_value, list):
                restored_value = np.asarray(restored_value)
            return np.asarray(restored_value) * unit
        return {key: _restore_header_attr_value(item) for key, item in value.items()}
    return value
