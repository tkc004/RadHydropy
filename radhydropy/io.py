"""HDF5 input and output helpers for simulations."""

from pathlib import Path

import h5py
import os
import time
import unyt
import numpy as np
import yaml
import radhydropy.utils as ru
from radhydropy.units import CodeUnits, code_unit_scales, code_quantity_to_cgs, _code_units
from radhydropy.arrays import as_named_array
from radhydropy.dark_matter import DarkMatterShells
from radhydropy.cosmology import EinsteinDeSitter
try:
    from sympy.core.basic import Basic as SympyBasic
except Exception:  # pragma: no cover - optional dependency shape
    SympyBasic = None


def _read_quantity(group, name):
    dataset = group[name]
    return np.asarray(dataset[()]) * unyt.Unit(dataset.attrs['units'])


def _scale_unit_for_key(scale_key):
    return {
        "length_cm": unyt.cm,
        "mass_g": unyt.g,
        "velocity_cm_s": unyt.cm / unyt.s,
        "time_s": unyt.s,
        "temperature_K": unyt.K,
        "area_cm2": unyt.cm**2,
        "volume_cm3": unyt.cm**3,
        "density_g_cm3": unyt.g / unyt.cm**3,
        "pressure_erg_cm3": unyt.erg / unyt.cm**3,
        "energy_erg": unyt.erg,
        "specific_energy_erg_g": unyt.erg / unyt.g,
        "momentum_g_cm_s": unyt.g * unyt.cm / unyt.s,
        "mass_flux_g_cm2_s": unyt.g / (unyt.cm**2 * unyt.s),
        "energy_flux_erg_cm2_s": unyt.erg / (unyt.cm**2 * unyt.s),
        "number_density_cm3": 1.0 / unyt.cm**3,
        "photon_flux_per_cm2_s": 1.0 / (unyt.cm**2 * unyt.s),
        "photon_rate_per_s": 1.0 / unyt.s,
        "alpha_cm3_s": unyt.cm**3 / unyt.s,
        "acceleration_cm_s2": unyt.cm / unyt.s**2,
        "specific_angular_momentum": unyt.cm**2 / unyt.s,
    }.get(scale_key, None)


def _code_unit_for_key(code_units, scale_key):
    if code_units is None:
        return None
    return {
        "length_cm": code_units.length_unit,
        "mass_g": code_units.mass_unit,
        "velocity_cm_s": code_units.velocity_unit,
        "time_s": code_units.time_unit,
        "temperature_K": code_units.temperature_unit,
        "area_cm2": code_units.area_unit,
        "volume_cm3": code_units.volume_unit,
        "density_g_cm3": code_units.density_unit,
        "pressure_erg_cm3": code_units.pressure_unit,
        "energy_erg": code_units.energy_unit,
        "specific_energy_erg_g": code_units.specific_energy_unit,
        "momentum_g_cm_s": code_units.momentum_unit,
        "mass_flux_g_cm2_s": code_units.mass_flux_unit,
        "energy_flux_erg_cm2_s": code_units.energy_flux_unit,
        "number_density_cm3": code_units.number_density_unit,
        "photon_flux_per_cm2_s": 1.0 / (code_units.area_unit * code_units.time_unit),
        "photon_rate_per_s": 1.0 / code_units.time_unit,
        "alpha_cm3_s": code_units.volume_unit / code_units.time_unit,
        "acceleration_cm_s2": code_units.length_unit / code_units.time_unit**2,
        "specific_angular_momentum": code_units.length_unit * code_units.velocity_unit,
        "potential": code_units.velocity_unit**2,
    }.get(scale_key, None)


def _read_runtime_quantity(group, name, code_units=None, scale_key=None):
    dataset = group[name]
    data = np.asarray(dataset[()], dtype=float)
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
            f"Cannot read dataset {name!r} with units {unit_name!r} without a code-unit mapping."
        )
    return as_named_array(data)


def _read_dataset(group, name):
    return as_named_array(group[name][()])


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


def _dataset_aliases(name):
    alias_map = {
        "Boundary": ("boundary",),
        "Density": ("rho",),
        "Velocity": ("vel",),
        "Temperature": ("temp",),
        "Mol_weight": ("mu",),
        "NeutralFraction": ("xHI",),
        "PhotonNumberDensity": ("ngamma",),
        "HeINeutralFraction": ("xHeI",),
        "HeIIFraction": ("xHeII",),
        "HeIIIFraction": ("xHeIII",),
    }
    return alias_map.get(name, ())


def _read_any_dataset(dataset, code_units=None, scale_key=None):
    """Read a dataset and normalize it into code-unit numeric arrays.

    When ``code_units`` and ``scale_key`` are provided, the stored dataset is
    interpreted in its declared unit, converted to the corresponding cgs scale
    for that physical quantity, and then divided by the runtime code-unit
    scale.
    """
    data = np.asarray(dataset[()], dtype=float)
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
            for alias in _dataset_aliases(name):
                setattr(target, alias, value)


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
    metadata=None,
):
    def _unit_label(unit_obj):
        return str(getattr(unit_obj, "units", unit_obj))

    if code_units is None and scale_key is not None:
        raise ValueError(f"{name} requires code_units for HDF5 serialization")

    if hasattr(value, "to_value"):
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
    header.attrs["CoordinateFrame"] = getattr(par, "coordinate_frame", "physical")
    header.attrs["TimeCoordinate"] = getattr(par, "time_coordinate", "cosmic")
    header.attrs["VelocityRepresentation"] = getattr(par, "velocity_representation", "physical")
    header.attrs["DensityRepresentation"] = getattr(par, "density_representation", "physical")
    header.attrs["PressureRepresentation"] = getattr(par, "pressure_representation", "physical")
    header.attrs["TemperatureRepresentation"] = getattr(par, "temperature_representation", "physical")
    header.attrs["ScaleFactor"] = float(cosmology.scale_factor(cosmic_time))
    header.attrs["CosmicTime"] = cosmic_time
    header.attrs["CosmicTimeUnits"] = str(code_units.time_unit)
    header.attrs["SupercomovingTime"] = tau
    header.attrs["SupercomovingTimeUnits"] = str(code_units.time_unit)
    header.attrs["HubbleParameter"] = float(cosmology.hubble(cosmic_time))
    header.attrs["HubbleParameterUnits"] = str(1.0 / code_units.time_unit)


def _restore_cosmology_from_header(par, header, code_units):
    """Restore and validate cosmology metadata from an HDF5 ``Header``."""
    enabled = bool(getattr(par, "cosmological_expansion", False))
    cosmology_type = _restore_header_attr_value(header.attrs.get("CosmologyType", None))
    if not enabled and cosmology_type is None:
        return
    if cosmology_type not in (None, "einstein_de_sitter", "EinsteinDeSitter"):
        raise ValueError("unsupported CosmologyType in HDF5 header: %s" % cosmology_type)
    t_ref = float(_restore_header_attr_value(header.attrs.get("CosmologyTRef", 1.0)))
    a_ref = float(_restore_header_attr_value(header.attrs.get("CosmologyARef", 1.0)))
    par.cosmological_expansion = True
    par.cosmology_type = "einstein_de_sitter"
    par.cosmology_t_ref = t_ref
    par.cosmology_a_ref = a_ref
    par.cosmology = EinsteinDeSitter.from_code_units(
        code_units, t_ref=t_ref, a_ref=a_ref
    )


def _used_parameters_payload(runparams=None, icparams=None, existing=None):
    payload = {}
    if isinstance(existing, dict):
        payload.update(existing)
    if runparams is not None:
        payload["runparams"] = {
            key: _yaml_config_value(value)
            for key, value in sorted(runparams.items())
            if not str(key).startswith("_")
        }
    elif "runparams" not in payload:
        payload["runparams"] = {}
    if icparams is not None:
        payload["ICparams"] = {
            key: _yaml_config_value(value)
            for key, value in sorted(icparams.items())
            if not str(key).startswith("_")
        }
    elif "ICparams" not in payload:
        payload["ICparams"] = {}
    return payload


def update_used_parameters_yaml(path, runparams=None, icparams=None):
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
        runparams=runparams,
        icparams=icparams,
        existing=existing,
    )
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True, default_flow_style=False)
    return path


def write_used_parameters(path, par):
    """Write the active runtime parameters to a YAML file."""
    path = Path(path)
    payload = {
        "runparams": {
            key: parameter_tree(value)
            for key, value in sorted(vars(par).items())
            if not key.startswith("_") and key not in {"runparams", "ICparams"}
        },
        "ICparams": parameter_tree(getattr(par, "ICparams", None)),
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True, default_flow_style=False)
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
    """Write ``Output_###.hdf5`` for the supplied simulation."""
    filename = (
        sim.par.outdir
        + '/'
        + sim.par.outfileprefix
        + '_%03d' % outindex
        + '.hdf5'
    )
    writehdf5(sim, filename)


def hdf5_output_callback(sim, outputtime=0, output_state=None):
    """Return a callback that writes HDF5 snapshots at fixed cadence."""
    if output_state is None:
        output_state = {
            'outtime': 0.0 * sim.par.timesim,
            'outindex': 1,
            'last_output_time_s': float(np.asarray(sim.fluid.time, dtype=float)),
        }
    else:
        output_state.setdefault('outtime', 0.0 * sim.par.timesim)
        output_state.setdefault('outindex', 1)
        output_state.setdefault(
            'last_output_time_s',
            float(np.asarray(sim.fluid.time, dtype=float)),
        )

    def callback(sim, step):
        dt = step["dt"]
        if getattr(dt, "shape", None) == (1,):
            dt = dt[0]
        if getattr(sim.par, 'verbose', 0) >= 1:
            print("time, dt", sim.fluid.time, dt)
        if output_state['outtime'] >= sim.par.outdeltatime:
            sim.fluid.SetTemperature()
            write_numbered_hdf5(sim, output_state['outindex'])
            output_state['last_output_time_s'] = float(
                np.asarray(sim.fluid.time, dtype=float)
            )
            output_state['outtime'] = 0.0 * sim.par.timesim
            output_state['outindex'] += 1
        else:
            output_state['outtime'] += dt

    return callback


def run_with_output_times(
    sim,
    outputtime=0,
    mode="hydro_sources",
    advect_chemistry=True,
    stop_condition=None,
    step_backend=None,
    step_backend_kwargs=None,
):
    """Run a simulation using an explicit output-time list."""
    start = time.time()
    if step_backend is None:
        step_backend = sim.Step
    if step_backend_kwargs is None:
        step_backend_kwargs = {}
    print("--- Initization finished. Start running ... ---")
    print("--- %s seconds ---" % (time.time() - start))
    write_numbered_hdf5(sim, 0)
    last_output_time_s = float(np.asarray(sim.fluid.time, dtype=float))

    current_time = sim.fluid.time
    final_time = sim.par.timesim
    time_tol = max(abs(float(np.asarray(final_time, dtype=float))) * 1.0e-12, 1.0e-30)
    output_times = load_output_time_list(getattr(sim.par, 'outputtimefilename', None))
    if output_times is None:
        output_times = []
    else:
        if hasattr(final_time, 'units'):
            target_unit = final_time.units
            sorted_values = np.unique(
                np.asarray(output_times.to_value(target_unit), dtype=float)
            )
        else:
            code_units = getattr(sim.par, 'CodeUnits', None)
            sorted_values = np.unique(
                np.asarray(output_times.to_value(unyt.s), dtype=float)
                / code_unit_scales(code_units)['time_s']
            )
        output_times = [
            value * final_time.units if hasattr(final_time, 'units') else value
            for value in sorted_values
            if (
                (value * final_time.units if hasattr(final_time, 'units') else value)
                > current_time
                and (value * final_time.units if hasattr(final_time, 'units') else value)
                <= final_time
            )
        ]

    outindex = 1
    for target_time in output_times:
        if stop_condition is not None and stop_condition(sim):
            break
        target_time_value = float(np.asarray(target_time, dtype=float))
        while float(np.asarray(sim.fluid.time, dtype=float)) < target_time_value - time_tol:
            if stop_condition is not None and stop_condition(sim):
                break
            dt = sim.GetStepTime(final_time=target_time)
            if getattr(sim.par, 'verbose', 0) >= 1:
                print("time, dt", sim.fluid.time, dt)
            step_backend(
                dt=dt,
                mode=mode,
                advect_chemistry=advect_chemistry,
                **step_backend_kwargs,
            )
        if stop_condition is not None and stop_condition(sim):
            break
        # Euler/source steps can cross a target by a roundoff- or CFL-sized
        # amount.  Treat the first state at or beyond the target as the
        # requested snapshot instead of silently dropping the output.
        if float(np.asarray(sim.fluid.time, dtype=float)) >= target_time_value - time_tol:
            sim.fluid.SetTemperature()
            write_numbered_hdf5(sim, outindex)
            last_output_time_s = float(np.asarray(sim.fluid.time, dtype=float))
            outindex += 1

    final_time_value = float(np.asarray(final_time, dtype=float))
    while float(np.asarray(sim.fluid.time, dtype=float)) < final_time_value - time_tol:
        if stop_condition is not None and stop_condition(sim):
            break
        dt = sim.GetStepTime(final_time=final_time)
        if getattr(sim.par, 'verbose', 0) >= 1:
            print("time, dt", sim.fluid.time, dt)
        step_backend(
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
            **step_backend_kwargs,
        )

    if stop_condition is not None and abs(float(np.asarray(sim.fluid.time, dtype=float)) - last_output_time_s) > time_tol:
        sim.fluid.SetTemperature()
        write_numbered_hdf5(sim, outindex)

    print("--- Simulation finished. ---")
    print("--- %s seconds ---" % (time.time() - start))

def writehdf5(ric,ICfilename):
    """Write simulation state to a RadHydropy HDF5 file.

    The output file contains a ``Header`` group for metadata and a ``Data``
    group for mesh and fluid arrays. Units are stored as HDF5 attributes.
    """
    ICfilename = str(ICfilename)
    print(f"--- writing {ICfilename} --- ")
    if hasattr(ric.fluid, "time"):
        output_time = ric.fluid.time
    else:
        output_time = ric.par.time
    with h5py.File(ICfilename, 'w') as fic:
        code_units = getattr(ric.par, "CodeUnits", None)
        # saving initial condition
        # first, save header:
        header = fic.create_group("Header")
        for key, value in sorted(vars(ric.par).items()):
            if key.startswith("_") or key in {"dark_matter", "dark_matter_snapshot", "cosmology"}:
                continue
            header.attrs[key] = _header_attr_value(value)
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
            "Time",
            output_time,
            code_units=code_units,
            scale_key="time_s",
            default_unit=unyt.s,
        )
        _write_quantity(
            header,
            "BoxSize",
            ric.par.boxsize,
            code_units=code_units,
            scale_key="length_cm",
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
            "Boundary",
            ric.mesh.boundary,
            code_units=code_units,
            scale_key="length_cm",
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
        _write_quantity(
            gdata,
            "Density",
            ric.fluid.rho,
            code_units=code_units,
            scale_key="density_g_cm3",
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
        )
        _write_quantity(
            gdata,
            "Velocity",
            ric.fluid.vel,
            code_units=code_units,
            scale_key="velocity_cm_s",
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
        )
        _write_quantity(
            gdata,
            "Temperature",
            ric.fluid.temp,
            code_units=code_units,
            scale_key="temperature_K",
            default_unit=unyt.K,
            metadata={
                "quantity": "temperature",
                "representation": (
                    getattr(ric.par, "temperature_representation", "physical")
                    if getattr(ric.par, "supercomoving_coordinates", False)
                    else "physical"
                ),
                "scale_factor_power": (
                    3.0 * (getattr(ric.par, "gamma", 5.0 / 3.0) - 1.0)
                    if getattr(ric.par, "supercomoving_coordinates", False)
                    else 0.0
                ),
            },
        )
        for attr, dataset_name in (
            ("Mass", "Mass"),
            ("Energy", "Energy"),
            ("InternalEnergy", "InternalEnergy"),
        ):
            if hasattr(ric.fluid, attr):
                _write_quantity(
                    gdata,
                    dataset_name,
                    getattr(ric.fluid, attr),
                    code_units=code_units,
                    scale_key="mass_g" if attr == "Mass" else "energy_erg",
                    default_unit=unyt.g if attr == "Mass" else unyt.erg,
                )
        gdata.create_dataset("Mol_weight", data=np.asarray(ric.fluid.mu))
        if hasattr(ric.fluid, "xHI"):
            gdata.create_dataset("NeutralFraction", data=np.asarray(ric.fluid.xHI))
        for attr, dataset in (("xHeI", "HeINeutralFraction"), ("xHeII", "HeIIFraction"), ("xHeIII", "HeIIIFraction")):
            if hasattr(ric.fluid, attr):
                gdata.create_dataset(dataset, data=np.asarray(getattr(ric.fluid, attr)))
        if hasattr(ric.fluid, "ngamma"):
            _write_quantity(
                gdata,
                "PhotonNumberDensity",
                ric.fluid.ngamma,
                code_units=code_units,
                scale_key="number_density_cm3",
                default_unit=1.0 / unyt.cm**3,
            )
        if getattr(ric.par, "cosmological_expansion", False):
            for dataset_name, dataset in gdata.items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                if dataset_name in {"Boundary", "Density", "Velocity", "Temperature"}:
                    continue
                dataset.attrs["representation"] = "physical"
        dark_matter = getattr(ric.par, "dark_matter", None)
        if dark_matter is None:
            gravity = getattr(ric.par, "gravity", None)
            dark_matter = getattr(gravity, "dark_matter", None)
        if dark_matter is not None:
            dmdata = fic.create_group("DarkMatter")
            _write_quantity(dmdata, "Radius", dark_matter.radius,
                            code_units=code_units, scale_key="length_cm",
                            default_unit=unyt.cm)
            _write_quantity(dmdata, "RadialVelocity", dark_matter.velocity,
                            code_units=code_units, scale_key="velocity_cm_s",
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
            icparams=vars(ric.par),
        )



def readhdf5(par, mesh, fluid, ICfilename): 
    """Read a RadHydropy HDF5 file into parameter, mesh, and fluid objects.

    Datasets such as ``Density`` are restored into the runtime code-unit
    system when ``CodeUnits`` is available in the file header, so
    ``fluid.rho`` comes back as a plain numeric array in code units.
    """
    ICfilename = str(ICfilename)
    print(f"--- reading {ICfilename} --- ")
    with h5py.File(ICfilename, 'r') as fic:
        expected_coordsys = getattr(par, "coordsys", None)
        expected_nogrid = getattr(par, "nogrid", None)
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
        for key, value in header.attrs.items():
            restored = _restore_header_attr_value(value)
            if key == "CodeUnits":
                if isinstance(restored, dict):
                    restored = CodeUnits.from_mapping(restored)
                setattr(par, "CodeUnits", restored)
                continue
            setattr(par, key, restored)
        _restore_cosmology_from_header(par, header, code_units)
        if expected_coordsys is not None and par.coordsys != expected_coordsys:
            raise Exception(
                "Coordinate systems in IC (%s) and run (%s) do not agree!"
                % (par.coordsys, expected_coordsys)
            )
        if expected_nogrid is not None and par.nogrid != expected_nogrid:
            raise Exception(
                "Number of grids in IC (%s) and run (%s) do not agree!"
                % (par.nogrid, expected_nogrid)
            )
        header_scale_map = {
            "Time": "time_s",
            "BoxSize": "length_cm",
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
        if hasattr(par, 'load_radiation_spectrum'):
            par.load_radiation_spectrum(getattr(par, 'outdir', None))
        par.time = getattr(par, "Time")
        par.boxsize = getattr(par, "BoxSize")
        fluid.time = par.time.copy() if hasattr(par.time, "copy") else float(par.time)

        #second, save mesh and fluid data:
        gdata = fic["Data"]
        data_scale_map = {
            "Boundary": "length_cm",
            "Density": "density_g_cm3",
            "Velocity": "velocity_cm_s",
            "Temperature": "temperature_K",
            "PhotonNumberDensity": "number_density_cm3",
            "Mass": "mass_g",
            "Energy": "energy_erg",
            "InternalEnergy": "energy_erg",
        }
        _populate_group_targets(
            gdata,
            (mesh, fluid),
            code_units=code_units,
            scale_map=data_scale_map,
        )
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
                "Radius": "length_cm",
                "RadialVelocity": "velocity_cm_s",
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

        # Preserve the canonical runtime field names expected by the solver.
        if hasattr(mesh, "Boundary"):
            mesh.boundary = getattr(mesh, "Boundary")
        if hasattr(fluid, "Density"):
            fluid.rho = getattr(fluid, "Density")
        if hasattr(fluid, "Velocity"):
            fluid.vel = getattr(fluid, "Velocity")
        if hasattr(fluid, "Temperature"):
            fluid.temp = getattr(fluid, "Temperature")
        if hasattr(fluid, "Mol_weight"):
            fluid.mu = getattr(fluid, "Mol_weight")
        if hasattr(fluid, "NeutralFraction"):
            fluid.xHI = getattr(fluid, "NeutralFraction")
        if hasattr(fluid, "PhotonNumberDensity"):
            fluid.ngamma = getattr(fluid, "PhotonNumberDensity")
        for dataset, attr in (("HeINeutralFraction", "xHeI"), ("HeIIFraction", "xHeII"), ("HeIIIFraction", "xHeIII")):
            if hasattr(fluid, dataset):
                setattr(fluid, attr, getattr(fluid, dataset))
