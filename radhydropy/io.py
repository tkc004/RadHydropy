"""HDF5 input and output helpers for simulations."""

from pathlib import Path

import h5py
import os
import time
import unyt
import numpy as np
import yaml
import radhydropy.utils as ru
from radhydropy.units import code_unit_scales, _code_units
from radhydropy.arrays import as_named_array
try:
    from sympy.core.basic import Basic as SympyBasic
except Exception:  # pragma: no cover - optional dependency shape
    SympyBasic = None


def _read_quantity(group, name):
    dataset = group[name]
    return np.asarray(dataset[()]) * unyt.Unit(dataset.attrs['units'])


def _read_runtime_quantity(group, name, code_units=None, scale_key=None, preserve_units=False):
    dataset = group[name]
    data = np.asarray(dataset[()], dtype=float)
    if preserve_units:
        return data * unyt.Unit(dataset.attrs['units'])
    if code_units is not None and scale_key is not None:
        scales = code_unit_scales(code_units)
        return as_named_array(data / scales[scale_key])
    return data * unyt.Unit(dataset.attrs['units'])


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
    }
    return alias_map.get(name, ())


def _read_any_dataset(dataset, code_units=None, scale_key=None, preserve_units=False):
    data = np.asarray(dataset[()], dtype=float)
    if preserve_units:
        unit_name = dataset.attrs.get("units", None)
        return data * unyt.Unit(unit_name) if unit_name else data
    if code_units is not None and scale_key is not None:
        scales = code_unit_scales(code_units)
        return as_named_array(data / scales[scale_key])
    unit_name = dataset.attrs.get("units", None)
    return as_named_array(data * unyt.Unit(unit_name)) if unit_name else as_named_array(data)


def _populate_group_targets(group, targets, code_units=None, preserve_units=False, scale_map=None):
    scale_map = scale_map or {}
    for name, dataset in group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        value = _read_any_dataset(
            dataset,
            code_units=code_units,
            scale_key=scale_map.get(name),
            preserve_units=preserve_units,
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


def _write_quantity(group, name, value, code_units=None, scale_key=None, default_unit=None):
    if hasattr(value, "to_value"):
        data = np.asarray(value.to_value(value.units))
        unit = str(value.units)
    elif code_units is not None and scale_key is not None:
        scales = code_unit_scales(code_units)
        data = np.asarray(value, dtype=float) * scales[scale_key]
        unit = str(default_unit)
    else:
        data = np.asarray(value)
        unit = str(default_unit) if default_unit is not None else "dimensionless"
    dataset = group.create_dataset(name, data=data)
    dataset.attrs["units"] = unit
    return dataset


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
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
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
        if outputtime == 1:
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
            code_units = getattr(sim.par, 'code_units', getattr(sim.par, 'CodeUnits', None))
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
        while sim.fluid.time < target_time:
            if stop_condition is not None and stop_condition(sim):
                break
            dt = sim.GetStepTime(final_time=target_time)
            if outputtime == 1:
                print("time, dt", sim.fluid.time, dt)
            step_backend(
                dt=dt,
                mode=mode,
                advect_chemistry=advect_chemistry,
                **step_backend_kwargs,
            )
        if stop_condition is not None and stop_condition(sim):
            break
        if sim.fluid.time == target_time:
            sim.fluid.SetTemperature()
            write_numbered_hdf5(sim, outindex)
            last_output_time_s = float(np.asarray(sim.fluid.time, dtype=float))
            outindex += 1

    while sim.fluid.time < final_time:
        if stop_condition is not None and stop_condition(sim):
            break
        dt = sim.GetStepTime(final_time=final_time)
        if outputtime == 1:
            print("time, dt", sim.fluid.time, dt)
        step_backend(
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
            **step_backend_kwargs,
        )

    if stop_condition is not None and float(np.asarray(sim.fluid.time, dtype=float)) != last_output_time_s:
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
        code_units = getattr(ric.par, "code_units", getattr(ric.par, "CodeUnits", None))
        # saving initial condition
        # first, save header:
        header = fic.create_group("Header")
        header.attrs['Coordinate_System'] = ric.par.coordsys
        header.attrs['Number_Grids'] = ric.par.nogrid
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
        )
        ric.par.time = np.asarray(output_time, dtype=float)

        #second, save mesh and fluid data:
        gdata = fic.create_group("Data")
        _write_quantity(
            gdata,
            "Boundary",
            ric.mesh.boundary,
            code_units=code_units,
            scale_key="length_cm",
            default_unit=unyt.cm,
        )
        _write_quantity(
            gdata,
            "Density",
            ric.fluid.rho,
            code_units=code_units,
            scale_key="density_g_cm3",
            default_unit=unyt.g / unyt.cm**3,
        )
        _write_quantity(
            gdata,
            "Velocity",
            ric.fluid.vel,
            code_units=code_units,
            scale_key="velocity_cm_s",
            default_unit=unyt.cm / unyt.s,
        )
        _write_quantity(
            gdata,
            "Temperature",
            ric.fluid.temp,
            code_units=code_units,
            scale_key="temperature_K",
            default_unit=unyt.K,
        )
        gdata.create_dataset("Mol_weight", data=np.asarray(ric.fluid.mu))
        if hasattr(ric.fluid, "xHI"):
            gdata.create_dataset("NeutralFraction", data=np.asarray(ric.fluid.xHI))
        if hasattr(ric.fluid, "ngamma"):
            _write_quantity(
                gdata,
                "PhotonNumberDensity",
                ric.fluid.ngamma,
                code_units=code_units,
                scale_key="number_density_cm3",
                default_unit=1.0 / unyt.cm**3,
            )

    if (
        not hasattr(ric, "solver")
        and Path(ICfilename).stem.lower() == "initialcondition"
    ):
        update_used_parameters_yaml(
            Path.cwd() / "used_parameters.yaml",
            icparams=vars(ric.par),
        )



def readhdf5(par, mesh, fluid, ICfilename, preserve_units=True): 
    """Read a RadHydropy HDF5 file into parameter, mesh, and fluid objects.

    Parameters
    ----------
    preserve_units : bool, optional
        When ``True``, return ``unyt`` quantities exactly as stored on disk.
        When ``False`` and code units are available, convert the runtime state
        into plain NumPy floats in code units for faster simulation.
    """
    ICfilename = str(ICfilename)
    print(f"--- reading {ICfilename} --- ")
    with h5py.File(ICfilename, 'r') as fic:
        code_units = getattr(par, "code_units", getattr(par, "CodeUnits", None))
        # saving initial condition
        # first, save header:
        header = fic["Header"]
        coordsys = header.attrs['Coordinate_System']
        if hasattr(par, "coordsys"): 
            if coordsys != par.coordsys:
                raise Exception("Coordinate systems in IC (%s) and run (%s) do not agree!"%(coordsys,par.coordsys))
        else:
            par.coordsys = coordsys
        par.nogrid = header.attrs['Number_Grids']
        header_scale_map = {
            "Time": "time_s",
            "BoxSize": "length_cm",
        }
        _populate_group_targets(
            header,
            (par,),
            code_units=code_units,
            preserve_units=preserve_units,
            scale_map=header_scale_map,
        )
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
        }
        _populate_group_targets(
            gdata,
            (mesh, fluid),
            code_units=code_units,
            preserve_units=preserve_units,
            scale_map=data_scale_map,
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
