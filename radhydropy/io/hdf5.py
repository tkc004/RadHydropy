# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""HDF5 input and output helpers for simulations."""

import logging
from pathlib import Path

import h5py
import numpy as np
import unyt

from radhydropy.dark_matter import DarkMatterShells
from radhydropy.diagnostic_logging import log_diagnostic
from radhydropy.io.cosmology import (
    restore_cosmology_context_from_header,
    restore_cosmology_from_header,
    runtime_field_spec,
    write_cosmology_header,
)
from radhydropy.io.fields import (
    attach_dark_matter_radarray_views,
    attach_radarray_views,
    normalize_attr_name,
    populate_group_targets,
    radarray_field_spec,
    read_any_dataset,
    scale_unit_for_key,
    write_quantity,
)
from radhydropy.io.metadata import (
    _header_attr_value,
    _read_provenance,
    _restore_header_attr_value,
    _write_provenance,
    update_used_parameters_yaml,
)
from radhydropy.io.validation import (
    validate_snapshot_configuration,
)
from radhydropy.runtime_fields import (
    PROPER_RUNTIME_FIELDS,
    SUPERCOMOVING_RUNTIME_FIELDS,
    FluidRuntimeState,
)
from radhydropy.units import CodeUnits

# Internal names remain stable while the field implementation lives in its
# dedicated module.  These are direct imports, not public compatibility APIs.
_scale_unit_for_key = scale_unit_for_key
_normalize_attr_name = normalize_attr_name
_read_any_dataset = read_any_dataset
_populate_group_targets = populate_group_targets
_write_quantity = write_quantity
_write_cosmology_header = write_cosmology_header
_restore_cosmology_from_header = restore_cosmology_from_header
_restore_cosmology_context_from_header = restore_cosmology_context_from_header
_runtime_field_spec = runtime_field_spec
_radarray_field_spec = radarray_field_spec
_attach_radarray_views = attach_radarray_views
_attach_dark_matter_radarray_views = attach_dark_matter_radarray_views


_GENERIC_HEADER_DATASETS = frozenset({"time_code", "box_size_code"})
_GENERIC_PRIMITIVE_DATASETS = frozenset(
    {"boundary", "rho_code", "vel_code", "temp_code", "pre_code"},
)


_validate_snapshot_configuration = validate_snapshot_configuration


def _write_runtime_header_attributes(header, ric, cosmological_schema):
    """Write scalar runtime parameters while excluding nested configuration groups."""
    excluded = {
        "dark_matter",
        "dark_matter_snapshot",
        "cosmology",
        "hydrodynamics",
        "boundary",
        "timestep",
        "thermochemistry",
        "gravity",
        "output",
        "simulation",
        "diagnostics",
        "mesh",
        "chemistry",
        "angular_momentum",
        "dark_matter_config",
        "dark_matter_radarrays",
        "dual_energy_config",
        "positivity",
        "radiation",
        "units",
    }
    for key, value in sorted(vars(ric.par).items()):
        if key.startswith("_") or key in excluded:
            continue
        if key in {"time_code", "box_size_code"}:
            continue
        if not cosmological_schema and key in {"tau_supercomoving_code", "time_cosmic_code"}:
            continue
        header.attrs[key] = _header_attr_value(value)


def _write_dark_matter_snapshot(fic, ric, code_units, output_time, cosmological_schema):
    dark_matter = getattr(ric.par, "dark_matter", None)
    if dark_matter is None:
        dark_matter = getattr(getattr(ric.par, "gravity", None), "dark_matter", None)
    if dark_matter is None:
        return
    dmdata = fic.create_group("DarkMatter")
    dm_radius_name = "radius_comoving_code" if cosmological_schema else "radius_proper_code"
    dm_velocity_name = "vel_supercomoving_code" if cosmological_schema else "vel_proper_code"
    dm_angular_name = (
        "specific_angular_momentum_supercomoving_code"
        if cosmological_schema
        else "specific_angular_momentum_proper_code"
    )
    fields = (
        ("Radius", dark_matter.radius, "length_cgs_cm", unyt.cm, dm_radius_name),
        (
            "RadialVelocity",
            dark_matter.velocity,
            "velocity_cgs_cm_s",
            unyt.cm / unyt.s,
            dm_velocity_name,
        ),
        ("Mass", dark_matter.mass, "mass_g", unyt.g, "dark_matter_mass_code"),
        (
            "SpecificAngularMomentum",
            dark_matter.angular_momentum,
            "specific_angular_momentum",
            unyt.cm**2 / unyt.s,
            dm_angular_name,
        ),
    )
    for name, values, scale_key, default_unit, field_name in fields:
        _write_quantity(
            dmdata,
            name,
            values,
            code_units=code_units,
            scale_key=scale_key,
            default_unit=default_unit,
            field_spec_obj=_runtime_field_spec(field_name, ric.par, code_units, output_time),
        )
    dmdata.attrs["Softening"] = _header_attr_value(dark_matter.softening * code_units.length_unit)


def _write_primary_fluid_snapshot_data(
    gdata,
    ric,
    code_units,
    output_time,
    cosmological_schema,
    boundary_runtime_code,
    density_runtime_code,
    velocity_runtime_code,
    temperature_runtime_code,
):
    """Write the mesh coordinates and primary fluid runtime fields."""
    boundary_name = "boundary_comoving_code" if cosmological_schema else "boundary_proper_code"
    density_name = "rho_comoving_code" if cosmological_schema else "rho_proper_code"
    velocity_name = "vel_supercomoving_code" if cosmological_schema else "vel_proper_code"
    temperature_name = "temp_supercomoving_code" if cosmological_schema else "temp_proper_code"
    scale_factor = getattr(ric.par, "supercomoving_coordinates", False)
    representation = "comoving" if cosmological_schema else "proper"
    coordinate_frame = "comoving" if cosmological_schema else "physical"

    _write_quantity(
        gdata,
        boundary_name,
        boundary_runtime_code,
        code_units=code_units,
        scale_key="length_cgs_cm",
        default_unit=unyt.cm,
        metadata={
            "quantity": "radius",
            "coordinate_frame": coordinate_frame,
            "representation": representation,
            "physical_relation": "physical = a * stored" if scale_factor else "physical = stored",
        },
        field_spec_obj=_runtime_field_spec(
            boundary_name,
            ric.par,
            code_units,
            output_time,
        ),
    )
    _write_quantity(
        gdata,
        density_name,
        density_runtime_code,
        code_units=code_units,
        scale_key="density_cgs_g_cm3",
        default_unit=unyt.g / unyt.cm**3,
        metadata={
            "quantity": "mass_density",
            "representation": representation,
            "scale_factor_power": 3.0 if scale_factor else 0.0,
            "physical_relation": (
                "physical = stored / a**3" if scale_factor else "physical = stored"
            ),
        },
        field_spec_obj=_runtime_field_spec(
            density_name,
            ric.par,
            code_units,
            output_time,
        ),
    )
    _write_quantity(
        gdata,
        velocity_name,
        velocity_runtime_code,
        code_units=code_units,
        scale_key="velocity_cgs_cm_s",
        default_unit=unyt.cm / unyt.s,
        metadata={
            "quantity": "velocity",
            "representation": "supercomoving" if cosmological_schema else "proper",
            "physical_relation": (
                "physical = H*a*x + stored/a" if scale_factor else "physical = stored"
            ),
        },
        field_spec_obj=_runtime_field_spec(
            velocity_name,
            ric.par,
            code_units,
            output_time,
        ),
    )
    _write_quantity(
        gdata,
        temperature_name,
        temperature_runtime_code,
        code_units=code_units,
        scale_key="temperature_cgs_K",
        default_unit=unyt.K,
        metadata={
            "quantity": "temperature",
            "representation": "supercomoving" if cosmological_schema else "proper",
            "scale_factor_power": (
                3.0 * (ric.par.hydrodynamics.gamma - 1.0) if scale_factor else 0.0
            ),
        },
        field_spec_obj=_runtime_field_spec(
            temperature_name,
            ric.par,
            code_units,
            output_time,
        ),
    )


def _write_optional_conserved_fluid_data(gdata, ric, code_units):
    """Write optional conserved fluid fields."""
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
        if not hasattr(ric.fluid, attr):
            continue
        scale_key = {
            "Mass_code": "mass_g",
            "AngularMomentum_code": "angular_momentum",
        }.get(attr, "energy_cgs_erg")
        default_unit = {
            "Mass_code": unyt.g,
            "AngularMomentum_code": unyt.g * unyt.cm**2 / unyt.s,
        }.get(attr, unyt.erg)
        _write_quantity(
            gdata,
            dataset_name,
            getattr(ric.fluid, attr),
            code_units=code_units,
            scale_key=scale_key,
            default_unit=default_unit,
        )


def _write_fluid_composition_data(gdata, ric, code_units):
    """Write composition and radiation fields attached to the fluid state."""
    gdata.create_dataset("mu", data=np.asarray(ric.fluid.mu))
    if hasattr(ric.fluid, "xHI"):
        gdata.create_dataset("xHI", data=np.asarray(ric.fluid.xHI))
    for attr in ("xHeI", "xHeII", "xHeIII"):
        if hasattr(ric.fluid, attr):
            gdata.create_dataset(attr, data=np.asarray(getattr(ric.fluid, attr)))
    if hasattr(ric.fluid, "ngamma_code"):
        ngamma_cgs_cm3 = ric.fluid.ngamma_code
        if hasattr(ngamma_cgs_cm3, "to_value") and code_units is not None:
            ngamma_cgs_cm3 = np.asarray(
                ngamma_cgs_cm3.to_value(code_units.number_density_unit),
            )
        _write_quantity(
            gdata,
            "ngamma_code",
            ngamma_cgs_cm3,
            code_units=code_units,
            scale_key="number_density_cgs_cm3",
            default_unit=1.0 / unyt.cm**3,
        )


def _mark_physical_fluid_datasets(gdata):
    """Mark optional datasets as physical in cosmological snapshots."""
    excluded = {
        "boundary_comoving_code",
        "rho_comoving_code",
        "vel_supercomoving_code",
        "temp_supercomoving_code",
    }
    for dataset_name, dataset in gdata.items():
        if isinstance(dataset, h5py.Dataset) and dataset_name not in excluded:
            dataset.attrs["representation"] = "physical"


def _write_optional_fluid_snapshot_data(gdata, ric, code_units):
    """Write optional conserved, composition, radiation, and metadata fields."""
    _write_optional_conserved_fluid_data(gdata, ric, code_units)
    _write_fluid_composition_data(gdata, ric, code_units)
    if getattr(ric.par, "cosmological_expansion", False):
        _mark_physical_fluid_datasets(gdata)


def _write_fluid_snapshot_data(
    gdata,
    ric,
    code_units,
    output_time,
    cosmological_schema,
    boundary_runtime_code,
    density_runtime_code,
    velocity_runtime_code,
    temperature_runtime_code,
):
    """Write all mesh and fluid datasets in a snapshot."""
    _write_primary_fluid_snapshot_data(
        gdata,
        ric,
        code_units,
        output_time,
        cosmological_schema,
        boundary_runtime_code,
        density_runtime_code,
        velocity_runtime_code,
        temperature_runtime_code,
    )
    _write_optional_fluid_snapshot_data(gdata, ric, code_units)


def write_snapshot_hdf5(ric, ICfilename, *, provenance=None):  # noqa: N803
    """Write an already-prepared runtime state to an HDF5 snapshot.

    The output file contains a ``Header`` group for metadata and a ``Data``
    group for mesh and fluid arrays. Units are stored as HDF5 attributes.
    """
    ICfilename = str(ICfilename)
    cosmological_schema = bool(
        getattr(ric.par, "cosmological_expansion", False)
        and getattr(ric.par, "supercomoving_coordinates", False),
    )
    log_diagnostic(
        logging.INFO,
        "hdf5_write",
        filename=ICfilename,
        representation=("supercomoving" if cosmological_schema else "proper"),
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
    with h5py.File(ICfilename, "w") as fic:
        code_units = getattr(getattr(ric.par, "units", None), "CodeUnits", None)
        # saving initial condition
        # first, save header:
        header = fic.create_group("Header")
        _write_runtime_header_attributes(header, ric, cosmological_schema)
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
            getattr(ric.par, "simulation", None),
            "coordinate_system",
            "cartesian",
        )
        _write_provenance(
            header,
            provenance
            if provenance is not None
            else getattr(ric, "provenance", None) or getattr(ric.par, "provenance", None),
        )
        if hasattr(ric, "cumulative_hydro_boundary_energy"):
            header.attrs["CumulativeHydroBoundaryEnergyCode"] = float(
                ric.cumulative_hydro_boundary_energy,
            )
        if hasattr(ric, "cumulative_gravity_work"):
            header.attrs["CumulativeGravityWorkCode"] = float(
                ric.cumulative_gravity_work,
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
                "coordinate_frame": ("comoving" if cosmological_schema else "physical"),
                "representation": ("comoving" if cosmological_schema else "proper"),
                "physical_relation": (
                    "physical = a * stored"
                    if getattr(ric.par, "supercomoving_coordinates", False)
                    else "physical = stored"
                ),
            },
        )

        # second, save mesh and fluid data:
        gdata = fic.create_group("Data")
        _write_fluid_snapshot_data(
            gdata,
            ric,
            code_units,
            output_time,
            cosmological_schema,
            boundary_runtime_code,
            density_runtime_code,
            velocity_runtime_code,
            temperature_runtime_code,
        )
        _write_dark_matter_snapshot(
            fic,
            ric,
            code_units,
            output_time,
            cosmological_schema,
        )

    if not hasattr(ric, "solver") and Path(ICfilename).stem.lower() == "initialcondition":
        update_used_parameters_yaml(
            Path.cwd() / "used_parameters.yaml",
            initial_condition=vars(ric.par),
        )


def writehdf5(ric, ICfilename, *, provenance=None):  # noqa: N803
    """Prepare and write an initial-condition HDF5 file.

    Representation-aware IC values are prepared by the shared
    :class:`InitialConditionWriter` boundary before serialization.
    """
    from radhydropy.initial_condition_writer import InitialConditionWriter  # noqa: PLC0415

    return InitialConditionWriter.from_rsim(
        ric,
        provenance=provenance,
    ).write(ICfilename)


def _restore_hdf5_scalar_attributes(par, header):
    """Restore scalar header attributes and nested mesh metadata."""
    for key, value in header.attrs.items():
        restored = _restore_header_attr_value(value)
        if key == "CodeUnits":
            if isinstance(restored, dict):
                restored = CodeUnits.from_mapping(restored)
            if hasattr(par, "set_code_units"):
                par.set_code_units(restored)
            else:
                par.CodeUnits = restored
            continue
        setattr(par, key, restored)
    if hasattr(par, "mesh"):
        if "GridCells" in header.attrs:
            par.mesh.grid_cells = int(header.attrs["GridCells"])
        if "GhostCells" in header.attrs:
            par.mesh.ghost_cells = int(header.attrs["GhostCells"])
    if hasattr(par, "simulation") and "CoordinateSystem" in header.attrs:
        par.simulation.coordinate_system = header.attrs["CoordinateSystem"]


def _restore_hdf5_header_attributes(par, header):
    """Restore code units, scalar attributes, and cosmology metadata."""
    if "CodeUnits" not in header.attrs:
        raise ValueError(
            "IC file is missing Header.attrs['CodeUnits']; "
            "cannot read datasets without a code-unit mapping.",
        )
    code_units = _restore_header_attr_value(header.attrs["CodeUnits"])
    if isinstance(code_units, dict):
        code_units = CodeUnits.from_mapping(code_units)
    if not isinstance(code_units, CodeUnits):
        raise ValueError("IC file Header.attrs['CodeUnits'] is not a valid CodeUnits mapping.")
    _validate_snapshot_configuration(par, header, code_units)
    file_provenance = _read_provenance(header)
    if file_provenance is not None:
        par.provenance = file_provenance
    _restore_hdf5_scalar_attributes(par, header)
    _restore_cosmology_from_header(par, header, code_units)
    _restore_cosmology_context_from_header(par, header)
    return code_units


def _identify_hdf5_schema(par, header, expected_coordsys, expected_nogrid):
    """Validate the canonical header schema and return its representation."""
    coordinate_system = par.simulation.coordinate_system
    if expected_coordsys is not None and coordinate_system != expected_coordsys:
        raise Exception(
            f"Coordinate systems in IC ({coordinate_system}) and run "
            f"({expected_coordsys}) do not agree!",
        )
    grid_cells = par.mesh.grid_cells
    if expected_nogrid is not None and grid_cells != expected_nogrid:
        raise Exception(
            f"Number of grids in IC ({grid_cells}) and run ({expected_nogrid}) do not agree!",
        )
    generic_names = _GENERIC_HEADER_DATASETS.intersection(
        header.keys(),
    ) or _GENERIC_HEADER_DATASETS.intersection(header.attrs.keys())
    if generic_names:
        names = ", ".join(sorted(generic_names))
        raise ValueError(
            f"HDF5 header uses unsupported generic field name(s): {names}; "
            "use the representation-specific schema",
        )
    cosmological = "tau_supercomoving_code" in header or "box_size_comoving_code" in header
    proper = "time_proper_code" in header or "box_size_proper_code" in header
    if cosmological and proper:
        raise ValueError("HDF5 header mixes cosmological and proper schemas")
    if not cosmological and not proper:
        raise ValueError(
            "HDF5 header has no canonical representation-specific time and box-size datasets",
        )
    required = (
        {"tau_supercomoving_code", "box_size_comoving_code"}
        if cosmological
        else {"time_proper_code", "box_size_proper_code"}
    )
    missing = required.difference(header.keys())
    if missing:
        raise ValueError(
            f"HDF5 header is missing canonical dataset(s): {', '.join(sorted(missing))}",
        )
    return cosmological


def _synchronize_hdf5_header(par, fluid, header, code_units, cosmological):
    """Populate typed header values and synchronize the restored runtime clock."""
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
    if cosmological:
        par.tau_supercomoving_code = np.asarray(par.tau_supercomoving_code, dtype=float)
        cosmic_time = header.attrs.get("time_cosmic_code", header.attrs.get("CosmicTime"))
        if cosmic_time is not None:
            par.time_cosmic_code = float(_restore_header_attr_value(cosmic_time))
    if hasattr(par, "sync_simulation_parameters"):
        par.sync_simulation_parameters()
    if hasattr(par, "sync_mesh_parameters"):
        par.sync_mesh_parameters()
    if hasattr(par, "load_radiation_spectrum"):
        par.load_radiation_spectrum(par.output.directory)
    time_field = "tau_supercomoving_code" if cosmological else "time_proper_code"
    box_field = "box_size_comoving_code" if cosmological else "box_size_proper_code"
    _populate_group_targets(
        header,
        (par,),
        code_units=code_units,
        scale_map={time_field: "time_s", box_field: "length_cgs_cm"},
    )
    runtime_time = getattr(par, time_field)
    if hasattr(runtime_time, "to_value"):
        runtime_time = float(np.asarray(runtime_time.to_value(code_units.time_unit)))
    else:
        runtime_time = float(np.asarray(runtime_time, dtype=float).reshape(-1)[0])
    if hasattr(par, "simulation"):
        setattr(par.simulation, time_field, runtime_time)
        setattr(par.simulation, box_field, getattr(par, box_field))
    setattr(fluid, time_field, runtime_time)


def _read_hdf5_header(par, fluid, fic, expected_coordsys, expected_nogrid):
    """Restore header state and determine the canonical file schema."""
    header = fic["Header"]
    code_units = _restore_hdf5_header_attributes(par, header)
    cosmological = _identify_hdf5_schema(par, header, expected_coordsys, expected_nogrid)
    _synchronize_hdf5_header(par, fluid, header, code_units, cosmological)
    return header, code_units, cosmological


def _read_hdf5_data(par, mesh, fluid, fic, header, code_units, canonical_cosmological_schema):
    """Restore canonical datasets, runtime views, and optional dark matter."""
    gdata = fic["Data"]
    generic_data_names = _GENERIC_PRIMITIVE_DATASETS.intersection(gdata.keys())
    if generic_data_names:
        names = ", ".join(sorted(generic_data_names))
        raise ValueError(
            f"HDF5 data uses unsupported generic field name(s): {names}; "
            "use the representation-specific schema",
        )
    data_scale_map = {
        "boundary_proper_code": "length_cgs_cm",
        "boundary_comoving_code": "length_cgs_cm",
        "rho_proper_code": "density_cgs_g_cm3",
        "rho_comoving_code": "density_cgs_g_cm3",
        "vel_proper_code": "velocity_cgs_cm_s",
        "vel_supercomoving_code": "velocity_cgs_cm_s",
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
    _populate_group_targets(gdata, (mesh, fluid), code_units=code_units, scale_map=data_scale_map)
    snapshot_cosmology = getattr(par, "cosmology_context", None)
    schema = "cosmological" if canonical_cosmological_schema else "proper"
    _attach_radarray_views(
        gdata,
        mesh,
        gdata,
        schema,
        code_units,
        snapshot_cosmology,
        allowed_names={"boundary_proper_code", "boundary_comoving_code"},
    )
    _attach_radarray_views(
        gdata,
        fluid,
        gdata,
        schema,
        code_units,
        snapshot_cosmology,
        allowed_names={
            "rho_proper_code",
            "rho_comoving_code",
            "vel_proper_code",
            "vel_supercomoving_code",
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
    _restore_runtime_state(par, fluid, gdata, canonical_cosmological_schema)
    if canonical_cosmological_schema:
        _validate_cosmological_data(gdata)
    _restore_field_metadata(par, gdata)
    if "DarkMatter" in fic:
        _restore_dark_matter_snapshot(par, fic["DarkMatter"], code_units, schema)


def _restore_runtime_state(par, fluid, gdata, canonical_cosmological_schema):
    """Build the typed runtime state after field restoration."""
    if canonical_cosmological_schema:
        fluid.runtime_fields = SUPERCOMOVING_RUNTIME_FIELDS
        fluid.runtime_state = FluidRuntimeState.from_arrays(
            SUPERCOMOVING_RUNTIME_FIELDS,
            rho_comoving_code=fluid.rho_comoving_code,
            vel_supercomoving_code=fluid.vel_supercomoving_code,
            pre_supercomoving_code=getattr(
                fluid,
                "pre_supercomoving_code",
                np.zeros_like(fluid.rho_comoving_code),
            ),
            temp_supercomoving_code=fluid.temp_supercomoving_code,
            tau_supercomoving_code=getattr(fluid, "tau_supercomoving_code", 0.0),
            mu_dimensionless=getattr(fluid, "mu", None),
            xHI_dimensionless=getattr(fluid, "xHI", None),
        )
        return
    fluid.runtime_fields = PROPER_RUNTIME_FIELDS
    fluid.runtime_state = FluidRuntimeState.from_arrays(
        PROPER_RUNTIME_FIELDS,
        rho_proper_code=fluid.rho_proper_code,
        vel_proper_code=fluid.vel_proper_code,
        pre_proper_code=getattr(fluid, "pre_proper_code", np.zeros_like(fluid.rho_proper_code)),
        temp_proper_code=fluid.temp_proper_code,
        time_proper_code=getattr(fluid, "time_proper_code", 0.0),
        mu_dimensionless=getattr(fluid, "mu", None),
        xHI_dimensionless=getattr(fluid, "xHI", None),
    )


def _validate_cosmological_data(gdata):
    """Require the primary fields in a cosmological restart."""
    if "boundary_comoving_code" not in gdata:
        raise ValueError("canonical cosmological HDF5 file is missing Data/boundary_comoving_code")
    for name in ("rho_comoving_code", "vel_supercomoving_code", "temp_supercomoving_code"):
        if name not in gdata:
            raise ValueError(f"canonical cosmological HDF5 file is missing Data/{name}")


def _restore_field_metadata(par, gdata):
    """Restore dataset metadata without exposing HDF5 objects."""
    if hasattr(par, "code_state"):
        _ = par.code_state
    par.field_metadata = {
        dataset_name: {
            key: _restore_header_attr_value(value)
            for key, value in dataset.attrs.items()
            if key != "units"
        }
        for dataset_name, dataset in gdata.items()
        if isinstance(dataset, h5py.Dataset)
    }


def _restore_dark_matter_snapshot(par, dmdata, code_units, schema):
    """Restore optional dark-matter shell and analysis state."""
    dm_scale_map = {
        "Radius": "length_cgs_cm",
        "RadialVelocity": "velocity_cgs_cm_s",
        "Mass": "mass_g",
        "SpecificAngularMomentum": "specific_angular_momentum",
    }
    _populate_group_targets(dmdata, (par,), code_units=code_units, scale_map=dm_scale_map)
    snapshot = {
        "radius": par.Radius,
        "velocity": par.RadialVelocity,
        "mass": par.Mass,
        "angular_momentum": par.SpecificAngularMomentum,
        "softening": _restore_header_attr_value(dmdata.attrs.get("Softening", 0.0)),
    }
    par.dark_matter_snapshot = snapshot
    par.dark_matter = DarkMatterShells(**snapshot, code_units=code_units)
    par.dark_matter_radarrays = _attach_dark_matter_radarray_views(
        dmdata,
        par,
        code_units,
        getattr(par, "cosmology_context", None),
        schema,
    )


def readhdf5(par, mesh, fluid, ICfilename):  # noqa: N803
    """Read a RadHydropy HDF5 file into parameter, mesh, and fluid objects.

    Canonical representation-specific datasets are restored into the runtime
    code-unit system when ``CodeUnits`` is available in the file header.
    """
    ICfilename = str(ICfilename)
    log_diagnostic(logging.INFO, "hdf5_read", filename=ICfilename)
    with h5py.File(ICfilename, "r") as fic:
        expected_coordsys = par.simulation.coordinate_system
        expected_nogrid = getattr(getattr(par, "mesh", None), "grid_cells", None)
        header, code_units, canonical_cosmological_schema = _read_hdf5_header(
            par,
            fluid,
            fic,
            expected_coordsys,
            expected_nogrid,
        )
        _read_hdf5_data(par, mesh, fluid, fic, header, code_units, canonical_cosmological_schema)
        return


def loadhdf5(config, ICfilename):  # noqa: N803
    """Construct and load an ``Rsim`` from a nested configuration.

    Parameters
    ----------
    config : mapping
        Complete nested example configuration containing at least ``par``.
        The loader compares the file header with the runtime coordinate system,
        grid size, code units, cosmology, and field representations.
    ICfilename : path-like
        Initial-condition or snapshot HDF5 file to load.

    Returns
    -------
    radhydropy.rsim.Rsim
        Runtime object containing restored ``par``, ``mesh``, and ``fluid``
        state. Dimensional analysis should use the typed ``*_radarray`` views
        on ``mesh``, ``fluid``, and the optional ``dark_matter`` component.

    Notes
    -----
    The file header supplies the authoritative code-unit and representation
    metadata. Proper-coordinate files restore fields such as
    ``rho_proper_code``; cosmological files restore fields such as
    ``rho_comoving_code`` and ``vel_supercomoving_code``. Chemistry,
    radiation, provenance, and live dark-matter state are restored when those
    groups are present in the file. Live shell solver state remains available
    through ``snapshot.par.dark_matter``; the typed analysis view is exposed
    as ``snapshot.dark_matter``.

    """
    if not hasattr(config, "__getitem__"):
        raise TypeError("loadhdf5 expects a nested configuration mapping")
    try:
        par_config = config["par"]
    except (KeyError, TypeError) as exc:
        raise ValueError("loadhdf5 configuration must contain a 'par' mapping") from exc
    if not hasattr(par_config, "items"):
        raise TypeError("loadhdf5 config['par'] must be a mapping")

    # Import locally to keep the I/O module independent from Rsim's import
    # path during package initialization.
    from radhydropy.rsim import Rsim  # noqa: PLC0415

    restored = Rsim(par_config)
    # Resolve through the package façade so callers that replace the public
    # reader for diagnostics/tests observe the same behavior as before the
    # io.py -> io/ package migration.
    from radhydropy import io as public_io  # noqa: PLC0415

    public_io.readhdf5(
        restored.par,
        restored.mesh,
        restored.fluid,
        ICfilename,
    )
    restored.dark_matter = getattr(
        restored.par,
        "dark_matter_radarrays",
        None,
    )
    return restored
