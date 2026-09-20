"""HDF5 input and output helpers for simulations."""

from pathlib import Path

import h5py
import os
import unyt
import numpy as np
import radhydropy.utils as ru
from radhydropy.units import CodeUnits, _code_units
from radhydropy.dark_matter import DarkMatterShells, DarkMatterSnapshot
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    PROPER_RUNTIME_FIELDS,
    SUPERCOMOVING_RUNTIME_FIELDS,
)
from radhydropy.field_metadata import FieldSpec, field_spec
from radhydropy.radarray import RadArray, RadQuantity
from radhydropy.io.metadata import (
    _header_attr_value,
    _provenance_yaml_text,
    _read_provenance,
    _restore_header_attr_value,
    _write_provenance,
    parameter_tree,
    update_used_parameters_yaml,
    write_used_parameters,
)
from radhydropy.io.fields import (
    normalize_attr_name,
    populate_group_targets,
    read_any_dataset,
    scale_unit_for_key,
    write_quantity,
)
from radhydropy.io.cosmology import (
    restore_cosmology_context_from_header,
    restore_cosmology_from_header,
    runtime_field_spec,
    write_cosmology_header,
)

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


_GENERIC_HEADER_DATASETS = frozenset({"time_code", "box_size_code"})
_GENERIC_PRIMITIVE_DATASETS = frozenset(
    {"boundary", "rho_code", "vel_code", "temp_code", "pre_code"}
)


class SnapshotConfigurationError(ValueError):
    """Raised when a snapshot is incompatible with the supplied runtime."""


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
        header_cosmology = str(header_cosmology)
        expected_cosmology = str(expected_cosmology)
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
    if (
        expected_representation is not None
        and restored.representation != expected_representation
    ):
        raise ValueError(
            f"{dataset.name!r} uses representation "
            f"{restored.representation!r}; expected "
            f"{expected_representation!r} for {canonical_name!r}"
        )
    return restored


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
                field_name=canonical_name,
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
                field_name=dataset_name,
            ),
        )


def _attach_dark_matter_radarray_views(
    group, par, code_units, cosmology, canonical_schema
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
                f"DarkMatter group is missing required dataset {dataset_name!r}"
            )
        dataset = group[dataset_name]
        spec = _radarray_field_spec(
            dataset, canonical_name, code_units, cosmology
        )
        views[dataset_name] = RadArray(
            np.asarray(getattr(par, dataset_name), dtype=float),
            code_units=code_units,
            field_spec=spec,
            cosmology=cosmology,
            field_name=canonical_name,
        )

    softening_runtime_code = _restore_header_attr_value(
        group.attrs.get("Softening", 0.0)
    )
    if hasattr(softening_runtime_code, "to_value"):
        softening_runtime_code = float(
            np.asarray(softening_runtime_code.to_value(code_units.length_unit))
        )
    else:
        softening_runtime_code = float(softening_runtime_code)
    softening_field_name = (
        "radius_comoving_code"
        if canonical_schema == "cosmological"
        else "radius_proper_code"
    )
    softening_spec = _radarray_field_spec(
        group["Radius"], softening_field_name, code_units, cosmology
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
                "dark_matter_radarrays",
                "dual_energy_config", "positivity", "radiation", "units",
            }:
                continue
            if key in {"time_code", "box_size_code"}:
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
                "coordinate_frame": (
                    "comoving" if cosmological_schema else "physical"
                ),
                "representation": (
                    "comoving" if cosmological_schema else "proper"
                ),
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
                "coordinate_frame": (
                    "comoving" if cosmological_schema else "physical"
                ),
                "representation": (
                    "comoving" if cosmological_schema else "proper"
                ),
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
                "representation": (
                    "comoving" if cosmological_schema else "proper"
                ),
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
                "representation": (
                    "supercomoving" if cosmological_schema else "proper"
                ),
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
                    "supercomoving" if cosmological_schema else "proper"
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
            dm_radius_name = (
                "radius_comoving_code"
                if cosmological_schema
                else "radius_proper_code"
            )
            dm_velocity_name = (
                "vel_supercomoving_code"
                if cosmological_schema
                else "vel_proper_code"
            )
            dm_angular_momentum_name = (
                "specific_angular_momentum_supercomoving_code"
                if cosmological_schema
                else "specific_angular_momentum_proper_code"
            )
            _write_quantity(dmdata, "Radius", dark_matter.radius,
                            code_units=code_units, scale_key="length_cgs_cm",
                            default_unit=unyt.cm,
                            field_spec_obj=_runtime_field_spec(
                                dm_radius_name, ric.par, code_units, output_time
                            ))
            _write_quantity(dmdata, "RadialVelocity", dark_matter.velocity,
                            code_units=code_units, scale_key="velocity_cgs_cm_s",
                            default_unit=unyt.cm / unyt.s,
                            field_spec_obj=_runtime_field_spec(
                                dm_velocity_name, ric.par, code_units, output_time
                            ))
            _write_quantity(dmdata, "Mass", dark_matter.mass,
                            code_units=code_units, scale_key="mass_g",
                            default_unit=unyt.g,
                            field_spec_obj=_runtime_field_spec(
                                "dark_matter_mass_code",
                                ric.par,
                                code_units,
                                output_time,
                            ))
            _write_quantity(dmdata, "SpecificAngularMomentum", dark_matter.angular_momentum,
                            code_units=code_units, scale_key="specific_angular_momentum",
                            default_unit=unyt.cm**2 / unyt.s,
                            field_spec_obj=_runtime_field_spec(
                                dm_angular_momentum_name,
                                ric.par,
                                code_units,
                                output_time,
                            ))
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

    Canonical representation-specific datasets are restored into the runtime
    code-unit system when ``CodeUnits`` is available in the file header.
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
        gdata = fic["Data"]
        generic_header_names = (
            _GENERIC_HEADER_DATASETS.intersection(header.keys())
            or _GENERIC_HEADER_DATASETS.intersection(header.attrs.keys())
        )
        if generic_header_names:
            names = ", ".join(sorted(generic_header_names))
            raise ValueError(
                f"HDF5 header uses unsupported generic field name(s): {names}; "
                "use the representation-specific schema"
            )
        generic_data_names = _GENERIC_PRIMITIVE_DATASETS.intersection(gdata.keys())
        if generic_data_names:
            names = ", ".join(sorted(generic_data_names))
            raise ValueError(
                f"HDF5 data uses unsupported generic field name(s): {names}; "
                "use the representation-specific schema"
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
        if not canonical_cosmological_schema and not canonical_proper_schema:
            raise ValueError(
                "HDF5 header has no canonical representation-specific time and "
                "box-size datasets"
            )
        if canonical_cosmological_schema:
            required_header_names = {
                "tau_supercomoving_code", "box_size_comoving_code"
            }
        else:
            required_header_names = {
                "time_proper_code", "box_size_proper_code"
            }
        missing_header_names = required_header_names.difference(header.keys())
        if missing_header_names:
            names = ", ".join(sorted(missing_header_names))
            raise ValueError(f"HDF5 header is missing canonical dataset(s): {names}")
        header_scale_map = {
            "tau_supercomoving_code" if canonical_cosmological_schema else "time_proper_code": "time_s",
            "box_size_comoving_code" if canonical_cosmological_schema else "box_size_proper_code": "length_cgs_cm",
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
                else "time_proper_code"
            )
            box_field = (
                "box_size_comoving_code"
                if canonical_cosmological_schema
                else "box_size_proper_code"
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
            setattr(par.simulation, time_field, runtime_time)
            if canonical_cosmological_schema:
                par.simulation.box_size_comoving_code = getattr(par, box_field)
            else:
                par.simulation.box_size_proper_code = getattr(par, box_field)
            if canonical_cosmological_schema:
                fluid.tau_supercomoving_code = runtime_time
            else:
                fluid.time_proper_code = runtime_time
        else:
            time_field = (
                "tau_supercomoving_code"
                if canonical_cosmological_schema
                else "time_proper_code"
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
            else:
                fluid.time_proper_code = runtime_time

        #second, save mesh and fluid data:
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
            par.dark_matter_radarrays = _attach_dark_matter_radarray_views(
                dmdata,
                par,
                code_units,
                getattr(par, "cosmology_context", None),
                "cosmological" if canonical_cosmological_schema else "proper",
            )


def loadhdf5(config, ICfilename):
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
    from radhydropy.rsim import Rsim

    restored = Rsim(par_config)
    # Resolve through the package façade so callers that replace the public
    # reader for diagnostics/tests observe the same behavior as before the
    # io.py -> io/ package migration.
    from radhydropy import io as public_io

    public_io.readhdf5(
        restored.par,
        restored.mesh,
        restored.fluid,
        ICfilename,
    )
    restored.dark_matter = getattr(
        restored.par, "dark_matter_radarrays", None
    )
    return restored
