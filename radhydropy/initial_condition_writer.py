# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Shared RadArray-to-runtime initial-condition writer boundary."""

import warnings

import numpy as np

from radhydropy.cosmology.context import CosmologyContext
from radhydropy.field_metadata import field_spec
from radhydropy.radarray import RadArray, RadQuantity, _code_unit_for_spec
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import (
    PROPER_RUNTIME_FIELDS,
    SUPERCOMOVING_RUNTIME_FIELDS,
    FluidRuntimeState,
    MeshGeometryState,
)


class _InitialConditionFieldGroup:
    """Small typed assignment facade used by ``InitialConditionWriter``."""

    def __init__(self, writer, allowed_fields):
        object.__setattr__(self, "_writer", writer)
        object.__setattr__(self, "_allowed_fields", frozenset(allowed_fields))

    def __setattr__(self, field_name, value):
        if field_name.startswith("_"):
            object.__setattr__(self, field_name, value)
            return
        if field_name == "eos" and "eos" not in self._allowed_fields:
            raise AttributeError(f"unsupported initial-condition field {field_name!r}")
        if field_name not in self._allowed_fields:
            raise AttributeError(f"unsupported initial-condition field {field_name!r}")
        self._writer.set_field(field_name, value)

    def __getattr__(self, field_name):
        if field_name in self._allowed_fields:
            try:
                return self._writer._fields[field_name]
            except KeyError:
                pass
        raise AttributeError(field_name)


class InitialConditionWriter:
    """Prepare representation-aware IC fields before shared HDF5 writing.

    Builders may assign ``RadArray`` values through :meth:`set_field`.  The
    writer converts them to the configured runtime representation and creates
    the plain numerical typed states required by the solver and HDF5 layer.
    """

    def __init__(
        self,
        simulation=None,
        *,
        par_config=None,
        code_units=None,
        ic_config=None,
        box_size=None,
        cosmology_context=None,
        scale_factor=None,
        hubble_parameter_km_s_Mpc=None,  # noqa: N803
        provenance=None,
    ):
        if simulation is None:
            simulation = self._simulation_from_config(par_config)
        self.simulation = simulation
        cosmology_parameters = getattr(simulation.par, "cosmology", None)
        cosmological = getattr(
            cosmology_parameters,
            "cosmological",
            getattr(
                simulation.par,
                "cosmological_gravity",
                getattr(simulation.par, "supercomoving_coordinates", False),
            ),
        )
        supercomoving_coordinates = getattr(
            cosmology_parameters,
            "supercomoving_coordinates",
            getattr(simulation.par, "supercomoving_coordinates", False),
        )
        if bool(cosmological) != bool(supercomoving_coordinates):
            raise ValueError(
                "InitialConditionWriter requires cosmological and "
                "supercomoving_coordinates to have matching values; got "
                f"cosmological={bool(cosmological)}, "
                f"supercomoving_coordinates={bool(supercomoving_coordinates)}",
            )
        self.code_units = simulation.par.units.CodeUnits
        self.ic_config = ic_config
        self.provenance = provenance
        self.box_size = box_size
        self._fields = {}
        if code_units is not None:
            actual_code_units = self.code_units
            for name in (
                "mass_in_cgs",
                "length_in_cgs",
                "velocity_in_cgs",
                "current_in_cgs",
                "temperature_in_cgs",
            ):
                if not np.isclose(
                    getattr(code_units, name),
                    getattr(actual_code_units, name),
                    rtol=1e-12,
                    atol=0.0,
                ):
                    raise ValueError(
                        f"writer code-unit {name} does not match par_config",
                    )
        if cosmology_context is not None:
            simulation.par.cosmology_context = cosmology_context
        elif scale_factor is not None:
            gamma = float(simulation.par.hydrodynamics.gamma)
            hydrodynamics = simulation.par.hydrodynamics
            cosmology = getattr(simulation.par, "cosmology", None)
            simulation.par.cosmology_context = CosmologyContext(
                gamma=gamma,
                cosmology=getattr(cosmology, "type_name", "proper"),
                isothermal=(getattr(hydrodynamics, "eos_type", "") == "isothermal"),
                scale_factor=float(scale_factor),
                hubble_parameter_km_s_Mpc=float(
                    hubble_parameter_km_s_Mpc or 0.0,
                ),
            )
        elif cosmological and supercomoving_coordinates and ic_config is not None:
            cosmic_time = ic_config.get("time_cosmic")
            cosmology_model = getattr(cosmology_parameters, "model", None)
            if cosmic_time is not None and cosmology_model is not None:
                cosmic_time_code = float(
                    np.asarray(
                        self._to_code_values(cosmic_time, self.code_units.time_unit),
                    ).flat[0],
                )
                scale_factor_value = float(
                    cosmology_model.scale_factor(cosmic_time_code),
                )
                hubble_code = float(cosmology_model.hubble(cosmic_time_code))
                hubble_unit_km_s_Mpc = self.code_units.velocity_unit.to_value(
                    "km/s",
                ) / self.code_units.length_unit.to_value("Mpc")
                simulation.par.cosmology_context = CosmologyContext(
                    gamma=float(simulation.par.hydrodynamics.gamma),
                    cosmology=cosmology_model.type_name,
                    isothermal=(
                        getattr(simulation.par.hydrodynamics, "eos_type", "") == "isothermal"
                    ),
                    scale_factor=scale_factor_value,
                    hubble_parameter_km_s_Mpc=(hubble_code * hubble_unit_km_s_Mpc),
                )
        self.mesh = _InitialConditionFieldGroup(
            self,
            {
                "x_radarray",
                "boundary_radarray",
            },
        )
        self.fluid = _InitialConditionFieldGroup(
            self,
            {
                "rho_radarray",
                "vel_radarray",
                "pre_radarray",
                "temp_radarray",
                "ngamma_radarray",
                "specific_angular_momentum_radarray",
                "mu",
                "xHI",
                "eos",
            },
        )

    @staticmethod
    def _simulation_from_config(par_config):
        if par_config is None:
            raise TypeError(
                "InitialConditionWriter requires simulation or par_config",
            )
        return Rsim(par_config)

    @staticmethod
    def _cosmological_schema(simulation):
        """Return whether ICs use comoving/supercomoving representations."""
        par = simulation.par
        cosmology = getattr(par, "cosmology", None)
        cosmological = getattr(
            cosmology,
            "cosmological",
            getattr(
                par,
                "cosmological_gravity",
                getattr(par, "supercomoving_coordinates", False),
            ),
        )
        supercomoving = getattr(
            cosmology,
            "supercomoving_coordinates",
            getattr(par, "supercomoving_coordinates", False),
        )
        return bool(cosmological and supercomoving)

    @classmethod
    def from_rsim(cls, simulation, *, provenance=None):
        return cls(simulation, provenance=provenance)

    def _primitive_field_name(self, values, context, representation=None):
        if not hasattr(values, "units"):
            raise TypeError("writer.radarray requires a unit-bearing array")
        cosmological_schema = self._cosmological_schema(self.simulation)
        field_names = (
            (
                "boundary_comoving_code",
                "rho_comoving_code",
                "vel_supercomoving_code",
                "temp_supercomoving_code",
                "pre_supercomoving_code",
                "ngamma_comoving_code",
                "specific_angular_momentum_supercomoving_code",
            )
            if cosmological_schema
            else (
                "boundary_proper_code",
                "rho_proper_code",
                "vel_proper_code",
                "temp_proper_code",
                "pre_proper_code",
                "ngamma_proper_code",
                "specific_angular_momentum_proper_code",
            )
        )
        if representation is not None:
            representation_fields = {
                "proper": (
                    "boundary_proper_code",
                    "rho_proper_code",
                    "vel_proper_code",
                    "temp_proper_code",
                    "pre_proper_code",
                    "ngamma_proper_code",
                    "specific_angular_momentum_proper_code",
                ),
                "comoving": (
                    "boundary_comoving_code",
                    "rho_comoving_code",
                    "ngamma_comoving_code",
                ),
                "supercomoving": (
                    "vel_supercomoving_code",
                    "temp_supercomoving_code",
                    "pre_supercomoving_code",
                    "specific_angular_momentum_supercomoving_code",
                ),
            }
            if representation not in representation_fields:
                raise ValueError(
                    "writer.radarray representation must be proper, comoving, or supercomoving",
                )
            field_names = tuple(representation_fields[representation])
        field_units = (
            self.code_units.length_unit,
            self.code_units.density_unit,
            self.code_units.velocity_unit,
            self.code_units.temperature_unit,
            self.code_units.pressure_unit,
            self.code_units.number_density_unit,
            self.code_units.length_unit * self.code_units.velocity_unit,
        )
        field_units_by_name = {
            "boundary_proper_code": self.code_units.length_unit,
            "boundary_comoving_code": self.code_units.length_unit,
            "rho_proper_code": self.code_units.density_unit,
            "rho_comoving_code": self.code_units.density_unit,
            "vel_proper_code": self.code_units.velocity_unit,
            "vel_supercomoving_code": self.code_units.velocity_unit,
            "temp_proper_code": self.code_units.temperature_unit,
            "temp_supercomoving_code": self.code_units.temperature_unit,
            "pre_proper_code": self.code_units.pressure_unit,
            "pre_supercomoving_code": self.code_units.pressure_unit,
            "ngamma_proper_code": self.code_units.number_density_unit,
            "ngamma_comoving_code": self.code_units.number_density_unit,
            "specific_angular_momentum_proper_code": self.code_units.length_unit
            * self.code_units.velocity_unit,
            "specific_angular_momentum_supercomoving_code": self.code_units.length_unit
            * self.code_units.velocity_unit,
        }
        if representation is not None:
            field_units = tuple(field_units_by_name[name] for name in field_names)
        value_dimensions = getattr(values.units, "dimensions", None)
        if value_dimensions is None:
            value_dimensions = values.units.units.dimensions
        matching_fields = [
            field_name
            for field_name, field_unit in zip(field_names, field_units, strict=False)
            if value_dimensions == field_unit.units.dimensions
        ]
        if len(matching_fields) != 1:
            raise ValueError(
                "writer.radarray cannot infer a canonical primitive field from "
                f"unit dimensions {value_dimensions}",
            )
        return matching_fields[0]

    def radarray(self, values, *, field_name=None, representation="proper"):
        """Create a ``RadArray`` from a unit-bearing array.

        Primitive IC fields are inferred from their dimensions and the
        requested representation.  Proper representation is the default;
        cosmological callers should request ``comoving`` or
        ``supercomoving`` explicitly.  Auxiliary dimensional fields, such as
        specific angular momentum, may provide their canonical field name
        explicitly.
        """
        context = self._context(self.simulation)
        if context is None:
            raise ValueError(
                "InitialConditionWriter requires a valid cosmology context "
                "before creating a RadArray",
            )
        if representation in ("comoving", "supercomoving"):
            cosmology_parameters = getattr(self.simulation.par, "cosmology", None)
            missing_flags = []
            cosmological = getattr(
                cosmology_parameters,
                "cosmological",
                getattr(
                    self.simulation.par,
                    "cosmological_gravity",
                    getattr(
                        self.simulation.par,
                        "supercomoving_coordinates",
                        False,
                    ),
                ),
            )
            supercomoving_coordinates = getattr(
                cosmology_parameters,
                "supercomoving_coordinates",
                getattr(self.simulation.par, "supercomoving_coordinates", False),
            )
            if not cosmological:
                missing_flags.append("cosmological=True")
            if not supercomoving_coordinates:
                missing_flags.append("supercomoving_coordinates=True")
            if missing_flags:
                raise ValueError(
                    f"writer.radarray representation={representation!r} requires "
                    + " and ".join(missing_flags),
                )
        elif representation == "proper":
            cosmology_parameters = getattr(self.simulation.par, "cosmology", None)
            cosmological = getattr(
                cosmology_parameters,
                "cosmological",
                getattr(
                    self.simulation.par,
                    "cosmological_gravity",
                    getattr(
                        self.simulation.par,
                        "supercomoving_coordinates",
                        False,
                    ),
                ),
            )
            supercomoving_coordinates = getattr(
                cosmology_parameters,
                "supercomoving_coordinates",
                getattr(self.simulation.par, "supercomoving_coordinates", False),
            )
            if cosmological or supercomoving_coordinates:
                warnings.warn(
                    "writer.radarray representation='proper' is being used "
                    "with cosmological or supercomoving coordinates enabled",
                    UserWarning,
                    stacklevel=2,
                )
        if field_name is None:
            field_name = self._primitive_field_name(values, context, representation)
        hubble_parameter_km_s_Mpc = (
            context.hubble_parameter_km_s_Mpc if field_name == "vel_supercomoving_code" else None
        )
        spec = field_spec(
            field_name,
            self.code_units,
            cosmology=context.cosmology,
            scale_factor=context.scale_factor,
            hubble_parameter_km_s_Mpc=hubble_parameter_km_s_Mpc,
        )
        if representation is not None and spec.representation != representation:
            raise ValueError(
                f"field {field_name!r} has representation {spec.representation!r}, "
                f"not {representation!r}",
            )
        return RadArray(
            self._to_code_values(values, _code_unit_for_spec(self.code_units, spec)),
            code_units=self.code_units,
            field_spec=spec,
            cosmology=context,
        )

    @staticmethod
    def _to_code_values(value, code_unit):
        """Convert a unit-bearing value to numerical values in ``code_unit``."""
        if not hasattr(value, "to_value"):
            raise TypeError("writer requires a unit-bearing value")
        return np.asarray(
            value.to_value(code_unit.units),
            dtype=float,
        ) / float(code_unit.value)

    def radquantity(self, value):
        """Create a ``RadQuantity`` from a unit-bearing primitive scalar."""
        context = self._context(self.simulation)
        if context is None:
            raise ValueError(
                "InitialConditionWriter requires a valid cosmology context "
                "before creating a RadQuantity",
            )
        field_name = self._primitive_field_name(value, context)
        spec = field_spec(
            field_name,
            self.code_units,
            cosmology=context.cosmology,
            scale_factor=context.scale_factor,
            hubble_parameter_km_s_Mpc=(
                context.hubble_parameter_km_s_Mpc
                if field_name == "vel_supercomoving_code"
                else None
            ),
        )
        return RadQuantity(
            self._to_code_values(value, _code_unit_for_spec(self.code_units, spec)),
            code_units=self.code_units,
            field_spec=spec,
            cosmology=context,
        )

    def set_field(self, field_name, value):
        """Register one canonical IC field for conversion at write time."""
        self._fields[field_name] = value
        return self

    def _resolve_field_value(self, field_name, container, radarray_field, source_field):
        value = self._fields.get(field_name)
        if value is None and field_name in ("ngamma_proper_code", "ngamma_comoving_code"):
            value = getattr(container, "ngamma_code", None)
            if (
                value is not None
                and hasattr(value, "units")
                and not isinstance(
                    value,
                    (RadArray, RadQuantity),
                )
            ):
                value = None
        candidates = (
            value,
            getattr(container, field_name, None),
            self._fields.get(radarray_field) if radarray_field is not None else None,
            self._fields.get(source_field) if source_field is not None else None,
            getattr(container, f"{field_name}_radarray", None),
            getattr(container, radarray_field, None) if radarray_field is not None else None,
            (
                getattr(container, f"{source_field}_radarray", None)
                if source_field is not None
                else None
            ),
            getattr(container, field_name, None),
            getattr(container, source_field, None) if source_field is not None else None,
            getattr(getattr(container, "geometry_state", None), field_name, None),
            getattr(getattr(container, "runtime_state", None), field_name, None),
        )
        return next((candidate for candidate in candidates if candidate is not None), None)

    def _field(self, field_name, container, *, required=True):
        radarray_field = {
            "boundary_comoving_code": "boundary_radarray",
            "boundary_proper_code": "boundary_radarray",
            "rho_comoving_code": "rho_radarray",
            "rho_proper_code": "rho_radarray",
            "vel_supercomoving_code": "vel_radarray",
            "vel_proper_code": "vel_radarray",
            "temp_supercomoving_code": "temp_radarray",
            "temp_proper_code": "temp_radarray",
            "pre_supercomoving_code": "pre_radarray",
            "pre_proper_code": "pre_radarray",
            "ngamma_comoving_code": "ngamma_radarray",
            "ngamma_proper_code": "ngamma_radarray",
        }.get(field_name)
        source_field = {
            "boundary_comoving_code": "boundary_proper_code",
            "rho_comoving_code": "rho_proper_code",
            "vel_supercomoving_code": "vel_proper_code",
            "temp_supercomoving_code": "temp_proper_code",
            "pre_supercomoving_code": "pre_proper_code",
            "ngamma_comoving_code": "ngamma_proper_code",
        }.get(field_name)
        value = self._resolve_field_value(
            field_name,
            container,
            radarray_field,
            source_field,
        )
        if value is None and required:
            raise ValueError(f"initial condition is missing {field_name}")
        return value

    @staticmethod
    def _context(simulation, field_value=None):
        context = getattr(simulation.par, "cosmology_context", None)
        if context is None and isinstance(field_value, (RadArray, RadQuantity)):
            context = field_value.cosmology
        if context is None:
            hydrodynamics = getattr(simulation.par, "hydrodynamics", None)
            gamma = getattr(hydrodynamics, "gamma", None)
            if gamma is None:
                return None
            isothermal = getattr(hydrodynamics, "eos_type", "") == "isothermal"
            if float(gamma) <= 1.0 and not isothermal:
                return None
            context = CosmologyContext(
                gamma=float(gamma),
                cosmology="proper",
                isothermal=isothermal,
            )
        if not isinstance(context, CosmologyContext):
            raise TypeError("par.cosmology_context must be a CosmologyContext")
        simulation.par.cosmology_context = context
        return context

    @staticmethod
    def _code_values(value):
        if isinstance(value, (RadArray, RadQuantity)):
            return np.asarray(value.value, dtype=float)
        if hasattr(value, "units"):
            raise TypeError(
                "initial-condition runtime fields must use RadArray or plain code values",
            )
        return np.asarray(value, dtype=float)

    @classmethod
    def _convert(cls, value, target_field, *, context, x_comoving_code=None):
        if not isinstance(value, (RadArray, RadQuantity)):
            return cls._code_values(value)
        if value.code_units is None:
            raise TypeError(f"{target_field} RadArray has no CodeUnits")
        target_representation = (
            "comoving"
            if target_field.endswith("comoving_code")
            else "supercomoving"
            if target_field.endswith("supercomoving_code")
            else "proper"
        )
        source_representation = value.field_spec.representation
        if source_representation == target_representation or (
            target_representation == "proper"
            and value.field_spec.quantity in ("angular_momentum", "specific_angular_momentum")
        ):
            return np.asarray(value.value, dtype=float)
        if target_representation == "proper":
            converted = value.to_proper(x_comoving_code=x_comoving_code)
        else:
            converted = value.to_comoving(x_comoving_code=x_comoving_code)
        return np.asarray(converted.value, dtype=float)

    @staticmethod
    def _validate_finite_fields(fields):
        for field_name, field_values in fields:
            if not np.all(np.isfinite(field_values)):
                raise ValueError(f"active {field_name} contains non-finite values")

    @staticmethod
    def _validate_active_primitive_state(
        fluid,
        rho_proper_code,
        vel_proper_code,
        pre_proper_code,
        temp_proper_code,
        mu_dimensionless,
        volume_proper_code,
    ):
        InitialConditionWriter._validate_finite_fields(
            (
                ("rho_proper_code", rho_proper_code),
                ("vel_proper_code", vel_proper_code),
                ("pre_proper_code", pre_proper_code),
                ("temp_proper_code", temp_proper_code),
                ("mu_dimensionless", mu_dimensionless),
                ("volume_proper_code", volume_proper_code),
            ),
        )
        for values, message in (
            (rho_proper_code, "active rho_proper_code must be strictly positive"),
            (temp_proper_code, "active temp_proper_code must be non-negative"),
            (pre_proper_code, "active pre_proper_code must be non-negative"),
            (volume_proper_code, "active volume_proper_code must be strictly positive"),
        ):
            if np.any(values <= 0.0 if "strictly" in message else values < 0.0):
                raise ValueError(message)
        expected_pre_proper_code = np.asarray(
            fluid.eos.pressure(rho_proper_code, temp_proper_code, mu_dimensionless),
            dtype=float,
        )
        if not np.allclose(
            pre_proper_code,
            expected_pre_proper_code,
            rtol=1.0e-10,
            atol=1.0e-14,
        ):
            raise ValueError("active proper-code pressure is inconsistent with rho/temp/mu")

    @staticmethod
    def _validate_active_conserved_state(
        fluid,
        first,
        last,
        rho_proper_code,
        vel_proper_code,
        pre_proper_code,
        volume_proper_code,
    ):
        expected_mass_code = rho_proper_code * volume_proper_code
        expected_mom_code = expected_mass_code * vel_proper_code
        expected_energy_code = (
            fluid.eos.total_energy_density(
                rho_proper_code,
                vel_proper_code,
                pre_proper_code,
            )
            * volume_proper_code
        )
        mass_code = np.asarray(fluid.Mass_code[first:last], dtype=float)
        mom_code = np.asarray(fluid.Mom_code[first:last], dtype=float)
        energy_code = np.asarray(fluid.Energy_code[first:last], dtype=float)
        InitialConditionWriter._validate_finite_fields(
            (("Mass_code", mass_code), ("Mom_code", mom_code), ("Energy_code", energy_code)),
        )
        for actual, expected, message in (
            (mass_code, expected_mass_code, "active Mass_code is inconsistent with rho/volume"),
            (mom_code, expected_mom_code, "active Mom_code is inconsistent with rho/vel/volume"),
            (
                energy_code,
                expected_energy_code,
                "active Energy_code is inconsistent with rho/vel/pre/volume",
            ),
        ):
            if not np.allclose(actual, expected, rtol=1.0e-10, atol=1.0e-14):
                raise ValueError(message)

    @staticmethod
    def _validate_active_proper_state(simulation, first, last):
        """Validate the active proper-code primitive and conserved state."""
        fluid = simulation.fluid
        mesh = simulation.mesh
        fields = {
            name: np.asarray(getattr(fluid, name)[first:last], dtype=float)
            for name in (
                "rho_proper_code",
                "vel_proper_code",
                "pre_proper_code",
                "temp_proper_code",
                "mu",
            )
        }
        volume_proper_code = np.asarray(mesh.volume_proper_code[first:last], dtype=float)
        InitialConditionWriter._validate_active_primitive_state(
            fluid,
            fields["rho_proper_code"],
            fields["vel_proper_code"],
            fields["pre_proper_code"],
            fields["temp_proper_code"],
            fields["mu"],
            volume_proper_code,
        )
        InitialConditionWriter._validate_active_conserved_state(
            fluid,
            first,
            last,
            fields["rho_proper_code"],
            fields["vel_proper_code"],
            fields["pre_proper_code"],
            volume_proper_code,
        )

    def _prepare_geometry(self, mesh, par, boundary_values, cosmological_schema):
        width_values = np.diff(boundary_values)
        x_values = 0.5 * (boundary_values[:-1] + boundary_values[1:])
        geometry_state = getattr(mesh, "geometry_state", None)
        if geometry_state is not None:
            representation = "comoving" if cosmological_schema else "proper"
            area_values = np.asarray(
                getattr(geometry_state, f"area_{representation}_code"),
                dtype=float,
            )
            volume_values = np.asarray(
                getattr(geometry_state, f"volume_{representation}_code"),
                dtype=float,
            )
            return x_values, width_values, area_values, volume_values
        coordinate_system = getattr(
            getattr(par, "simulation", None),
            "coordinate_system",
            "cartesian",
        )
        if cosmological_schema or coordinate_system == "cartesian":
            area_values = np.ones_like(width_values)
            area_config = getattr(getattr(par, "mesh", None), "area_proper", None)
            if not cosmological_schema and area_config is not None:
                area_values *= float(area_config.to_value(self.code_units.area_unit))
            return x_values, width_values, area_values, width_values * area_values
        if coordinate_system != "spherical":
            raise ValueError(f"unsupported coordinate system {coordinate_system!r}")
        inner_radius_values = boundary_values[:-1]
        outer_radius_values = boundary_values[1:]
        area_values = 4.0 * np.pi * inner_radius_values**2
        volume_values = 4.0 * np.pi / 3.0 * (outer_radius_values**3 - inner_radius_values**3)
        volume_denominator = outer_radius_values**3 - inner_radius_values**3
        x_values = 0.5 * (inner_radius_values + outer_radius_values)
        nonzero_volume = volume_denominator != 0.0
        x_values[nonzero_volume] = (
            0.75
            * (outer_radius_values[nonzero_volume] ** 4 - inner_radius_values[nonzero_volume] ** 4)
            / volume_denominator[nonzero_volume]
        )
        return x_values, width_values, area_values, volume_values

    def _initialize_solver_state(
        self,
        simulation,
        source_ngamma_values,
        specific_angular_momentum_values,
        source_pressure,
        pressure_field,
        density_field,
        velocity_field,
        temperature_field,
        cosmological_schema,
        active_count,
        validate,
    ):
        par = simulation.par
        mesh = simulation.mesh
        fluid = simulation.fluid
        original_grid_cells = int(par.mesh.grid_cells)
        original_ghost_cells = int(par.mesh.ghost_cells)
        par.mesh.grid_cells = active_count
        par.mesh.ghost_cells = max(1, original_ghost_cells)
        mesh.par = par
        simulation.SetMesh()
        fluid.SetUpFluid(par, mesh=mesh)
        ghost_cells = int(par.mesh.ghost_cells)
        if source_ngamma_values is not None:
            photon_values = np.asarray(source_ngamma_values, dtype=float)
            if photon_values.shape[-1] != active_count:
                start = original_ghost_cells
                photon_values = photon_values[..., start : start + active_count]
            padding = (
                ((0, 0), (ghost_cells, ghost_cells))
                if photon_values.ndim == 2  # noqa: PLR2004
                else (ghost_cells, ghost_cells)
            )
            fluid.ngamma_code = np.pad(photon_values, padding, mode="edge")
        if specific_angular_momentum_values is not None:
            fluid.specific_angular_momentum_code = np.pad(
                specific_angular_momentum_values,
                (ghost_cells, ghost_cells),
                mode="edge",
            )
        simulation.solver.SetConserved(
            mesh,
            fluid,
            verbose=getattr(par, "verbose", 0),
        )
        first = ghost_cells
        last = first + active_count
        if source_pressure is None:
            pressure_values = np.asarray(getattr(fluid, pressure_field), dtype=float)[first:last]
        else:
            pressure_values = None
        if validate and not cosmological_schema:
            self._validate_active_proper_state(simulation, first, last)
        representation = "comoving" if cosmological_schema else "proper"
        mesh_fields = (
            f"boundary_{representation}_code",
            f"x_{representation}_code",
            f"width_{representation}_code",
            f"area_{representation}_code",
            f"volume_{representation}_code",
        )
        for field_name in mesh_fields:
            values = np.asarray(getattr(mesh, field_name), dtype=float)
            start, stop = (first, last + 1) if field_name.startswith("boundary_") else (first, last)
            setattr(mesh, field_name, values[start:stop])
        self._trim_solver_fields(
            fluid,
            density_field,
            velocity_field,
            temperature_field,
            pressure_field,
            cosmological_schema,
            active_count,
            ghost_cells,
            first,
            last,
        )
        par.mesh.grid_cells = original_grid_cells
        par.mesh.ghost_cells = original_ghost_cells
        return pressure_values

    @staticmethod
    def _trim_solver_fields(
        fluid,
        density_field,
        velocity_field,
        temperature_field,
        pressure_field,
        cosmological_schema,
        active_count,
        ghost_cells,
        first,
        last,
    ):
        fluid_fields = {
            density_field,
            velocity_field,
            temperature_field,
            pressure_field,
            "mu",
            "Mass_code",
            "Mom_code",
            "Energy_code",
            "InternalEnergy_code",
            "AngularMomentum_code",
            "specific_angular_momentum_code",
            "GravitationalPotentialEnergy_code",
            "xHI",
            "xHeI",
            "xHeII",
            "xHeIII",
            "ngamma_code",
            "tau_supercomoving_code" if cosmological_schema else "time_proper_code",
        }
        full_size = active_count + 2 * ghost_cells
        for field_name in fluid_fields:
            if not hasattr(fluid, field_name):
                continue
            values = np.asarray(getattr(fluid, field_name))
            if values.ndim == 1 and values.size == full_size:
                setattr(fluid, field_name, values[first:last])
            elif values.ndim == 2 and values.shape[-1] == full_size:  # noqa: PLR2004
                setattr(fluid, field_name, values[..., first:last])

    def _finalize_runtime_state(
        self,
        simulation,
        cosmological_schema,
        x_values,
        boundary_values,
        width_values,
        area_values,
        volume_values,
        density_values,
        velocity_values,
        pressure_values,
        temperature_values,
    ):
        par = simulation.par
        mesh = simulation.mesh
        fluid = simulation.fluid
        if cosmological_schema:
            fields = SUPERCOMOVING_RUNTIME_FIELDS
            time_name = "tau_supercomoving_code"
            if self.ic_config is not None and "time_cosmic" in self.ic_config:
                cosmic_time = self._to_code_values(
                    self.ic_config["time_cosmic"],
                    self.code_units.time_unit,
                )
                model = getattr(getattr(par, "cosmology", None), "model", None)
                if model is None:
                    raise ValueError("cosmological IC time conversion requires par.cosmology.model")
                cosmic_time_value = float(np.asarray(cosmic_time).flat[0])
                time_value = (
                    0.0 if cosmic_time_value == 0.0 else model.supercomoving_time(cosmic_time_value)
                )
            else:
                time_value = getattr(
                    par,
                    "tau_supercomoving_code",
                    getattr(
                        getattr(par, "simulation", None),
                        time_name,
                        getattr(fluid, time_name, 0.0),
                    ),
                )
            time_value = float(np.asarray(time_value).flat[0])
            fluid.tau_supercomoving_code = time_value
            par.tau_supercomoving_code = np.asarray([time_value], dtype=float)
            par.simulation.tau_supercomoving_code = par.tau_supercomoving_code.copy()
            mesh.geometry_state = MeshGeometryState.from_arrays(
                fields,
                x_comoving_code=x_values,
                boundary_comoving_code=boundary_values,
                width_comoving_code=width_values,
                area_comoving_code=area_values,
                volume_comoving_code=volume_values,
            )
            fluid.runtime_state = FluidRuntimeState.from_arrays(
                fields,
                rho_comoving_code=density_values,
                vel_supercomoving_code=velocity_values,
                pre_supercomoving_code=pressure_values,
                temp_supercomoving_code=temperature_values,
                tau_supercomoving_code=time_value,
                mu_dimensionless=np.asarray(fluid.mu, dtype=float),
            )
            par.simulation.box_size_comoving_code = float(boundary_values[-1])
            return
        fields = PROPER_RUNTIME_FIELDS
        if self.ic_config is not None and "time_proper" in self.ic_config:
            time_value = self._to_code_values(
                self.ic_config["time_proper"],
                self.code_units.time_unit,
            )
        else:
            time_value = getattr(
                fluid,
                "time_proper_code",
                getattr(getattr(fluid, "runtime_state", None), "time_proper_code", 0.0),
            )
        time_value = float(np.asarray(time_value).flat[0])
        fluid.time_proper_code = time_value
        mesh.geometry_state = MeshGeometryState.from_arrays(
            fields,
            x_proper_code=x_values,
            boundary_proper_code=boundary_values,
            width_proper_code=width_values,
            area_proper_code=area_values,
            volume_proper_code=volume_values,
        )
        fluid.runtime_state = FluidRuntimeState.from_arrays(
            fields,
            rho_proper_code=density_values,
            vel_proper_code=velocity_values,
            pre_proper_code=pressure_values,
            temp_proper_code=temperature_values,
            time_proper_code=time_value,
            mu_dimensionless=np.asarray(fluid.mu, dtype=float),
        )
        par.simulation.box_size_proper_code = float(boundary_values[-1])

    def _prepare_optional_fields(self, fluid, context, cosmological_schema, density_values):
        ngamma_field = "ngamma_comoving_code" if cosmological_schema else "ngamma_proper_code"
        source_ngamma = self._field(ngamma_field, fluid, required=False)
        source_ngamma_values = None
        if source_ngamma is not None:
            source_ngamma_values = self._convert(source_ngamma, ngamma_field, context=context)
            fluid.ngamma_code = source_ngamma_values.copy()
        specific = getattr(fluid, "specific_angular_momentum_radarray", None)
        if specific is None:
            specific = self._fields.get("specific_angular_momentum_radarray")
        if specific is None:
            specific = getattr(fluid, "specific_angular_momentum_code", None)
        if specific is not None:
            specific = self._convert_angular_momentum(specific, context)
            fluid.specific_angular_momentum_code = specific
        mu = self._fields.get("mu")
        if mu is not None:
            fluid.mu = self._code_values(mu)
        elif not hasattr(fluid, "mu"):
            fluid.mu = np.ones_like(density_values)
        xhi = self._fields.get("xHI")
        if xhi is not None:
            fluid.xHI = self._code_values(xhi)
        specific_values = getattr(fluid, "specific_angular_momentum_code", None)
        if specific_values is not None:
            specific_values = np.asarray(specific_values, dtype=float).copy()
        return source_ngamma_values, specific_values

    def _convert_angular_momentum(self, value, context):
        if hasattr(value, "units") and not isinstance(value, (RadArray, RadQuantity)):
            code_unit = _code_unit_for_spec(
                self.code_units,
                field_spec("specific_angular_momentum_code", self.code_units),
            )
            return np.asarray(value.to_value(code_unit.units), dtype=float) / float(code_unit.value)
        return self._convert(value, "specific_angular_momentum_code", context=context)

    def prepare(self, *, validate=False):
        """Convert fields and create canonical typed runtime state.

        Set ``validate=True`` to run the active proper-code consistency
        checks after solver initialization.  Validation is opt-in because
        preparation is also used for conversion-only writer workflows.
        """
        simulation = self.simulation
        par = simulation.par
        mesh = simulation.mesh
        fluid = simulation.fluid
        cosmological_schema = self._cosmological_schema(simulation)
        boundary_field = "boundary_comoving_code" if cosmological_schema else "boundary_proper_code"
        density_field = "rho_comoving_code" if cosmological_schema else "rho_proper_code"
        velocity_field = "vel_supercomoving_code" if cosmological_schema else "vel_proper_code"
        temperature_field = "temp_supercomoving_code" if cosmological_schema else "temp_proper_code"
        pressure_field = "pre_supercomoving_code" if cosmological_schema else "pre_proper_code"
        source_boundary = self._field(boundary_field, mesh)
        context = self._context(simulation, source_boundary)
        boundary_values = self._convert(
            source_boundary,
            boundary_field,
            context=context,
        )
        if self.box_size is not None:
            box_size_values = self._convert(
                self.box_size,
                boundary_field,
                context=context,
            )
            if not np.isclose(
                float(np.asarray(box_size_values).flat[-1]),
                float(boundary_values[-1]),
                rtol=1e-12,
                atol=0.0,
            ):
                raise ValueError(
                    "box_size does not match the assigned mesh boundary",
                )
        x_values, width_values, area_values, volume_values = self._prepare_geometry(
            mesh,
            par,
            boundary_values,
            cosmological_schema,
        )

        source_x = self._fields.get(
            "x_proper_code" if not cosmological_schema else "x_comoving_code",
        )
        if source_x is not None:
            x_values = self._convert(
                source_x,
                "x_comoving_code" if cosmological_schema else "x_proper_code",
                context=context,
            )

        source_density = self._field(density_field, fluid)
        source_velocity = self._field(velocity_field, fluid)
        source_temperature = self._field(temperature_field, fluid)
        source_pressure = self._field(pressure_field, fluid, required=False)
        density_values = self._convert(source_density, density_field, context=context)
        velocity_values = self._convert(
            source_velocity,
            velocity_field,
            context=context,
            x_comoving_code=x_values if cosmological_schema else None,
        )
        temperature_values = self._convert(source_temperature, temperature_field, context=context)
        if source_pressure is None:
            pressure_values = np.zeros_like(density_values)
        else:
            pressure_values = self._convert(source_pressure, pressure_field, context=context)

        setattr(mesh, boundary_field, boundary_values)
        setattr(mesh, "x_comoving_code" if cosmological_schema else "x_proper_code", x_values)
        setattr(
            mesh,
            "width_comoving_code" if cosmological_schema else "width_proper_code",
            width_values,
        )
        setattr(
            mesh,
            "area_comoving_code" if cosmological_schema else "area_proper_code",
            area_values,
        )
        setattr(
            mesh,
            "volume_comoving_code" if cosmological_schema else "volume_proper_code",
            volume_values,
        )
        setattr(fluid, density_field, density_values)
        setattr(fluid, velocity_field, velocity_values)
        setattr(fluid, temperature_field, temperature_values)
        setattr(fluid, pressure_field, pressure_values)
        source_ngamma_values, specific_angular_momentum_values = self._prepare_optional_fields(
            fluid,
            context,
            cosmological_schema,
            density_values,
        )

        active_count = int(np.asarray(density_values).size)
        solver_ready = all(
            hasattr(simulation, attribute) for attribute in ("SetMesh", "fluid", "solver")
        ) and hasattr(simulation.solver, "SetConserved")
        if solver_ready:
            computed_pressure_values = self._initialize_solver_state(
                simulation,
                source_ngamma_values,
                specific_angular_momentum_values,
                source_pressure,
                pressure_field,
                density_field,
                velocity_field,
                temperature_field,
                cosmological_schema,
                active_count,
                validate,
            )
            if source_pressure is None:
                pressure_values = computed_pressure_values

        self._finalize_runtime_state(
            simulation,
            cosmological_schema,
            x_values,
            boundary_values,
            width_values,
            area_values,
            volume_values,
            density_values,
            velocity_values,
            pressure_values,
            temperature_values,
        )
        return simulation

    def write(self, filename, *, validate=False):
        """Prepare and write the state, optionally validating active cells."""
        from radhydropy import io  # noqa: PLC0415

        self.prepare(validate=validate)
        return io.write_snapshot_hdf5(
            self.simulation,
            filename,
            provenance=self.provenance,
        )
