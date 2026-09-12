"""Shared RadArray-to-runtime initial-condition writer boundary."""

import warnings

import numpy as np

from radhydropy.cosmology_context import CosmologyContext
from radhydropy.field_metadata import field_spec
from radhydropy.radarray import RadArray, RadQuantity, _code_unit_for_spec
from radhydropy.rsim import Rsim
from radhydropy.runtime_fields import (
    FluidRuntimeState,
    MeshGeometryState,
    PROPER_RUNTIME_FIELDS,
    SUPERCOMOVING_RUNTIME_FIELDS,
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
        box_size=None,
        cosmology_context=None,
        scale_factor=None,
        hubble_parameter_km_s_Mpc=None,
        provenance=None,
    ):
        if simulation is None:
            if par_config is None:
                raise TypeError(
                    "InitialConditionWriter requires simulation or par_config"
                )
            simulation = Rsim(par_config)
        self.simulation = simulation
        cosmology_parameters = getattr(simulation.par, "cosmology", None)
        cosmological = getattr(
            cosmology_parameters,
            "cosmological",
            getattr(simulation.par, "cosmological", False),
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
                f"supercomoving_coordinates={bool(supercomoving_coordinates)}"
            )
        self.code_units = simulation.par.units.CodeUnits
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
                        f"writer code-unit {name} does not match par_config"
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
                isothermal=(
                    getattr(hydrodynamics, "eos_type", "") == "isothermal"
                ),
                scale_factor=float(scale_factor),
                hubble_parameter_km_s_Mpc=float(
                    hubble_parameter_km_s_Mpc or 0.0
                ),
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
            },
        )

    @classmethod
    def from_rsim(cls, simulation, *, provenance=None):
        return cls(simulation, provenance=provenance)

    def _primitive_field_name(self, values, context, representation=None):
        if not hasattr(values, "units"):
            raise TypeError("writer.radarray requires a unit-bearing array")
        cosmological_schema = bool(
            getattr(self.simulation.par, "cosmological_expansion", False)
            and getattr(self.simulation.par, "supercomoving_coordinates", False)
        )
        field_names = (
            (
                "boundary_comoving_code",
                "rho_comoving_code",
                "vel_supercomoving_code",
                "temp_supercomoving_code",
                "pre_supercomoving_code",
            )
            if cosmological_schema
            else (
                "boundary_proper_code",
                "rho_proper_code",
                "vel_proper_code",
                "temp_proper_code",
                "pre_proper_code",
            )
        )
        if representation is not None:
            representation_fields = {
                "proper": (
                    "boundary_proper_code", "rho_proper_code", "vel_proper_code",
                    "temp_proper_code", "pre_proper_code",
                ),
                "comoving": ("boundary_comoving_code", "rho_comoving_code"),
                "supercomoving": (
                    "vel_supercomoving_code", "temp_supercomoving_code",
                    "pre_supercomoving_code",
                ),
            }
            if representation not in representation_fields:
                raise ValueError(
                    "writer.radarray representation must be proper, comoving, "
                    "or supercomoving"
                )
            field_names = tuple(representation_fields[representation])
        field_units = (
            self.code_units.length_unit,
            self.code_units.density_unit,
            self.code_units.velocity_unit,
            self.code_units.temperature_unit,
            self.code_units.pressure_unit,
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
        }
        if representation is not None:
            field_units = tuple(field_units_by_name[name] for name in field_names)
        value_dimensions = getattr(values.units, "dimensions", None)
        if value_dimensions is None:
            value_dimensions = values.units.units.dimensions
        matching_fields = [
            field_name
            for field_name, field_unit in zip(field_names, field_units)
            if value_dimensions == field_unit.units.dimensions
        ]
        if len(matching_fields) != 1:
            raise ValueError(
                "writer.radarray cannot infer a canonical primitive field from "
                f"unit dimensions {value_dimensions}"
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
                "before creating a RadArray"
            )
        if representation in ("comoving", "supercomoving"):
            cosmology_parameters = getattr(self.simulation.par, "cosmology", None)
            missing_flags = []
            cosmological = getattr(
                cosmology_parameters,
                "cosmological",
                getattr(self.simulation.par, "cosmological", False),
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
                    + " and ".join(missing_flags)
                )
        elif representation == "proper":
            cosmology_parameters = getattr(self.simulation.par, "cosmology", None)
            cosmological = getattr(
                cosmology_parameters,
                "cosmological",
                getattr(self.simulation.par, "cosmological", False),
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
            context.hubble_parameter_km_s_Mpc
            if field_name == "vel_supercomoving_code"
            else None
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
                f"not {representation!r}"
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
            value.to_value(code_unit.units), dtype=float
        ) / float(code_unit.value)

    def radquantity(self, value):
        """Create a ``RadQuantity`` from a unit-bearing primitive scalar."""
        context = self._context(self.simulation)
        if context is None:
            raise ValueError(
                "InitialConditionWriter requires a valid cosmology context "
                "before creating a RadQuantity"
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
        }.get(field_name)
        source_field = {
            "boundary_comoving_code": "boundary_proper_code",
            "rho_comoving_code": "rho_proper_code",
            "vel_supercomoving_code": "vel_proper_code",
            "temp_supercomoving_code": "temp_proper_code",
            "pre_supercomoving_code": "pre_proper_code",
        }.get(field_name)
        value = self._fields.get(field_name)
        # A prepared runtime simulation owns the canonical target field.  Use
        # it before the analysis-oriented ``*_radarray`` view, which may carry
        # stale or differently represented metadata after a restart.
        if value is None:
            value = getattr(container, field_name, None)
        if value is None and radarray_field is not None:
            value = self._fields.get(radarray_field)
        if value is None:
            if source_field is not None:
                value = self._fields.get(source_field)
        if value is None:
            value = getattr(container, f"{field_name}_radarray", None)
        if value is None and radarray_field is not None:
            value = getattr(container, radarray_field, None)
        if value is None and source_field is not None:
            value = getattr(container, f"{source_field}_radarray", None)
        if value is None:
            value = getattr(container, field_name, None)
        if value is None and source_field is not None:
            value = getattr(container, source_field, None)
        if value is None:
            state = getattr(container, "geometry_state", None)
            value = getattr(state, field_name, None)
        if value is None:
            state = getattr(container, "runtime_state", None)
            value = getattr(state, field_name, None)
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
            raise TypeError("initial-condition runtime fields must use RadArray or plain code values")
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
        if (
            source_representation == target_representation
            or (
                target_representation == "proper"
                and value.field_spec.quantity
                in ("angular_momentum", "specific_angular_momentum")
            )
        ):
            return np.asarray(value.value, dtype=float)
        if target_representation == "proper":
            converted = value.to_proper(x_comoving_code=x_comoving_code)
        else:
            converted = value.to_comoving(x_comoving_code=x_comoving_code)
        return np.asarray(converted.value, dtype=float)

    @staticmethod
    def _validate_active_proper_state(simulation, first, last):
        """Validate the active proper-code primitive and conserved state.

        This mirrors the active-state checks used by the hydro IC helpers, but
        lives at the writer boundary so HDF5 input cannot bypass them.
        """
        fluid = simulation.fluid
        mesh = simulation.mesh
        rho_proper_code = np.asarray(
            fluid.rho_proper_code[first:last], dtype=float
        )
        vel_proper_code = np.asarray(
            fluid.vel_proper_code[first:last], dtype=float
        )
        pre_proper_code = np.asarray(
            fluid.pre_proper_code[first:last], dtype=float
        )
        temp_proper_code = np.asarray(
            fluid.temp_proper_code[first:last], dtype=float
        )
        mu_dimensionless = np.asarray(fluid.mu[first:last], dtype=float)
        volume_proper_code = np.asarray(
            mesh.volume_proper_code[first:last], dtype=float
        )

        for field_name, field_values in (
            ("rho_proper_code", rho_proper_code),
            ("vel_proper_code", vel_proper_code),
            ("pre_proper_code", pre_proper_code),
            ("temp_proper_code", temp_proper_code),
            ("mu_dimensionless", mu_dimensionless),
            ("volume_proper_code", volume_proper_code),
        ):
            if not np.all(np.isfinite(field_values)):
                raise ValueError(f"active {field_name} contains non-finite values")

        if np.any(rho_proper_code <= 0.0):
            raise ValueError("active rho_proper_code must be strictly positive")
        if np.any(temp_proper_code < 0.0):
            raise ValueError("active temp_proper_code must be non-negative")
        if np.any(pre_proper_code < 0.0):
            raise ValueError("active pre_proper_code must be non-negative")
        if np.any(volume_proper_code <= 0.0):
            raise ValueError("active volume_proper_code must be strictly positive")

        expected_pre_proper_code = np.asarray(
            fluid.eos.pressure(
                rho_proper_code,
                temp_proper_code,
                mu_dimensionless,
            ),
            dtype=float,
        )
        if not np.allclose(
            pre_proper_code,
            expected_pre_proper_code,
            rtol=1.0e-10,
            atol=1.0e-14,
        ):
            raise ValueError(
                "active proper-code pressure is inconsistent with rho/temp/mu"
            )

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
        for field_name, field_values in (
            ("Mass_code", mass_code),
            ("Mom_code", mom_code),
            ("Energy_code", energy_code),
        ):
            if not np.all(np.isfinite(field_values)):
                raise ValueError(f"active {field_name} contains non-finite values")
        if not np.allclose(
            mass_code, expected_mass_code, rtol=1.0e-10, atol=1.0e-14
        ):
            raise ValueError("active Mass_code is inconsistent with rho/volume")
        if not np.allclose(
            mom_code, expected_mom_code, rtol=1.0e-10, atol=1.0e-14
        ):
            raise ValueError("active Mom_code is inconsistent with rho/vel/volume")
        if not np.allclose(
            energy_code, expected_energy_code, rtol=1.0e-10, atol=1.0e-14
        ):
            raise ValueError(
                "active Energy_code is inconsistent with rho/vel/pre/volume"
            )

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
        cosmological_schema = bool(
            getattr(par, "cosmological_expansion", False)
            and getattr(par, "supercomoving_coordinates", False)
        )
        boundary_field = (
            "boundary_comoving_code"
            if cosmological_schema
            else "boundary_proper_code"
        )
        density_field = "rho_comoving_code" if cosmological_schema else "rho_proper_code"
        velocity_field = (
            "vel_supercomoving_code" if cosmological_schema else "vel_proper_code"
        )
        temperature_field = (
            "temp_supercomoving_code" if cosmological_schema else "temp_proper_code"
        )
        pressure_field = (
            "pre_supercomoving_code" if cosmological_schema else "pre_proper_code"
        )
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
                    "box_size does not match the assigned mesh boundary"
                )
        x_values = 0.5 * (boundary_values[:-1] + boundary_values[1:])
        width_values = np.diff(boundary_values)

        geometry_state = getattr(mesh, "geometry_state", None)
        if geometry_state is not None:
            area_values = np.asarray(getattr(geometry_state, "area_proper_code" if not cosmological_schema else "area_comoving_code"), dtype=float)
            volume_values = np.asarray(getattr(geometry_state, "volume_proper_code" if not cosmological_schema else "volume_comoving_code"), dtype=float)
        else:
            area_values = np.ones_like(width_values)
            volume_values = width_values.copy()

        source_x = self._fields.get(
            "x_proper_code" if not cosmological_schema else "x_comoving_code"
        )
        if source_x is not None:
            x_values = self._convert(source_x, "x_comoving_code" if cosmological_schema else "x_proper_code", context=context)

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
        setattr(mesh, "width_comoving_code" if cosmological_schema else "width_proper_code", width_values)
        setattr(mesh, "area_comoving_code" if cosmological_schema else "area_proper_code", area_values)
        setattr(mesh, "volume_comoving_code" if cosmological_schema else "volume_proper_code", volume_values)
        setattr(fluid, density_field, density_values)
        setattr(fluid, velocity_field, velocity_values)
        setattr(fluid, temperature_field, temperature_values)
        setattr(fluid, pressure_field, pressure_values)
        ngamma_radarray = self._fields.get("ngamma_radarray")
        if ngamma_radarray is not None:
            fluid.ngamma_code = self._code_values(ngamma_radarray)
        specific_angular_momentum = getattr(
            fluid, "specific_angular_momentum_radarray", None
        )
        if specific_angular_momentum is None:
            specific_angular_momentum = self._fields.get(
                "specific_angular_momentum_radarray"
            )
        if specific_angular_momentum is None:
            specific_angular_momentum = getattr(
                fluid, "specific_angular_momentum_code", None
            )
        if specific_angular_momentum is not None:
            if hasattr(specific_angular_momentum, "units") and not isinstance(
                specific_angular_momentum, (RadArray, RadQuantity)
            ):
                code_unit = _code_unit_for_spec(
                    self.code_units,
                    field_spec("specific_angular_momentum_code", self.code_units),
                )
                specific_angular_momentum = np.asarray(
                    specific_angular_momentum.to_value(code_unit.units),
                    dtype=float,
                ) / float(code_unit.value)
            else:
                specific_angular_momentum = self._convert(
                    specific_angular_momentum,
                    "specific_angular_momentum_code",
                    context=context,
                )
            fluid.specific_angular_momentum_code = specific_angular_momentum
        if not hasattr(fluid, "mu"):
            fluid.mu = np.ones_like(density_values)
        specific_angular_momentum_values = getattr(
            fluid, "specific_angular_momentum_code", None
        )
        if specific_angular_momentum_values is not None:
            specific_angular_momentum_values = np.asarray(
                specific_angular_momentum_values, dtype=float
            ).copy()

        active_count = int(np.asarray(density_values).size)
        solver_ready = all(
            hasattr(simulation, attribute)
            for attribute in ("SetMesh", "fluid", "solver")
        ) and hasattr(simulation.solver, "SetConserved")
        if solver_ready:
            # Complete the same solver-ready initialization used by the normal
            # IC path.  The writer receives active-cell primitive values, so
            # setup temporarily adds ghost cells, derives conserved fields,
            # and then trims the state back to the canonical IC representation.
            original_grid_cells = int(par.mesh.grid_cells)
            original_ghost_cells = int(par.mesh.ghost_cells)
            par.mesh.grid_cells = active_count
            par.mesh.ghost_cells = max(1, original_ghost_cells)
            mesh._par = par
            simulation.SetMesh()
            fluid.SetUpFluid(par, mesh=mesh)
            if specific_angular_momentum_values is not None:
                fluid.specific_angular_momentum_code = (
                    np.pad(
                        specific_angular_momentum_values,
                        (int(par.mesh.ghost_cells), int(par.mesh.ghost_cells)),
                        mode="edge",
                    )
                )
            simulation.solver.SetConserved(
                mesh,
                fluid,
                verbose=getattr(par, "verbose", 0),
            )
            ghost_cells = int(par.mesh.ghost_cells)
            first = ghost_cells
            last = first + active_count
            if source_pressure is None:
                pressure_values = np.asarray(
                    getattr(fluid, pressure_field), dtype=float
                )[first:last]
            if validate and not cosmological_schema:
                self._validate_active_proper_state(simulation, first, last)
            mesh_attribute = "boundary_comoving_code" if cosmological_schema else "boundary_proper_code"
            for field_name in (
                mesh_attribute,
                "x_comoving_code" if cosmological_schema else "x_proper_code",
                "width_comoving_code" if cosmological_schema else "width_proper_code",
                "area_comoving_code" if cosmological_schema else "area_proper_code",
                "volume_comoving_code" if cosmological_schema else "volume_proper_code",
            ):
                values = getattr(mesh, field_name)
                start, stop = (first, last + 1) if field_name == mesh_attribute else (first, last)
                setattr(mesh, field_name, np.asarray(values, dtype=float)[start:stop])
            fluid_attribute_names = {
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
            }
            if cosmological_schema:
                fluid_attribute_names.add("tau_supercomoving_code")
            else:
                fluid_attribute_names.add("time_proper_code")
            for field_name in fluid_attribute_names:
                if not hasattr(fluid, field_name):
                    continue
                values_array = np.asarray(getattr(fluid, field_name))
                if values_array.ndim == 0:
                    continue
                if values_array.ndim == 1:
                    if values_array.size != active_count + 2 * ghost_cells:
                        continue
                    trimmed_values = values_array[first:last]
                elif (
                    values_array.ndim == 2
                    and values_array.shape[-1] == active_count + 2 * ghost_cells
                ):
                    trimmed_values = values_array[..., first:last]
                else:
                    continue
                setattr(fluid, field_name, trimmed_values)
            par.mesh.grid_cells = original_grid_cells
            par.mesh.ghost_cells = original_ghost_cells

        if cosmological_schema:
            geometry_fields = SUPERCOMOVING_RUNTIME_FIELDS
            fluid_fields = SUPERCOMOVING_RUNTIME_FIELDS
            time_value = getattr(
                par,
                "tau_supercomoving_code",
                getattr(
                    getattr(par, "simulation", None),
                    "tau_supercomoving_code",
                    getattr(
                        fluid,
                        "tau_supercomoving_code",
                        getattr(
                            getattr(fluid, "runtime_state", None),
                            "tau_supercomoving_code",
                            0.0,
                        ),
                    ),
                ),
            )
            setattr(fluid, "tau_supercomoving_code", float(np.asarray(time_value).flat[0]))
            par.tau_supercomoving_code = np.asarray(
                [fluid.tau_supercomoving_code], dtype=float
            )
            par.simulation.tau_supercomoving_code = (
                par.tau_supercomoving_code.copy()
            )
            mesh.geometry_state = MeshGeometryState.from_arrays(
                geometry_fields,
                x_comoving_code=x_values,
                boundary_comoving_code=boundary_values,
                width_comoving_code=width_values,
                area_comoving_code=area_values,
                volume_comoving_code=volume_values,
            )
            fluid.runtime_state = FluidRuntimeState.from_arrays(
                fluid_fields,
                rho_comoving_code=density_values,
                vel_supercomoving_code=velocity_values,
                pre_supercomoving_code=pressure_values,
                temp_supercomoving_code=temperature_values,
                tau_supercomoving_code=fluid.tau_supercomoving_code,
                mu_dimensionless=np.asarray(fluid.mu, dtype=float),
            )
            par.simulation.box_size_comoving_code = float(boundary_values[-1])
        else:
            geometry_fields = PROPER_RUNTIME_FIELDS
            fluid_fields = PROPER_RUNTIME_FIELDS
            time_value = getattr(
                fluid,
                "time_proper_code",
                getattr(
                    getattr(fluid, "runtime_state", None),
                    "time_proper_code",
                    getattr(
                        getattr(par, "simulation", None),
                        "time_proper_code",
                        getattr(par, "time_proper_code", 0.0),
                    ),
                ),
            )
            setattr(fluid, "time_proper_code", float(np.asarray(time_value).flat[0]))
            mesh.geometry_state = MeshGeometryState.from_arrays(
                geometry_fields,
                x_proper_code=x_values,
                boundary_proper_code=boundary_values,
                width_proper_code=width_values,
                area_proper_code=area_values,
                volume_proper_code=volume_values,
            )
            fluid.runtime_state = FluidRuntimeState.from_arrays(
                fluid_fields,
                rho_proper_code=density_values,
                vel_proper_code=velocity_values,
                pre_proper_code=pressure_values,
                temp_proper_code=temperature_values,
                time_proper_code=fluid.time_proper_code,
                mu_dimensionless=np.asarray(fluid.mu, dtype=float),
            )
            par.simulation.box_size_proper_code = float(boundary_values[-1])
        return simulation

    def write(self, filename, *, validate=False):
        """Prepare and write the state, optionally validating active cells."""
        from radhydropy import io

        self.prepare(validate=validate)
        return io._writehdf5(
            self.simulation,
            filename,
            provenance=self.provenance,
        )
