"""Shared RadArray-to-runtime initial-condition writer boundary."""

import numpy as np

from radhydropy.cosmology_context import CosmologyContext
from radhydropy.radarray import RadArray, RadQuantity
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
        self.provenance = provenance
        self.box_size = box_size
        self._fields = {}
        if code_units is not None:
            actual_code_units = simulation.par.units.CodeUnits
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
            cosmology = getattr(simulation.par, "cosmology", None)
            simulation.par.cosmology_context = CosmologyContext(
                gamma=gamma,
                cosmology=getattr(cosmology, "type_name", "proper"),
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
            },
        )

    @classmethod
    def from_rsim(cls, simulation, *, provenance=None):
        return cls(simulation, provenance=provenance)

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
            if float(gamma) <= 1.0:
                return None
            context = CosmologyContext(gamma=float(gamma), cosmology="proper")
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
        if source_representation == target_representation:
            return np.asarray(value.value, dtype=float)
        if target_representation == "proper":
            converted = value.to_proper(x_comoving_code=x_comoving_code)
        else:
            converted = value.to_comoving(x_comoving_code=x_comoving_code)
        return np.asarray(converted.value, dtype=float)

    def prepare(self):
        """Convert registered fields and create canonical typed runtime state."""
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
        if isinstance(source_boundary, RadArray):
            area_values = np.ones_like(width_values)
            volume_values = width_values.copy()
        elif geometry_state is not None:
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
        if not hasattr(fluid, "mu"):
            fluid.mu = np.ones_like(density_values)

        if cosmological_schema:
            geometry_fields = SUPERCOMOVING_RUNTIME_FIELDS
            fluid_fields = SUPERCOMOVING_RUNTIME_FIELDS
            time_value = getattr(
                fluid,
                "tau_supercomoving_code",
                getattr(
                    getattr(fluid, "runtime_state", None),
                    "tau_supercomoving_code",
                    getattr(
                        getattr(par, "simulation", None),
                        "tau_supercomoving_code",
                        getattr(par, "tau_supercomoving_code", 0.0),
                    ),
                ),
            )
            setattr(fluid, "tau_supercomoving_code", float(np.asarray(time_value).flat[0]))
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

    def write(self, filename):
        """Prepare the state and delegate to the canonical HDF5 serializer."""
        from radhydropy import io

        self.prepare()
        return io._writehdf5(
            self.simulation,
            filename,
            provenance=self.provenance,
        )
