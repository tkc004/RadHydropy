"""Representation-aware numerical arrays for RadHydropy."""

import numpy as np
import unyt

from radhydropy.cosmology_context import CosmologyContext
from radhydropy.field_metadata import (
    FieldSpec,
    field_spec as make_field_spec,
    hubble_parameter_code,
)


class RepresentationMismatchError(ValueError):
    """Raised when arithmetic combines incompatible coordinate representations."""


_QUANTITY_UNIT_PROPERTIES = {
    "radius": "length_unit",
    "mass_density": "density_unit",
    "velocity": "velocity_unit",
    "temperature": "temperature_unit",
    "pressure": "pressure_unit",
    "mass": "mass_unit",
    "energy": "energy_unit",
    "angular_momentum": "angular_momentum_unit",
    "specific_angular_momentum": "specific_angular_momentum_unit",
    "number_density": "number_density_unit",
}


def _code_unit_for_spec(code_units, spec):
    if spec.quantity == "angular_momentum":
        return code_units.mass_unit * code_units.length_unit * code_units.velocity_unit
    if spec.quantity == "specific_angular_momentum":
        return code_units.length_unit * code_units.velocity_unit
    try:
        return getattr(code_units, _QUANTITY_UNIT_PROPERTIES[spec.quantity])
    except KeyError:
        return _unit_for_dimensions(code_units, spec.dimensions)


def _unit_for_dimensions(code_units, dimensions):
    mass, length, velocity, current, temperature = dimensions
    return (
        code_units.mass_unit ** mass
        * code_units.length_unit ** length
        * code_units.velocity_unit ** velocity
        * code_units.current_unit ** current
        * code_units.temperature_unit ** temperature
    )


class RadArray(unyt.unyt_array):
    """A code-unit ``unyt_array`` carrying field and cosmology metadata."""

    def __new__(
        cls,
        values,
        *,
        code_units,
        field_spec,
        cosmology,
    ):
        if not isinstance(field_spec, FieldSpec):
            raise TypeError("field_spec must be a FieldSpec")
        if not isinstance(cosmology, CosmologyContext):
            raise TypeError("cosmology must be a CosmologyContext")
        code_unit = _code_unit_for_spec(code_units, field_spec)
        obj = super().__new__(cls, values, code_unit)
        obj.code_units = code_units
        obj.field_spec = field_spec
        obj.cosmology = cosmology
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.code_units = getattr(obj, "code_units", None)
        self.field_spec = getattr(obj, "field_spec", None)
        self.cosmology = getattr(obj, "cosmology", None)

    @property
    def representation(self):
        return self.field_spec.representation

    def _target(self, values, field_name):
        hubble = (
            self.cosmology.hubble_parameter_km_s_Mpc
            if field_name == "vel_supercomoving_code"
            else None
        )
        target_spec = make_field_spec(
            field_name,
            self.code_units,
            cosmology=self.cosmology.cosmology,
            scale_factor=self.cosmology.scale_factor,
            hubble_parameter_km_s_Mpc=hubble,
        )
        return RadArray(
            values,
            code_units=self.code_units,
            field_spec=target_spec,
            cosmology=self.cosmology,
        )

    def to_proper(self, *, x_comoving_code=None):
        """Return a new array converted to the proper-code representation."""
        source = self.field_spec
        if source.representation == "proper":
            return self

        a = self.cosmology.scale_factor
        values = np.asarray(self)
        quantity = source.quantity

        if quantity == "radius":
            return self._target(values * a, "boundary_proper_code")
        if quantity == "mass_density":
            return self._target(values / a**3, "rho_proper_code")
        if quantity == "temperature":
            gamma = self.cosmology.gamma
            return self._target(
                values / a ** (3.0 * (gamma - 1.0)),
                "temp_proper_code",
            )
        if quantity == "pressure":
            gamma = self.cosmology.gamma
            return self._target(values / a ** (3.0 * gamma), "pre_proper_code")
        if quantity == "velocity":
            if x_comoving_code is None:
                raise ValueError(
                    "supercomoving velocity conversion requires x_comoving_code"
                )
            hubble_code = hubble_parameter_code(
                self.code_units,
                self.cosmology.hubble_parameter_km_s_Mpc,
            )
            return self._target(
                hubble_code * a * np.asarray(x_comoving_code) + values / a,
                "vel_proper_code",
            )

        raise ValueError(f"unsupported proper conversion for {quantity!r}")

    def to_comoving(self, *, x_comoving_code=None):
        """Return a new array converted to the comoving representation."""
        source = self.field_spec
        if source.representation in {"comoving", "supercomoving"}:
            return self

        a = self.cosmology.scale_factor
        values = np.asarray(self)
        quantity = source.quantity

        if source.representation == "proper" and quantity == "radius":
            return self._target(values / a, "boundary_comoving_code")
        if source.representation == "proper" and quantity == "mass_density":
            return self._target(values * a**3, "rho_comoving_code")
        if source.representation == "proper" and quantity == "temperature":
            gamma = self.cosmology.gamma
            return self._target(
                values * a ** (3.0 * (gamma - 1.0)),
                "temp_supercomoving_code",
            )
        if source.representation == "proper" and quantity == "pressure":
            gamma = self.cosmology.gamma
            return self._target(
                values * a ** (3.0 * gamma),
                "pre_supercomoving_code",
            )
        if source.representation == "proper" and quantity == "velocity":
            if x_comoving_code is None:
                raise ValueError(
                    "proper velocity conversion requires x_comoving_code"
                )
            hubble_code = hubble_parameter_code(
                self.code_units,
                self.cosmology.hubble_parameter_km_s_Mpc,
            )
            return self._target(
                a * (values - hubble_code * a * np.asarray(x_comoving_code)),
                "vel_supercomoving_code",
            )
        raise ValueError(f"unsupported comoving conversion for {quantity!r}")

    def to_cgs(self):
        """Return the current representation as a cgs ``unyt_array``."""
        # Construct an ordinary unyt array so unyt's conversion helpers do not
        # attempt to call RadArray's metadata-requiring constructor.
        return unyt.unyt_array(self.value, self.units).in_cgs()

    def to_value(self, units=None, equivalence=None):
        """Return numerical values after an explicit unit conversion.

        ``unyt_array.to_value`` internally reconstructs ``type(self)`` with
        ``bypass_validation``.  ``RadArray`` carries additional metadata and
        intentionally does not accept that constructor argument, so delegate
        conversion through an ordinary ``unyt_array`` instead.
        """
        ordinary_array = unyt.unyt_array(np.asarray(self, dtype=float), self.units)
        return ordinary_array.to_value(units, equivalence=equivalence)

    def to(self, units, equivalence=None):
        """Return an ordinary unit-bearing array in ``units``.

        Delegate through ``unyt_array`` because unyt's default implementation
        reconstructs subclasses with constructor arguments that ``RadArray``
        intentionally does not accept.
        """
        ordinary_array = unyt.unyt_array(np.asarray(self, dtype=float), self.units)
        return ordinary_array.to(units, equivalence=equivalence)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        rad_inputs = [value for value in inputs if isinstance(value, RadArray)]
        if len(rad_inputs) > 1:
            first = rad_inputs[0]
            for other in rad_inputs[1:]:
                if first.representation != other.representation:
                    raise RepresentationMismatchError(
                        "RadArray arithmetic requires matching representations"
                    )
                if first.cosmology != other.cosmology:
                    raise ValueError(
                        "RadArray arithmetic requires matching cosmology contexts"
                    )

        # Perform arithmetic on the stored code values.  Delegating directly
        # to unyt would first convert the operands to cgs, which is correct for
        # ordinary unyt arrays but would change the numerical code values that
        # RadArray promises to preserve.
        if method == "__call__" and rad_inputs and "out" not in kwargs:
            if ufunc in (np.add, np.subtract) and len(rad_inputs) == 2:
                left, right = inputs
                result_value = getattr(np, ufunc.__name__)(
                    left.value, right.value
                )
                return RadArray(
                    result_value,
                    code_units=rad_inputs[0].code_units,
                    field_spec=rad_inputs[0].field_spec,
                    cosmology=rad_inputs[0].cosmology,
                )
            if ufunc in (np.multiply, np.true_divide, np.divide) and len(rad_inputs) == 2:
                left, right = inputs
                if ufunc is np.multiply:
                    result_value = left.value * right.value
                    result_units = left.units * right.units
                    dimension_operation = lambda a, b: a + b
                else:
                    result_value = left.value / right.value
                    result_units = left.units / right.units
                    dimension_operation = lambda a, b: a - b
                first, second = rad_inputs
                dimensions = tuple(
                    dimension_operation(a, b)
                    for a, b in zip(
                        first.field_spec.dimensions,
                        second.field_spec.dimensions,
                    )
                )
                result_spec = FieldSpec(
                    quantity=f"derived_{ufunc.__name__}",
                    dimensions=dimensions,
                    representation=first.representation,
                    coordinate_frame=first.field_spec.coordinate_frame,
                    code_unit_cgs=float(
                        unyt.unyt_quantity(1.0, result_units).in_cgs().value
                    ),
                    physical_relation="derived from RadArray arithmetic",
                    cosmology=first.cosmology.cosmology,
                    scale_factor=first.cosmology.scale_factor,
                    hubble_parameter_km_s_Mpc=(
                        first.cosmology.hubble_parameter_km_s_Mpc
                    ),
                )
                result = RadArray(
                    result_value,
                    code_units=first.code_units,
                    field_spec=result_spec,
                    cosmology=first.cosmology,
                )
                result.units = result_units
                return result
            if ufunc in (np.multiply, np.true_divide, np.divide) and len(rad_inputs) == 1:
                first = rad_inputs[0]
                other = inputs[1] if inputs[0] is first else inputs[0]
                result_value = (
                    first.value * other
                    if ufunc is np.multiply
                    else first.value / other
                    if inputs[0] is first
                    else other / first.value
                )
                return RadArray(
                    result_value,
                    code_units=first.code_units,
                    field_spec=first.field_spec,
                    cosmology=first.cosmology,
                )

        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if not isinstance(result, unyt.unyt_array) or not rad_inputs:
            return result
        if method != "__call__":
            return result

        first = rad_inputs[0]
        if len(rad_inputs) == 1:
            result_spec = first.field_spec
        elif ufunc in (np.add, np.subtract):
            result_spec = first.field_spec
        else:
            if ufunc is np.multiply:
                dimensions = tuple(
                    left + right
                    for left, right in zip(
                        rad_inputs[0].field_spec.dimensions,
                        rad_inputs[1].field_spec.dimensions,
                    )
                )
            elif ufunc in (np.true_divide, np.divide, np.floor_divide):
                dimensions = tuple(
                    left - right
                    for left, right in zip(
                        rad_inputs[0].field_spec.dimensions,
                        rad_inputs[1].field_spec.dimensions,
                    )
                )
            else:
                return result
            result_spec = FieldSpec(
                quantity=f"derived_{ufunc.__name__}",
                dimensions=dimensions,
                representation=first.representation,
                coordinate_frame=first.field_spec.coordinate_frame,
                code_unit_cgs=float(
                    unyt.unyt_quantity(1.0, result.units).in_cgs().value
                ),
                physical_relation="derived from RadArray arithmetic",
                cosmology=first.cosmology.cosmology,
                scale_factor=first.cosmology.scale_factor,
                hubble_parameter_km_s_Mpc=(
                    first.cosmology.hubble_parameter_km_s_Mpc
                ),
            )
        wrapped = RadArray(
            result.value,
            code_units=first.code_units,
            field_spec=result_spec,
            cosmology=first.cosmology,
        )
        wrapped.units = result.units
        return wrapped


class RadQuantity(unyt.unyt_quantity):
    """Scalar counterpart to :class:`RadArray`."""

    def __new__(
        cls,
        value,
        *,
        code_units,
        field_spec,
        cosmology,
    ):
        if not isinstance(field_spec, FieldSpec):
            raise TypeError("field_spec must be a FieldSpec")
        if not isinstance(cosmology, CosmologyContext):
            raise TypeError("cosmology must be a CosmologyContext")
        code_unit = _code_unit_for_spec(code_units, field_spec)
        obj = super().__new__(cls, value, code_unit)
        obj.code_units = code_units
        obj.field_spec = field_spec
        obj.cosmology = cosmology
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.code_units = getattr(obj, "code_units", None)
        self.field_spec = getattr(obj, "field_spec", None)
        self.cosmology = getattr(obj, "cosmology", None)

    @property
    def representation(self):
        return self.field_spec.representation

    def _target(self, value, field_name):
        hubble = (
            self.cosmology.hubble_parameter_km_s_Mpc
            if field_name == "vel_supercomoving_code"
            else None
        )
        target_spec = make_field_spec(
            field_name,
            self.code_units,
            cosmology=self.cosmology.cosmology,
            scale_factor=self.cosmology.scale_factor,
            hubble_parameter_km_s_Mpc=hubble,
        )
        return RadQuantity(
            value,
            code_units=self.code_units,
            field_spec=target_spec,
            cosmology=self.cosmology,
        )

    def to_proper(self, *, x_comoving_code=None):
        """Return a new scalar converted to the proper-code representation."""
        source = self.field_spec
        if source.representation == "proper":
            return self

        a = self.cosmology.scale_factor
        value = float(self.value)
        quantity = source.quantity
        if quantity == "radius":
            return self._target(value * a, "boundary_proper_code")
        if quantity == "mass_density":
            return self._target(value / a**3, "rho_proper_code")
        if quantity == "temperature":
            return self._target(
                value / a ** (3.0 * (self.cosmology.gamma - 1.0)),
                "temp_proper_code",
            )
        if quantity == "pressure":
            return self._target(
                value / a ** (3.0 * self.cosmology.gamma),
                "pre_proper_code",
            )
        if quantity == "velocity":
            if x_comoving_code is None:
                raise ValueError(
                    "supercomoving velocity conversion requires x_comoving_code"
                )
            hubble_code = hubble_parameter_code(
                self.code_units,
                self.cosmology.hubble_parameter_km_s_Mpc,
            )
            return self._target(
                hubble_code * a * float(x_comoving_code) + value / a,
                "vel_proper_code",
            )
        raise ValueError(f"unsupported proper conversion for {quantity!r}")

    def to_comoving(self, *, x_comoving_code=None):
        """Return a new scalar converted to comoving or supercomoving form."""
        source = self.field_spec
        if source.representation in {"comoving", "supercomoving"}:
            return self

        a = self.cosmology.scale_factor
        value = float(self.value)
        quantity = source.quantity
        if quantity == "radius":
            return self._target(value / a, "boundary_comoving_code")
        if quantity == "mass_density":
            return self._target(value * a**3, "rho_comoving_code")
        if quantity == "temperature":
            return self._target(
                value * a ** (3.0 * (self.cosmology.gamma - 1.0)),
                "temp_supercomoving_code",
            )
        if quantity == "pressure":
            return self._target(
                value * a ** (3.0 * self.cosmology.gamma),
                "pre_supercomoving_code",
            )
        if quantity == "velocity":
            if x_comoving_code is None:
                raise ValueError(
                    "proper velocity conversion requires x_comoving_code"
                )
            hubble_code = hubble_parameter_code(
                self.code_units,
                self.cosmology.hubble_parameter_km_s_Mpc,
            )
            return self._target(
                a * (value - hubble_code * a * float(x_comoving_code)),
                "vel_supercomoving_code",
            )
        raise ValueError(f"unsupported comoving conversion for {quantity!r}")

    def to_cgs(self):
        """Return the current scalar as an ordinary cgs ``unyt_quantity``."""
        return unyt.unyt_quantity(self.value, self.units).in_cgs()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        rad_inputs = [value for value in inputs if isinstance(value, (RadArray, RadQuantity))]
        if len(rad_inputs) > 1:
            first = rad_inputs[0]
            for other in rad_inputs[1:]:
                if first.representation != other.representation:
                    raise RepresentationMismatchError(
                        "RadQuantity arithmetic requires matching representations"
                    )
                if first.cosmology != other.cosmology:
                    raise ValueError(
                        "RadQuantity arithmetic requires matching cosmology contexts"
                    )
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
