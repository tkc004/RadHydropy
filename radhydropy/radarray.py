# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Representation-aware numerical arrays for RadHydropy."""

import operator
from typing import Any

import numpy as np
import unyt

from radhydropy.cosmology.context import CosmologyContext
from radhydropy.field_metadata import (
    FieldSpec,
    hubble_parameter_code,
)
from radhydropy.field_metadata import (
    field_spec as make_field_spec,
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


def _code_unit_for_spec(code_units: Any, spec: Any) -> Any:
    if spec.quantity == "angular_momentum":
        return code_units.mass_unit * code_units.length_unit * code_units.velocity_unit
    if spec.quantity == "specific_angular_momentum":
        return code_units.length_unit * code_units.velocity_unit
    try:
        return getattr(code_units, _QUANTITY_UNIT_PROPERTIES[spec.quantity])
    except KeyError:
        return _unit_for_dimensions(code_units, spec.dimensions)


def _unit_for_dimensions(code_units: Any, dimensions: Any) -> Any:
    mass, length, velocity, current, temperature = dimensions
    return (
        code_units.mass_unit**mass
        * code_units.length_unit**length
        * code_units.velocity_unit**velocity
        * code_units.current_unit**current
        * code_units.temperature_unit**temperature
    )


def _convert_to_proper_values(
    source: Any,
    cosmology: Any,
    code_units: Any,
    values: Any,
    x_comoving_code: Any,
    proper_name: str,
) -> tuple[Any, str]:
    scale_factor = cosmology.scale_factor
    quantity = source.quantity
    if quantity == "radius":
        converted = values * scale_factor
        target_name = proper_name
    elif quantity == "mass_density":
        converted = values / scale_factor**3
        target_name = "rho_proper_code"
    elif quantity == "temperature":
        converted = values / scale_factor ** (3.0 * (cosmology.gamma - 1.0))
        target_name = "temp_proper_code"
    elif quantity == "number_density":
        converted = values / scale_factor**3
        target_name = "ngamma_proper_code"
    elif quantity == "pressure":
        converted = values / scale_factor ** (3.0 * cosmology.gamma)
        target_name = "pre_proper_code"
    elif quantity == "velocity":
        if x_comoving_code is None:
            raise ValueError("supercomoving velocity conversion requires x_comoving_code")
        hubble_code = hubble_parameter_code(
            code_units,
            cosmology.hubble_parameter_km_s_Mpc,
        )
        converted = hubble_code * scale_factor * x_comoving_code + values / scale_factor
        target_name = "vel_proper_code"
    elif quantity == "specific_angular_momentum":
        converted = values
        target_name = "specific_angular_momentum_proper_code"
    else:
        raise ValueError(f"unsupported proper conversion for {quantity!r}")
    return converted, target_name


def _convert_to_comoving_values(
    source: Any,
    cosmology: Any,
    code_units: Any,
    values: Any,
    x_comoving_code: Any,
    comoving_name: str,
) -> tuple[Any, str]:
    scale_factor = cosmology.scale_factor
    quantity = source.quantity
    if quantity == "radius":
        converted = values / scale_factor
        target_name = comoving_name
    elif quantity == "mass_density":
        converted = values * scale_factor**3
        target_name = "rho_comoving_code"
    elif quantity == "temperature":
        converted = values * scale_factor ** (3.0 * (cosmology.gamma - 1.0))
        target_name = "temp_supercomoving_code"
    elif quantity == "number_density":
        converted = values * scale_factor**3
        target_name = "ngamma_comoving_code"
    elif quantity == "pressure":
        converted = values * scale_factor ** (3.0 * cosmology.gamma)
        target_name = "pre_supercomoving_code"
    elif quantity == "velocity":
        if x_comoving_code is None:
            raise ValueError("proper velocity conversion requires x_comoving_code")
        hubble_code = hubble_parameter_code(
            code_units,
            cosmology.hubble_parameter_km_s_Mpc,
        )
        converted = scale_factor * (values - hubble_code * scale_factor * x_comoving_code)
        target_name = "vel_supercomoving_code"
    elif quantity == "specific_angular_momentum":
        converted = values
        target_name = "specific_angular_momentum_supercomoving_code"
    else:
        raise ValueError(f"unsupported comoving conversion for {quantity!r}")
    return converted, target_name


class RadArray(unyt.unyt_array):  # type: ignore[misc]
    """A code-unit ``unyt_array`` carrying field and cosmology metadata."""

    code_units: Any
    field_spec: Any
    cosmology: Any
    field_name: Any

    def __new__(
        cls,
        values: Any,
        *,
        code_units: Any,
        field_spec: Any,
        cosmology: Any,
        field_name: Any = None,
    ) -> Any:
        if not isinstance(field_spec, FieldSpec):
            raise TypeError("field_spec must be a FieldSpec")
        if not isinstance(cosmology, CosmologyContext):
            raise TypeError("cosmology must be a CosmologyContext")
        code_unit = _code_unit_for_spec(code_units, field_spec)
        obj = super().__new__(cls, values, code_unit)
        obj.code_units = code_units
        obj.field_spec = field_spec
        obj.cosmology = cosmology
        obj.field_name = field_name
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.code_units = getattr(obj, "code_units", None)
        self.field_spec = getattr(obj, "field_spec", None)
        self.cosmology = getattr(obj, "cosmology", None)
        self.field_name = getattr(obj, "field_name", None)

    @property
    def representation(self) -> Any:
        return self.field_spec.representation

    def _target(self, values: Any, field_name: str) -> Any:
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
            field_name=field_name,
        )

    def _representation_field_name(self, proper_name: str, comoving_name: str) -> tuple[str, str]:
        if self.field_name in {"radius_proper_code", "radius_comoving_code"}:
            return proper_name.replace("boundary_", "radius_"), comoving_name.replace(
                "boundary_",
                "radius_",
            )
        return proper_name, comoving_name

    def to_proper(self, *, x_comoving_code: Any = None) -> Any:
        """Return a new array converted to the proper-code representation."""
        source = self.field_spec
        if source.representation == "proper":
            return self
        values = np.asarray(self)
        proper_name, _ = self._representation_field_name(
            "boundary_proper_code",
            "boundary_comoving_code",
        )
        converted, target_name = _convert_to_proper_values(
            source,
            self.cosmology,
            self.code_units,
            values,
            None if x_comoving_code is None else np.asarray(x_comoving_code),
            proper_name,
        )
        return self._target(converted, target_name)

    def to_comoving(self, *, x_comoving_code: Any = None) -> Any:
        """Return a new array converted to the comoving representation."""
        source = self.field_spec
        if source.representation in {"comoving", "supercomoving"}:
            return self
        values = np.asarray(self)
        _, comoving_name = self._representation_field_name(
            "boundary_proper_code",
            "boundary_comoving_code",
        )
        converted, target_name = _convert_to_comoving_values(
            source,
            self.cosmology,
            self.code_units,
            values,
            None if x_comoving_code is None else np.asarray(x_comoving_code),
            comoving_name,
        )
        return self._target(converted, target_name)

    def to_cgs(self) -> Any:
        """Return the current representation as a cgs ``unyt_array``."""
        # Construct an ordinary unyt array so unyt's conversion helpers do not
        # attempt to call RadArray's metadata-requiring constructor.
        return unyt.unyt_array(self.value, self.units).in_cgs()

    def to_value(self, units: Any = None, equivalence: Any = None) -> Any:
        """Return numerical values after an explicit unit conversion.

        ``unyt_array.to_value`` internally reconstructs ``type(self)`` with
        ``bypass_validation``.  ``RadArray`` carries additional metadata and
        intentionally does not accept that constructor argument, so delegate
        conversion through an ordinary ``unyt_array`` instead.
        """
        ordinary_array = unyt.unyt_array(np.asarray(self, dtype=float), self.units)
        return ordinary_array.to_value(units, equivalence=equivalence)

    def to(self, units: Any, equivalence: Any = None) -> Any:
        """Return an ordinary unit-bearing array in ``units``.

        Delegate through ``unyt_array`` because unyt's default implementation
        reconstructs subclasses with constructor arguments that ``RadArray``
        intentionally does not accept.
        """
        ordinary_array = unyt.unyt_array(np.asarray(self, dtype=float), self.units)
        return ordinary_array.to(units, equivalence=equivalence)

    def __array_ufunc__(
        self, ufunc: Any, method: Any, *inputs: Any, **kwargs: Any,
    ) -> Any:
        rad_inputs = [value for value in inputs if isinstance(value, RadArray)]
        _validate_radarray_operands(rad_inputs)

        # Perform arithmetic on the stored code values.  Delegating directly
        # to unyt would first convert the operands to cgs, which is correct for
        # ordinary unyt arrays but would change the numerical code values that
        # RadArray promises to preserve.
        if method == "__call__" and rad_inputs and "out" not in kwargs:
            fast_result = _fast_radarray_operation(ufunc, inputs, rad_inputs)
            if fast_result is not None:
                return fast_result

        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if not isinstance(result, unyt.unyt_array) or not rad_inputs:
            return result
        if method != "__call__":
            return result

        first = rad_inputs[0]
        result_spec = _result_field_spec(ufunc, rad_inputs, result.units)
        if result_spec is None:
            return result
        wrapped = RadArray(
            result.value,
            code_units=first.code_units,
            field_spec=result_spec,
            cosmology=first.cosmology,
        )
        wrapped.units = result.units
        return wrapped


def _validate_radarray_operands(rad_inputs: list[Any]) -> None:
    if len(rad_inputs) <= 1:
        return
    first = rad_inputs[0]
    for other in rad_inputs[1:]:
        if first.representation != other.representation:
            raise RepresentationMismatchError(
                "RadArray arithmetic requires matching representations",
            )
        if first.cosmology != other.cosmology:
            raise ValueError("RadArray arithmetic requires matching cosmology contexts")


def _result_field_spec(ufunc: Any, rad_inputs: list[Any], result_units: Any) -> Any:
    first = rad_inputs[0]
    if len(rad_inputs) == 1 or ufunc in (np.add, np.subtract):
        return first.field_spec
    if ufunc is np.multiply:
        operation = operator.add
    elif ufunc in (np.true_divide, np.divide, np.floor_divide):
        operation = operator.sub
    else:
        return None
    dimensions = tuple(
        operation(left, right)
        for left, right in zip(
            rad_inputs[0].field_spec.dimensions,
            rad_inputs[1].field_spec.dimensions,
            strict=False,
        )
    )
    return FieldSpec(
        quantity=f"derived_{ufunc.__name__}",
        dimensions=dimensions,
        representation=first.representation,
        coordinate_frame=first.field_spec.coordinate_frame,
        code_unit_cgs=float(unyt.unyt_quantity(1.0, result_units).in_cgs().value),
        physical_relation="derived from RadArray arithmetic",
        cosmology=first.cosmology.cosmology,
        scale_factor=first.cosmology.scale_factor,
        hubble_parameter_km_s_Mpc=first.cosmology.hubble_parameter_km_s_Mpc,
    )


def _fast_radarray_operation(ufunc: Any, inputs: Any, rad_inputs: list[Any]) -> Any:
    if ufunc in (np.add, np.subtract) and len(rad_inputs) == 2:  # noqa: PLR2004
        left, right = inputs
        return RadArray(
            getattr(np, ufunc.__name__)(left.value, right.value),
            code_units=rad_inputs[0].code_units,
            field_spec=rad_inputs[0].field_spec,
            cosmology=rad_inputs[0].cosmology,
        )
    if ufunc not in (np.multiply, np.true_divide, np.divide):
        return None
    first = rad_inputs[0]
    if len(rad_inputs) == 1:
        other = inputs[1] if inputs[0] is first else inputs[0]
        result_value = (
            first.value * other
            if ufunc is np.multiply
            else (first.value / other if inputs[0] is first else other / first.value)
        )
        return RadArray(
            result_value,
            code_units=first.code_units,
            field_spec=first.field_spec,
            cosmology=first.cosmology,
        )
    left, right = inputs
    result_value = left.value * right.value if ufunc is np.multiply else left.value / right.value
    result_units = left.units * right.units if ufunc is np.multiply else left.units / right.units
    result = RadArray(
        result_value,
        code_units=first.code_units,
        field_spec=_result_field_spec(ufunc, rad_inputs, result_units),
        cosmology=first.cosmology,
    )
    result.units = result_units
    return result


class RadQuantity(unyt.unyt_quantity):  # type: ignore[misc]
    """Scalar counterpart to :class:`RadArray`."""

    code_units: Any
    field_spec: Any
    cosmology: Any
    field_name: Any

    def __new__(
        cls,
        value: Any,
        *,
        code_units: Any,
        field_spec: Any,
        cosmology: Any,
        field_name: Any = None,
    ) -> Any:
        if not isinstance(field_spec, FieldSpec):
            raise TypeError("field_spec must be a FieldSpec")
        if not isinstance(cosmology, CosmologyContext):
            raise TypeError("cosmology must be a CosmologyContext")
        code_unit = _code_unit_for_spec(code_units, field_spec)
        obj = super().__new__(cls, value, code_unit)
        obj.code_units = code_units
        obj.field_spec = field_spec
        obj.cosmology = cosmology
        obj.field_name = field_name
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.code_units = getattr(obj, "code_units", None)
        self.field_spec = getattr(obj, "field_spec", None)
        self.cosmology = getattr(obj, "cosmology", None)
        self.field_name = getattr(obj, "field_name", None)

    @property
    def representation(self) -> Any:
        return self.field_spec.representation

    def _target(self, value: Any, field_name: str) -> Any:
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
            field_name=field_name,
        )

    def _representation_field_name(self, proper_name: str, comoving_name: str) -> tuple[str, str]:
        if self.field_name in {"radius_proper_code", "radius_comoving_code"}:
            return proper_name.replace("boundary_", "radius_"), comoving_name.replace(
                "boundary_",
                "radius_",
            )
        return proper_name, comoving_name

    def to_proper(self, *, x_comoving_code: Any = None) -> Any:
        """Return a new scalar converted to the proper-code representation."""
        source = self.field_spec
        if source.representation == "proper":
            return self
        value = float(self.value)
        proper_name, _ = self._representation_field_name(
            "boundary_proper_code",
            "boundary_comoving_code",
        )
        converted, target_name = _convert_to_proper_values(
            source,
            self.cosmology,
            self.code_units,
            value,
            None if x_comoving_code is None else float(x_comoving_code),
            proper_name,
        )
        return self._target(converted, target_name)

    def to_comoving(self, *, x_comoving_code: Any = None) -> Any:
        """Return a new scalar converted to comoving or supercomoving form."""
        source = self.field_spec
        if source.representation in {"comoving", "supercomoving"}:
            return self
        value = float(self.value)
        _, comoving_name = self._representation_field_name(
            "boundary_proper_code",
            "boundary_comoving_code",
        )
        converted, target_name = _convert_to_comoving_values(
            source,
            self.cosmology,
            self.code_units,
            value,
            None if x_comoving_code is None else float(x_comoving_code),
            comoving_name,
        )
        return self._target(converted, target_name)

    def to_cgs(self) -> Any:
        """Return the current scalar as an ordinary cgs ``unyt_quantity``."""
        return unyt.unyt_quantity(self.value, self.units).in_cgs()

    def to_value(self, units: Any = None, equivalence: Any = None) -> Any:
        """Return the scalar value after an explicit unit conversion."""
        ordinary_quantity = unyt.unyt_quantity(float(self.value), self.units)
        return ordinary_quantity.to_value(units, equivalence=equivalence)

    def to(self, units: Any, equivalence: Any = None) -> Any:
        """Return an ordinary unit-bearing scalar in ``units``."""
        ordinary_quantity = unyt.unyt_quantity(float(self.value), self.units)
        return ordinary_quantity.to(units, equivalence=equivalence)

    def __array_ufunc__(
        self, ufunc: Any, method: Any, *inputs: Any, **kwargs: Any,
    ) -> Any:
        rad_inputs = [value for value in inputs if isinstance(value, (RadArray, RadQuantity))]
        if len(rad_inputs) > 1:
            first = rad_inputs[0]
            for other in rad_inputs[1:]:
                if first.representation != other.representation:
                    raise RepresentationMismatchError(
                        "RadQuantity arithmetic requires matching representations",
                    )
                if first.cosmology != other.cosmology:
                    raise ValueError(
                        "RadQuantity arithmetic requires matching cosmology contexts",
                    )
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
