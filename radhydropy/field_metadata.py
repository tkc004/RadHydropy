"""Immutable metadata for dimensional, representation-aware fields."""

from dataclasses import dataclass
import math
from numbers import Real

import unyt


FIELD_DIMENSION_BASIS = (
    "mass",
    "length",
    "velocity",
    "current",
    "temperature",
)
FIELD_DIMENSION_BASIS_NAME = ",".join(FIELD_DIMENSION_BASIS)


_FIELD_DEFINITIONS = {
    "boundary": {
        "quantity": "radius", "dimensions": (0, 1, 0, 0, 0),
        "unit_property": "length_unit", "representation": "proper",
        "physical_relation": "physical = stored",
    },
    "boundary_proper_code": {
        "quantity": "radius", "dimensions": (0, 1, 0, 0, 0),
        "unit_property": "length_unit", "representation": "proper",
        "physical_relation": "physical = stored",
    },
    "boundary_comoving_code": {
        "quantity": "radius", "dimensions": (0, 1, 0, 0, 0),
        "unit_property": "length_unit", "representation": "comoving",
        "physical_relation": "physical = a * stored",
    },
    "rho_proper_code": {
        "quantity": "mass_density", "dimensions": (1, -3, 0, 0, 0),
        "unit_property": "density_unit", "representation": "proper",
        "physical_relation": "physical = stored",
    },
    "rho_comoving_code": {
        "quantity": "mass_density", "dimensions": (1, -3, 0, 0, 0),
        "unit_property": "density_unit", "representation": "comoving",
        "physical_relation": "physical = stored / a**3",
    },
    "vel_proper_code": {
        "quantity": "velocity", "dimensions": (0, 0, 1, 0, 0),
        "unit_property": "velocity_unit", "representation": "proper",
        "physical_relation": "physical = stored",
    },
    "vel_supercomoving_code": {
        "quantity": "velocity", "dimensions": (0, 0, 1, 0, 0),
        "unit_property": "velocity_unit", "representation": "supercomoving",
        "physical_relation": "physical = H*a*x + stored/a",
        "requires_hubble_parameter": True,
    },
    "temp_proper_code": {
        "quantity": "temperature", "dimensions": (0, 0, 0, 0, 1),
        "unit_property": "temperature_unit", "representation": "proper",
        "physical_relation": "physical = stored",
    },
    "temp_supercomoving_code": {
        "quantity": "temperature", "dimensions": (0, 0, 0, 0, 1),
        "unit_property": "temperature_unit", "representation": "supercomoving",
        "physical_relation": "physical = stored / a**(3*(gamma - 1))",
    },
    "pre_proper_code": {
        "quantity": "pressure", "dimensions": (1, -3, 2, 0, 0),
        "unit_property": "pressure_unit", "representation": "proper",
        "physical_relation": "physical = stored",
    },
    "pre_supercomoving_code": {
        "quantity": "pressure", "dimensions": (1, -3, 2, 0, 0),
        "unit_property": "pressure_unit", "representation": "supercomoving",
        "physical_relation": "physical = stored / a**(3*gamma)",
    },
    "Mass_code": {
        "quantity": "mass", "dimensions": (1, 0, 0, 0, 0),
        "unit_property": "mass_unit", "representation": "physical",
        "physical_relation": "physical = stored",
    },
    "Energy_code": {
        "quantity": "energy", "dimensions": (1, 0, 2, 0, 0),
        "unit_property": "energy_unit", "representation": "physical",
        "physical_relation": "physical = stored",
    },
    "InternalEnergy_code": {
        "quantity": "energy", "dimensions": (1, 0, 2, 0, 0),
        "unit_property": "energy_unit", "representation": "physical",
        "physical_relation": "physical = stored",
    },
    "GravitationalPotentialEnergy_code": {
        "quantity": "energy", "dimensions": (1, 0, 2, 0, 0),
        "unit_property": "energy_unit", "representation": "physical",
        "physical_relation": "physical = stored",
    },
    "AngularMomentum_code": {
        "quantity": "angular_momentum", "dimensions": (1, 1, 1, 0, 0),
        "unit_property": "angular_momentum_unit", "representation": "physical",
        "physical_relation": "physical = stored",
    },
    "specific_angular_momentum_code": {
        "quantity": "specific_angular_momentum", "dimensions": (0, 1, 1, 0, 0),
        "unit_property": "specific_angular_momentum_unit", "representation": "physical",
        "physical_relation": "physical = stored",
    },
    "ngamma_code": {
        "quantity": "number_density", "dimensions": (0, -3, 0, 0, 0),
        "unit_property": "number_density_unit", "representation": "physical",
        "physical_relation": "physical = stored",
    },
}


def _code_unit_cgs(code_units, property_name):
    """Return the cgs value of one field code unit."""
    if property_name == "angular_momentum_unit":
        unit = code_units.mass_unit * code_units.length_unit * code_units.velocity_unit
    elif property_name == "specific_angular_momentum_unit":
        unit = code_units.length_unit * code_units.velocity_unit
    else:
        unit = getattr(code_units, property_name)
    cgs_unit = {
        "length_unit": unyt.cm,
        "density_unit": unyt.g / unyt.cm**3,
        "velocity_unit": unyt.cm / unyt.s,
        "temperature_unit": unyt.K,
        "pressure_unit": unyt.erg / unyt.cm**3,
        "mass_unit": unyt.g,
        "energy_unit": unyt.erg,
        "angular_momentum_unit": unyt.g * unyt.cm**2 / unyt.s,
        "specific_angular_momentum_unit": unyt.cm**2 / unyt.s,
        "number_density_unit": 1.0 / unyt.cm**3,
    }[property_name]
    return float(unit.to_value(cgs_unit))


def hubble_parameter_code(code_units, hubble_parameter_km_s_Mpc):
    """Convert an observational Hubble parameter to inverse code time."""
    unit_hubble_km_s_Mpc = (
        code_units.velocity_unit.to_value(unyt.km / unyt.s)
        / code_units.length_unit.to_value(unyt.Mpc)
    )
    return float(hubble_parameter_km_s_Mpc) / float(unit_hubble_km_s_Mpc)


def field_spec(
    field_name,
    code_units,
    *,
    cosmology=None,
    scale_factor=1.0,
    scale_factor_power=0.0,
    conversion_factor=1.0,
    hubble_parameter_km_s_Mpc=None,
):
    """Return the canonical :class:`FieldSpec` for a runtime field.

    ``code_units`` must be a ``CodeUnits`` instance. Cosmological values are
    supplied by the snapshot or runtime state because they cannot be inferred
    from the base unit system alone.
    """
    try:
        definition = _FIELD_DEFINITIONS[field_name]
    except KeyError as exc:
        raise ValueError(f"no canonical FieldSpec is defined for {field_name!r}") from exc
    unit_property = definition["unit_property"]
    if not hasattr(code_units, unit_property) and unit_property not in {
        "angular_momentum_unit",
        "specific_angular_momentum_unit",
    }:
        raise TypeError("field_spec requires a RadHydropy CodeUnits instance")
    if (
        definition.get("requires_hubble_parameter")
        and hubble_parameter_km_s_Mpc is None
    ):
        raise ValueError(
            f"{field_name} requires hubble_parameter_km_s_Mpc for physical conversion"
        )
    return FieldSpec(
        quantity=definition["quantity"],
        dimensions=definition["dimensions"],
        representation=definition["representation"],
        coordinate_frame=(
            "comoving"
            if definition["representation"] in {"comoving", "supercomoving"}
            else "physical"
        ),
        code_unit_cgs=_code_unit_cgs(code_units, definition["unit_property"]),
        physical_relation=definition["physical_relation"],
        cosmology=cosmology,
        scale_factor=scale_factor,
        scale_factor_power=scale_factor_power,
        conversion_factor=conversion_factor,
        hubble_parameter_km_s_Mpc=hubble_parameter_km_s_Mpc,
    )


@dataclass(frozen=True)
class FieldSpec:
    """Describe the units and representation of a serialized field.

    ``dimensions`` uses the five RadHydropy code-unit bases in this order:
    ``(mass, length, velocity, current, temperature)``.  The values are
    exponents, so mass density is ``(1, -3, 0, 0, 0)`` and energy is
    ``(1, 0, 2, 0, 0)``.

    ``code_unit_cgs`` is the cgs value of one unit in the field's code-unit
    representation.  The descriptor is deliberately independent of HDF5 so
    it can be used by runtime state, readers, writers, and diagnostics.
    """

    quantity: str
    dimensions: tuple[int, int, int, int, int]
    representation: str
    coordinate_frame: str
    code_unit_cgs: float
    physical_relation: str
    dimension_basis: str = FIELD_DIMENSION_BASIS_NAME
    storage_unit: str = "code"
    cosmology: str | None = None
    scale_factor: float = 1.0
    scale_factor_power: float = 0.0
    conversion_factor: float = 1.0
    hubble_parameter_km_s_Mpc: float | None = None

    def __post_init__(self):
        if not isinstance(self.quantity, str) or not self.quantity:
            raise ValueError("FieldSpec.quantity must be a non-empty string")
        if self.dimension_basis != FIELD_DIMENSION_BASIS_NAME:
            raise ValueError(
                "FieldSpec.dimension_basis must be "
                f"{FIELD_DIMENSION_BASIS_NAME!r}"
            )
        if self.storage_unit not in {"code", "cgs"}:
            raise ValueError(
                "FieldSpec.storage_unit must be either 'code' or 'cgs'"
            )
        try:
            dimensions = tuple(self.dimensions)
        except TypeError as exc:
            raise TypeError("FieldSpec.dimensions must be a five-item sequence") from exc
        if len(dimensions) != len(FIELD_DIMENSION_BASIS):
            raise ValueError(
                "FieldSpec.dimensions must contain five exponents ordered as "
                "(mass, length, velocity, current, temperature)"
            )
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in dimensions):
            raise TypeError("FieldSpec.dimensions must contain numeric exponents")
        if any(float(value) != int(value) for value in dimensions):
            raise ValueError("FieldSpec.dimensions must contain integer exponents")
        dimensions = tuple(int(value) for value in dimensions)
        object.__setattr__(self, "dimensions", dimensions)

        if not isinstance(self.representation, str) or not self.representation:
            raise ValueError("FieldSpec.representation must be a non-empty string")
        if not isinstance(self.coordinate_frame, str) or not self.coordinate_frame:
            raise ValueError("FieldSpec.coordinate_frame must be a non-empty string")
        if not isinstance(self.physical_relation, str) or not self.physical_relation:
            raise ValueError("FieldSpec.physical_relation must be a non-empty string")
        if self.cosmology is not None and (
            not isinstance(self.cosmology, str) or not self.cosmology
        ):
            raise ValueError("FieldSpec.cosmology must be a non-empty string or None")
        if isinstance(self.scale_factor, bool) or not isinstance(self.scale_factor, Real):
            raise TypeError("FieldSpec.scale_factor must be a positive real number")
        scale_factor = float(self.scale_factor)
        if not math.isfinite(scale_factor) or scale_factor <= 0.0:
            raise ValueError("FieldSpec.scale_factor must be finite and positive")
        object.__setattr__(self, "scale_factor", scale_factor)
        if isinstance(self.scale_factor_power, bool) or not isinstance(self.scale_factor_power, Real):
            raise TypeError("FieldSpec.scale_factor_power must be a real number")
        scale_factor_power = float(self.scale_factor_power)
        if not math.isfinite(scale_factor_power):
            raise ValueError("FieldSpec.scale_factor_power must be finite")
        object.__setattr__(self, "scale_factor_power", scale_factor_power)
        if isinstance(self.conversion_factor, bool) or not isinstance(self.conversion_factor, Real):
            raise TypeError("FieldSpec.conversion_factor must be a positive real number")
        conversion_factor = float(self.conversion_factor)
        if not math.isfinite(conversion_factor) or conversion_factor <= 0.0:
            raise ValueError("FieldSpec.conversion_factor must be finite and positive")
        object.__setattr__(self, "conversion_factor", conversion_factor)
        if self.hubble_parameter_km_s_Mpc is not None:
            if isinstance(self.hubble_parameter_km_s_Mpc, bool) or not isinstance(
                self.hubble_parameter_km_s_Mpc, Real
            ):
                raise TypeError(
                    "FieldSpec.hubble_parameter_km_s_Mpc must be real or None"
                )
            hubble_parameter_km_s_Mpc = float(self.hubble_parameter_km_s_Mpc)
            if not math.isfinite(hubble_parameter_km_s_Mpc):
                raise ValueError(
                    "FieldSpec.hubble_parameter_km_s_Mpc must be finite"
                )
            object.__setattr__(
                self, "hubble_parameter_km_s_Mpc", hubble_parameter_km_s_Mpc
            )
        if isinstance(self.code_unit_cgs, bool) or not isinstance(self.code_unit_cgs, Real):
            raise TypeError("FieldSpec.code_unit_cgs must be a positive real number")
        code_unit_cgs = float(self.code_unit_cgs)
        if not math.isfinite(code_unit_cgs) or code_unit_cgs <= 0.0:
            raise ValueError("FieldSpec.code_unit_cgs must be finite and positive")
        object.__setattr__(self, "code_unit_cgs", code_unit_cgs)

    def to_metadata(self):
        """Return HDF5-attribute-compatible metadata for this field."""
        return {
            "quantity": self.quantity,
            "dimension_basis": self.dimension_basis,
            "storage_unit": self.storage_unit,
            "dimensions": self.dimensions,
            "representation": self.representation,
            "coordinate_frame": self.coordinate_frame,
            "code_unit_cgs": self.code_unit_cgs,
            "physical_relation": self.physical_relation,
            "cosmology": self.cosmology,
            "scale_factor": self.scale_factor,
            "scale_factor_power": self.scale_factor_power,
            "conversion_factor": self.conversion_factor,
            "hubble_parameter_km_s_Mpc": self.hubble_parameter_km_s_Mpc,
        }

    @classmethod
    def from_metadata(cls, metadata):
        """Build a ``FieldSpec`` from its metadata mapping."""
        required = {
            "quantity",
            "dimensions",
            "representation",
            "coordinate_frame",
            "code_unit_cgs",
            "storage_unit",
            "physical_relation",
        }
        missing = required.difference(metadata)
        if missing:
            raise ValueError(
                "FieldSpec metadata is missing: " + ", ".join(sorted(missing))
            )
        return cls(
            quantity=metadata["quantity"],
            dimension_basis=metadata.get("dimension_basis", FIELD_DIMENSION_BASIS_NAME),
            storage_unit=metadata["storage_unit"],
            dimensions=metadata["dimensions"],
            representation=metadata["representation"],
            coordinate_frame=metadata["coordinate_frame"],
            code_unit_cgs=metadata["code_unit_cgs"],
            physical_relation=metadata["physical_relation"],
            cosmology=metadata.get("cosmology"),
            scale_factor=metadata.get("scale_factor", 1.0),
            scale_factor_power=metadata.get("scale_factor_power", 0.0),
            conversion_factor=metadata.get("conversion_factor", 1.0),
            hubble_parameter_km_s_Mpc=metadata.get("hubble_parameter_km_s_Mpc"),
        )
