"""Shared helpers for internal code-unit handling."""

from dataclasses import dataclass

import numpy as np
import unyt


PHOTON_FLUX_UNIT = 1.0 / (unyt.cm**2 * unyt.s)
PHOTON_RATE_UNIT = 1.0 / unyt.s
PHOTON_DENSITY_UNIT = 1.0 / unyt.cm**3
PHOTON_ABSORPTION_RATE_UNIT = 1.0 / (unyt.cm**3 * unyt.s)


def _as_cgs_float(value, unit):
    if hasattr(value, "to_value"):
        return float(value.to_value(unit))
    return float(value)


def _code_units(par):
    return getattr(par, "code_units", getattr(par, "CodeUnits", None))


def _to_code_quantity(value, unit):
    if value is None:
        return None
    if hasattr(value, "to"):
        return value.to(unit)
    return np.asarray(value, dtype=float) * unit


def photon_number_density(ngamma):
    if ngamma is None:
        return 0.0 * PHOTON_DENSITY_UNIT
    if hasattr(ngamma, "to"):
        return ngamma.to(PHOTON_DENSITY_UNIT)
    return np.asarray(ngamma, dtype=float) * PHOTON_DENSITY_UNIT


def _as_photon_flux(value):
    if value is None:
        return 0.0 * PHOTON_FLUX_UNIT
    if hasattr(value, "to"):
        return value.to(PHOTON_FLUX_UNIT)
    return np.asarray(value, dtype=float) * PHOTON_FLUX_UNIT


def _as_photon_rate(value):
    if value is None:
        return 0.0 * PHOTON_RATE_UNIT
    if hasattr(value, "to"):
        return value.to(PHOTON_RATE_UNIT)
    return np.asarray(value, dtype=float) * PHOTON_RATE_UNIT


def _optional_photon_quantity(value, default, units):
    if value is None:
        value = default
    if hasattr(value, "to"):
        return value.to(units)
    return np.asarray(value, dtype=float) * units


@dataclass(frozen=True)
class CodeUnits:
    """Internal unit system used to run the hydro solver in float space."""

    name: str
    mass_in_cgs: float
    length_in_cgs: float
    velocity_in_cgs: float
    current_in_cgs: float
    temperature_in_cgs: float
    unit_system: unyt.unit_systems.UnitSystem

    @property
    def time_in_cgs(self):
        return self.length_in_cgs / self.velocity_in_cgs

    @property
    def mass_unit(self):
        return self.mass_in_cgs * unyt.g

    @property
    def length_unit(self):
        return self.length_in_cgs * unyt.cm

    @property
    def time_unit(self):
        return self.time_in_cgs * unyt.s

    @property
    def velocity_unit(self):
        return self.velocity_in_cgs * unyt.cm / unyt.s

    @property
    def current_unit(self):
        return self.current_in_cgs * unyt.A

    @property
    def temperature_unit(self):
        return self.temperature_in_cgs * unyt.K

    @property
    def area_unit(self):
        return self.length_unit ** 2

    @property
    def volume_unit(self):
        return self.length_unit ** 3

    @property
    def density_unit(self):
        return self.mass_unit / self.volume_unit

    @property
    def pressure_unit(self):
        return self.mass_unit / (self.length_unit * self.time_unit ** 2)

    @property
    def energy_unit(self):
        return self.mass_unit * self.velocity_unit ** 2

    @property
    def specific_energy_unit(self):
        return self.energy_unit / self.mass_unit

    @property
    def momentum_unit(self):
        return self.mass_unit * self.velocity_unit

    @property
    def mass_flux_unit(self):
        return self.mass_unit / (self.length_unit ** 2 * self.time_unit)

    @property
    def momentum_flux_unit(self):
        return self.pressure_unit

    @property
    def energy_flux_unit(self):
        return self.energy_unit / (self.length_unit ** 2 * self.time_unit)

    @property
    def number_density_unit(self):
        return 1.0 / self.volume_unit

    @property
    def proton_mass_code(self):
        return float(unyt.mp.to_value(self.mass_unit))

    @property
    def boltzmann_code(self):
        return float(unyt.kb.to_value(self.energy_unit / self.temperature_unit))

    @property
    def speed_of_light_code(self):
        return float(unyt.c.to_value(self.velocity_unit))

    def to_value(self, quantity, unit):
        if hasattr(quantity, "to_value"):
            return np.asarray(quantity.to_value(unit), dtype=float)
        return np.asarray(quantity, dtype=float)

    def from_value(self, values, unit):
        return np.asarray(values, dtype=float) * unit

    @classmethod
    def from_mapping(cls, mapping=None, name="code"):
        """Build a code-unit system from a YAML block or a UnitSystem."""
        if isinstance(mapping, cls):
            return mapping
        if isinstance(mapping, unyt.unit_systems.UnitSystem):
            base_units = mapping.base_units
            length_unit = base_units[unyt.dimensions.length]
            mass_unit = base_units[unyt.dimensions.mass]
            time_unit = base_units[unyt.dimensions.time]
            temperature_unit = base_units[unyt.dimensions.temperature]
            current_unit = base_units[unyt.dimensions.current_mks]
            return cls(
                name=getattr(mapping, "name", name),
                mass_in_cgs=_as_cgs_float(mass_unit, unyt.g),
                length_in_cgs=_as_cgs_float(length_unit, unyt.cm),
                velocity_in_cgs=_as_cgs_float(length_unit / time_unit, unyt.cm / unyt.s),
                current_in_cgs=_as_cgs_float(current_unit, unyt.A),
                temperature_in_cgs=_as_cgs_float(temperature_unit, unyt.K),
                unit_system=mapping,
            )

        data = dict(mapping or {})
        internal = data.get("InternalUnitSystem", data)
        if not isinstance(internal, dict):
            raise TypeError(
                "CodeUnits must be built from a mapping, a UnitSystem, or None"
            )

        mass_in_cgs = _as_cgs_float(
            internal.get("UnitMass_in_cgs", internal.get("mass_in_cgs", 1.0)),
            unyt.g,
        )
        length_in_cgs = _as_cgs_float(
            internal.get("UnitLength_in_cgs", internal.get("length_in_cgs", 1.0)),
            unyt.cm,
        )
        velocity_in_cgs = _as_cgs_float(
            internal.get("UnitVelocity_in_cgs", internal.get("velocity_in_cgs", 1.0)),
            unyt.cm / unyt.s,
        )
        current_in_cgs = _as_cgs_float(
            internal.get("UnitCurrent_in_cgs", internal.get("current_in_cgs", 1.0)),
            unyt.A,
        )
        temperature_in_cgs = _as_cgs_float(
            internal.get("UnitTemp_in_cgs", internal.get("temperature_in_cgs", 1.0)),
            unyt.K,
        )
        if mass_in_cgs <= 0.0 or length_in_cgs <= 0.0 or velocity_in_cgs <= 0.0:
            raise ValueError("CodeUnits mass, length, and velocity scales must be positive")
        time_in_cgs = length_in_cgs / velocity_in_cgs
        unit_system = unyt.UnitSystem(
            internal.get("name", data.get("name", name)),
            length_in_cgs * unyt.cm,
            mass_in_cgs * unyt.g,
            time_in_cgs * unyt.s,
            temperature_in_cgs * unyt.K,
            current_mks_unit=current_in_cgs * unyt.A,
        )
        return cls(
            name=internal.get("name", data.get("name", name)),
            mass_in_cgs=mass_in_cgs,
            length_in_cgs=length_in_cgs,
            velocity_in_cgs=velocity_in_cgs,
            current_in_cgs=current_in_cgs,
            temperature_in_cgs=temperature_in_cgs,
            unit_system=unit_system,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "InternalUnitSystem": {
                "UnitMass_in_cgs": self.mass_in_cgs,
                "UnitLength_in_cgs": self.length_in_cgs,
                "UnitVelocity_in_cgs": self.velocity_in_cgs,
                "UnitCurrent_in_cgs": self.current_in_cgs,
                "UnitTemp_in_cgs": self.temperature_in_cgs,
            },
        }
