"""Shared helpers for internal code-unit handling."""

from dataclasses import dataclass

import numpy as np
import unyt

from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS


PHOTON_FLUX_UNIT = 1.0 / (unyt.cm**2 * unyt.s)
PHOTON_RATE_UNIT = 1.0 / unyt.s
PHOTON_DENSITY_UNIT = 1.0 / unyt.cm**3

CGS_LENGTH_UNIT = unyt.cm
CGS_AREA_UNIT = CGS_LENGTH_UNIT**2
CGS_VOLUME_UNIT = CGS_LENGTH_UNIT**3
CGS_MASS_DENSITY_UNIT = unyt.g / CGS_VOLUME_UNIT
CGS_NUMBER_DENSITY_UNIT = 1.0 / CGS_VOLUME_UNIT
CGS_RATE_UNIT = 1.0 / unyt.s
CGS_PHOTON_FLUX_UNIT = 1.0 / (CGS_AREA_UNIT * unyt.s)

def _as_cgs_float(value, unit):
    if hasattr(value, "to_value"):
        return float(value.to_value(unit))
    return float(value)


def _code_units(par):
    return getattr(par, "code_units", getattr(par, "CodeUnits", None))


@dataclass(frozen=True)
class _CodeUnitGroup:
    target: str
    specs: tuple[tuple[str, str], ...]


_CODE_UNIT_GROUPS = (
    _CodeUnitGroup(
        'mesh',
        (
            ('boundary', 'length'),
            ('xdelta', 'length'),
            ('oneoverdx', 'length_inv'),
            ('coordinate', 'length'),
            ('area', 'area'),
            ('vol', 'volume'),
        ),
    ),
    _CodeUnitGroup(
        'fluid',
        (
            ('rho', 'density'),
            ('vel', 'velocity'),
            ('pre', 'pressure'),
            ('temp', 'temperature'),
            ('Mass', 'mass'),
            ('Mom', 'momentum'),
            ('Energy', 'energy'),
            ('ngamma', 'number_density'),
            ('cs', 'velocity'),
            ('vsignal', 'velocity'),
            ('flux', 'mass_flux'),
        ),
    ),
    _CodeUnitGroup(
        'par',
        (
            ('time', 'time'),
            ('timesim', 'time'),
            ('dtmin', 'time'),
            ('dtmax', 'time'),
            ('outdeltatime', 'time'),
            ('boxsize', 'length'),
            ('area', 'area'),
            ('rho_inflow', 'density'),
            ('rho_outflow', 'density'),
            ('vel_inflow', 'velocity'),
            ('vel_outflow', 'velocity'),
            ('temp_inflow', 'temperature'),
            ('temp_outflow', 'temperature'),
            ('hydrogen_ngamma_initial', 'number_density'),
            ('hydrogen_ngamma_inflow', 'number_density'),
            ('hydrogen_ngamma_outflow', 'number_density'),
            ('radiative_transfer_boundary_flux', 'photon_flux'),
            ('radiative_transfer_source_photon_rate', 'photon_rate'),
            ('hydrogen_sigma_gamma', 'area'),
            ('hydrogen_epsilon_gamma', 'energy'),
            ('hydrogen_alpha_B', 'alpha'),
            ('hydrogen_beta', 'alpha'),
            ('gravity_coordinate', 'length'),
            ('gravity_potential', 'potential'),
            ('gravity_acceleration', 'acceleration'),
            ('gravity_strength', 'acceleration'),
        ),
    ),
)


def to_quantity(value, unit):
    """Convert a quantity-like value to the supplied unit."""
    if value is None:
        return None
    if hasattr(value, "to"):
        return value.to(unit)
    return np.asarray(value, dtype=float) * unit


_to_code_quantity = to_quantity


def code_units_from_system(code):
    """Return the core unit mapping used for runtime conversions."""
    return {
        "length": code.unit_system["length"],
        "mass": code.unit_system["mass"],
        "time": code.unit_system["time"],
        "velocity": code.unit_system["velocity"],
        "temperature": code.unit_system["temperature"],
        "density": code.unit_system["density"],
        "pressure": code.unit_system["pressure"],
        "energy": code.unit_system["energy"],
    }


def to_code_value(value, unit):
    """Return a plain NumPy array in code units.

    When ``value`` carries units, it is converted to the supplied unit and the
    unit metadata is stripped. Plain arrays are treated as already being in
    code units and are returned unchanged as ``float`` arrays.
    """
    if value is None:
        return None
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(unit), dtype=float)
    return np.asarray(value, dtype=float)


def code_to_cgs_value(value, unit):
    """Return a plain NumPy array in cgs units.

    ``value`` may already be a quantity or may be a raw float/array expressed in
    code units. ``unit`` should be the corresponding code-unit quantity that
    represents one code unit in cgs.
    """
    if value is None:
        return None
    if hasattr(unit, "to_value"):
        scale = np.asarray(unit.to_value(unit.units), dtype=float)
        if hasattr(value, "to_value"):
            return np.asarray(value.to_value(unit.units), dtype=float)
        return np.asarray(value, dtype=float) * scale
    return np.asarray(value, dtype=float)


def cgs_to_code_value(value, unit):
    """Return a plain NumPy array in code units from cgs input."""
    if value is None:
        return None
    if hasattr(unit, "to_value"):
        scale = np.asarray(unit.to_value(unit.units), dtype=float)
        if hasattr(value, "to_value"):
            return np.asarray(value.to_value(unit.units), dtype=float) / scale
        return np.asarray(value, dtype=float) / scale
    return np.asarray(value, dtype=float)


def apply_code_unit_specs(obj, specs, units):
    """Apply code-unit conversions for each named attribute in ``specs``."""
    for attr, unit_key in specs:
        if hasattr(obj, attr):
            setattr(obj, attr, to_code_value(getattr(obj, attr), units[unit_key]))


def time_seconds(value, code_units=None):
    """Return a time value in seconds as a float."""
    if hasattr(value, "to_value"):
        return float(np.ravel(value.to_value(unyt.s))[0])
    if code_units is not None:
        return float(np.asarray(value, dtype=float) * code_unit_scales(code_units)["time_s"])
    return float(np.asarray(value, dtype=float))


def time_code_value(value, code_units):
    """Return a time value in code units as a float."""
    if hasattr(value, "to_value"):
        return float(np.asarray(value.to_value(code_units.time_unit), dtype=float))
    return float(np.asarray(value, dtype=float))


def code_unit_scales(code):
    """Return cgs scale factors for the supplied code-unit system."""
    if code is None:
        return None
    length_cm = float(code.length_in_cgs)
    mass_g = float(code.mass_in_cgs)
    velocity_cm_s = float(code.velocity_in_cgs)
    time_s = length_cm / velocity_cm_s
    area_cm2 = length_cm**2
    volume_cm3 = length_cm**3
    density_g_cm3 = mass_g / volume_cm3
    pressure_erg_cm3 = mass_g / (length_cm * time_s**2)
    energy_erg = mass_g * velocity_cm_s**2
    return {
        "length_cm": length_cm,
        "mass_g": mass_g,
        "velocity_cm_s": velocity_cm_s,
        "time_s": time_s,
        "temperature_K": float(code.temperature_in_cgs),
        "area_cm2": area_cm2,
        "volume_cm3": volume_cm3,
        "density_g_cm3": density_g_cm3,
        "pressure_erg_cm3": pressure_erg_cm3,
        "energy_erg": energy_erg,
        "specific_energy_erg_g": velocity_cm_s**2,
        "momentum_g_cm_s": mass_g * velocity_cm_s,
        "mass_flux_g_cm2_s": mass_g / (area_cm2 * time_s),
        "energy_flux_erg_cm2_s": energy_erg / (area_cm2 * time_s),
        "number_density_cm3": 1.0 / volume_cm3,
        "photon_flux_per_cm2_s": 1.0 / (area_cm2 * time_s),
        "photon_rate_per_s": 1.0 / time_s,
        "alpha_cm3_s": volume_cm3 / time_s,
        "acceleration_cm_s2": length_cm / time_s**2,
    }


def code_quantity_to_cgs(value, code, scale_key):
    """Convert a code-unit quantity or float to a cgs float array."""
    scales = code_unit_scales(code)
    if scales is None:
        return np.asarray(value, dtype=float)
    return np.asarray(value, dtype=float) * scales[scale_key]


def cgs_quantity_to_code(value, code, scale_key):
    """Convert a cgs float quantity to code units."""
    scales = code_unit_scales(code)
    if scales is None:
        return np.asarray(value, dtype=float)
    return np.asarray(value, dtype=float) / scales[scale_key]


def _gravitational_constant_code(code_units):
    """Return the gravitational constant in the supplied code units."""
    if code_units is None:
        raise ValueError("gravity helpers require code_units")
    return (
        GRAVITATIONAL_CONSTANT_CGS
        * code_units.mass_in_cgs
        / (code_units.length_in_cgs * code_units.velocity_in_cgs**2)
    )


def _potential_unit(code_units):
    """Return the gravitational potential unit for the supplied code system."""
    if code_units is None:
        return unyt.cm**2 / unyt.s**2
    return code_units.velocity_unit**2


def _acceleration_unit(code_units):
    """Return the gravitational acceleration unit for the supplied code system."""
    if code_units is None:
        return unyt.cm / unyt.s**2
    return code_units.length_unit / code_units.time_unit**2


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
