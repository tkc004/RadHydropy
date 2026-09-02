"""Shared helpers for internal code-unit handling."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import unyt

from radhydropy.constants import GRAVITATIONAL_CONSTANT_CGS
from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS


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
    units = getattr(par, "units", None)
    if units is not None and getattr(units, "CodeUnits", None) is not None:
        return units.CodeUnits
    # Component objects such as EOS, Gravity, Fluid, and Mesh own their
    # code-unit system directly; this is not a Par-level configuration alias.
    return getattr(par, "CodeUnits", None)


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
            ('rho_code', 'density'),
            ('vel_code', 'velocity'),
            ('pre_code', 'pressure'),
            ('temp_code', 'temperature'),
            ('Mass_code', 'mass'),
            ('Mom_code', 'momentum'),
            ('Energy_code', 'energy'),
            ('AngularMomentum_code', 'angular_momentum'),
            ('specific_angular_momentum_code', 'specific_angular_momentum'),
            ('ngamma_code', 'number_density'),
            ('cs_code', 'velocity'),
            ('vsignal_code', 'velocity'),
            ('flux_code', 'mass_flux'),
        ),
    ),
    _CodeUnitGroup(
        'par',
        (
            ('time', 'time'),
            ('timesim', 'time'),
            ('initial_time', 'time'),
            ('time_interval', 'time'),
            ('dtmin', 'time'),
            ('dtmax', 'time'),
            ('chemistry_timestep', 'time'),
            ('evolution_timestep', 'time'),
            ('output_interval', 'time'),
            ('supercomoving_timestep', 'time'),
            ('relaxation_damping_time', 'time'),
            ('outdeltatime', 'time'),
            ('hydrogen_source_dtmin', 'time'),
            ('boxsize', 'length'),
            ('area', 'area'),
            ('selfgravity_softening', 'length'),
            ('selfgravity_boundary_acceleration', 'acceleration'),
            ('rho_inflow', 'density'),
            ('rho_outflow', 'density'),
            ('cfl_density_floor', 'density'),
            ('positivity_density_floor', 'density'),
            ('vel_inflow', 'velocity'),
            ('vel_outflow', 'velocity'),
            ('specific_angular_momentum_inflow', 'specific_angular_momentum'),
            ('specific_angular_momentum_outflow', 'specific_angular_momentum'),
            ('temperature', 'temperature'),
            ('temp_inflow', 'temperature'),
            ('temp_outflow', 'temperature'),
            ('cooling_temperature_floor', 'temperature'),
            ('hydro_temperature_floor', 'temperature'),
            ('cosmology_t_ref', 'time'),
            ('cosmology_hubble_ref', 'time_inv'),
            ('hydrogen_implicit_absolute_temperature_tolerance', 'temperature'),
            ('cmb_temperature_0', 'temperature'),
            ('hydrogen_photon_energy', 'energy'),
            ('hydrogen_ngamma_initial', 'number_density'),
            ('hydrogen_ngamma_inflow', 'number_density'),
            ('hydrogen_ngamma_outflow', 'number_density'),
            ('radiative_transfer_boundary_flux', 'photon_flux'),
            ('radiative_transfer_source_photon_rate', 'photon_rate'),
            ('radiation_pressure_source_luminosity', 'luminosity'),
            ('radiative_transfer_boundary_flux_groups', 'photon_flux'),
            ('radiative_transfer_source_photon_rate_groups', 'photon_rate'),
            ('radiation_spectrum_total_photon_rate', 'photon_rate'),
            ('radiation_group_sigma_gamma', 'area'),
            ('radiation_group_epsilon_gamma', 'energy'),
            ('radiation_group_sigma_gamma_HeI', 'area'),
            ('radiation_group_sigma_gamma_HeII', 'area'),
            ('radiation_group_epsilon_gamma_HeI', 'energy'),
            ('radiation_group_epsilon_gamma_HeII', 'energy'),
            ('hydrogen_sigma_gamma', 'area'),
            ('hydrogen_epsilon_gamma', 'energy'),
            ('hydrogen_alpha_B', 'alpha'),
            ('hydrogen_beta', 'alpha'),
            ('gravity_coordinate', 'length'),
            ('gravity_potential', 'potential'),
            ('gravity_acceleration', 'acceleration'),
            ('gravity_strength', 'acceleration'),
            ('selfgravity_softening', 'length'),
            ('selfgravity_boundary_acceleration', 'acceleration'),
            ('gas_core_radius', 'length'),
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


def quantity_to_value(value, unit):
    """Return a plain NumPy array in the requested unit.

    When ``value`` carries units, it is converted to the supplied unit and the
    unit metadata is stripped. Plain arrays are treated as already being in the
    requested unit and are returned unchanged as ``float`` arrays.
    """
    if value is None:
        return None
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(unit), dtype=float)
    return np.asarray(value, dtype=float)

def to_unit_value(value, unit):
    """Return a plain NumPy array expressed in the supplied unit.

    Quantity-like inputs are converted to the supplied unit and stripped of
    metadata. Plain NumPy inputs are treated as numeric values already in the
    source scale and are scaled into the requested unit as raw floats.
    """
    if value is None:
        return None
    if hasattr(unit, "to_value"):
        scale = np.asarray(unit.to_value(unit.units), dtype=float)
        if hasattr(value, "to_value"):
            return np.asarray(value.to_value(unit.units), dtype=float)
        return np.asarray(value, dtype=float) * scale
    return np.asarray(value, dtype=float)

def from_unit_value(value, unit):
    """Return a plain NumPy array converted from the supplied unit scale.

    Quantity-like inputs are converted to the supplied unit and stripped of
    metadata. Plain NumPy inputs are treated as numeric values already in the
    supplied unit and are scaled back to the source scale as raw floats.
    """
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
        parameter_values = getattr(obj, "_parameter_values", None)
        if hasattr(obj, attr):
            source = getattr(obj, attr)
        elif parameter_values is not None and attr in parameter_values:
            source = parameter_values[attr]
        else:
            continue
        # An explicitly dimensionless YAML quantity denotes a code value for
        # a quantity whose runtime representation is dimensionless (for
        # example supercomoving time). Preserve its numeric value rather than
        # attempting an invalid dimensional conversion to seconds.
        if hasattr(source, "units") and source.units.is_dimensionless:
            value = np.asarray(source.value, dtype=float)
        else:
            value = quantity_to_value(source, units[unit_key])
        if hasattr(obj, attr):
            setattr(obj, attr, value)
        # ``Par`` uses this mapping as the source of truth when rebuilding its
        # nested runtime groups. Keep it synchronized with the conversion,
        # including fields intentionally exposed only through nested groups.
        if parameter_values is not None and attr in parameter_values:
            parameter_values[attr] = value


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
        "specific_angular_momentum": length_cm * velocity_cm_s,
        "angular_momentum": mass_g * length_cm * velocity_cm_s,
        "proton_mass_code": PROTON_MASS_CGS / mass_g,
        "boltzmann_code": BOLTZMANN_CONSTANT_CGS / (energy_erg / float(code.temperature_in_cgs)),
    }


def code_quantity_to_cgs(value, code, scale_key):
    """Convert a code-unit quantity or float to a cgs float array."""
    scales = code_unit_scales(code)
    if scales is None:
        return np.asarray(value, dtype=float)
    return np.asarray(value, dtype=float) * scales[scale_key]


def quantity_or_code_to_cgs(value, code, unit, scale_key):
    """Convert a unitful or code-unit value to a plain CGS array.

    Runtime parameters may still be unyt quantities at source-state
    boundaries, while values loaded from serialized runtime parameters may be
    plain code-unit numbers.  Keep this distinction in the shared units
    layer so physics modules do not each implement their own conversion.
    """
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(unit), dtype=float)
    return np.asarray(code_quantity_to_cgs(value, code, scale_key), dtype=float)


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

    @cached_property
    def unit_conversion(self):
        """Cached numeric conversion factors for the runtime solver.

        Configuration and output may use unyt quantities, but the inner
        solver loop operates on plain arrays and these factors avoid creating
        unyt unit expressions for every source or EOS evaluation.
        """
        return code_unit_scales(self)

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
        if mapping is None:
            raise ValueError("CodeUnits requires a mapping or UnitSystem")
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

        data = dict(mapping)
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
