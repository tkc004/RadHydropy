"""Default simulation parameters and parameter container."""

from dataclasses import dataclass

import numpy as np
import unyt

refparams = {
    'simname':'advection1d',
    'ICfilename':'/InitialCondition.hdf5',
    'outdir':'./',
    'outfileprefix':'Output', 
    'outdeltatime':2.0*unyt.s *0.1,
    'outputtimefilename': None,
    'savedir':'./',
    'coordsys':'cartesian', #
    'selfgravity': False,
    'externalgravity': False,
    'gravity': None,
    'gravity_potential': None,
    'gravity_coordinate': None,
    'gravity_acceleration': None,
    'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
    'gamma':1.4, # for polytropic, the polytropic index
    'timesim':2.0*unyt.s, # final simulation time
    'CFL':0.1, # CFL condition for time-step
    'boundcond':'Periodic',
    'CodeUnits': None,
    'area': 1.0 * unyt.cm**2,
    'vel_inflow':1.0*unyt.cm/unyt.s,
    'rho_inflow':1.0*unyt.g/unyt.cm**3,
    'temp_inflow':0.0*unyt.K,
    'mu_inflow':1.0,
    'vel_outflow':1.0*unyt.cm/unyt.s,
    'rho_outflow':1.0*unyt.g/unyt.cm**3,
    'temp_outflow':0.0*unyt.K,
    'mu_outflow':1.0,    
    'verbose':0, # speak out details?
    'order': 0,  
    'noghost':2,
    'dtmin': 2.0e-8*unyt.s,
    'dtmax': 2.0e-1*unyt.s,   
    'thermochemistry_network': 'hydrogen',
    'chemistry_key': 'H',
    'hydrogen_chemistry': False,
    'hydrogen_mass_fraction': 1.0,
    'hydrogen_xHI_initial': 1.0,
    'hydrogen_xHI_inflow': 1.0,
    'hydrogen_xHI_outflow': 1.0,
    'hydrogen_source_CFL': 0.1,
    'hydrogen_source_dtmin': 0.0 * unyt.s,
    'hydrogen_update_mu': False,
    'hydrogen_thermal_coupling': True,
    'hydrogen_recombination': True,
    'hydrogen_collisional_ionization': True,
    'hydrogen_alpha_B': None,
    'hydrogen_beta': None,
    'hydrogen_radiation_field': False,
    'hydrogen_radiation_evolution': True,
    'hydrogen_ngamma_initial': 0.0 / unyt.cm**3,
    'hydrogen_ngamma_inflow': 0.0 / unyt.cm**3,
    'hydrogen_ngamma_outflow': 0.0 / unyt.cm**3,
    'hydrogen_sigma_gamma': 1.62e-18 * unyt.cm**2,
    'hydrogen_epsilon_gamma': 0.0 * unyt.erg,
    'radiative_transfer': False,
    'radiative_transfer_method': 'long_characteristics',
    'radiative_transfer_boundary_flux': 0.0 / (unyt.cm**2 * unyt.s),
    'radiative_transfer_source_photon_rate': 0.0 / unyt.s,
    'radiative_transfer_direction': 1,
}


def _as_cgs_float(value, unit):
    if hasattr(value, 'to_value'):
        return float(value.to_value(unit))
    return float(value)


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
        if hasattr(quantity, 'to_value'):
            return np.asarray(quantity.to_value(unit), dtype=float)
        return np.asarray(quantity, dtype=float)

    def from_value(self, values, unit):
        return np.asarray(values, dtype=float) * unit

    @classmethod
    def from_mapping(cls, mapping=None, name='code'):
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
                name=getattr(mapping, 'name', name),
                mass_in_cgs=_as_cgs_float(mass_unit, unyt.g),
                length_in_cgs=_as_cgs_float(length_unit, unyt.cm),
                velocity_in_cgs=_as_cgs_float(length_unit / time_unit, unyt.cm / unyt.s),
                current_in_cgs=_as_cgs_float(current_unit, unyt.A),
                temperature_in_cgs=_as_cgs_float(temperature_unit, unyt.K),
                unit_system=mapping,
            )

        data = dict(mapping or {})
        internal = data.get('InternalUnitSystem', data)
        if not isinstance(internal, dict):
            raise TypeError(
                'CodeUnits must be built from a mapping, a UnitSystem, or None'
            )

        mass_in_cgs = _as_cgs_float(
            internal.get('UnitMass_in_cgs', internal.get('mass_in_cgs', 1.0)),
            unyt.g,
        )
        length_in_cgs = _as_cgs_float(
            internal.get('UnitLength_in_cgs', internal.get('length_in_cgs', 1.0)),
            unyt.cm,
        )
        velocity_in_cgs = _as_cgs_float(
            internal.get('UnitVelocity_in_cgs', internal.get('velocity_in_cgs', 1.0)),
            unyt.cm / unyt.s,
        )
        current_in_cgs = _as_cgs_float(
            internal.get('UnitCurrent_in_cgs', internal.get('current_in_cgs', 1.0)),
            unyt.A,
        )
        temperature_in_cgs = _as_cgs_float(
            internal.get('UnitTemp_in_cgs', internal.get('temperature_in_cgs', 1.0)),
            unyt.K,
        )
        if mass_in_cgs <= 0.0 or length_in_cgs <= 0.0 or velocity_in_cgs <= 0.0:
            raise ValueError('CodeUnits mass, length, and velocity scales must be positive')
        time_in_cgs = length_in_cgs / velocity_in_cgs
        unit_system = unyt.UnitSystem(
            internal.get('name', data.get('name', name)),
            length_in_cgs * unyt.cm,
            mass_in_cgs * unyt.g,
            time_in_cgs * unyt.s,
            temperature_in_cgs * unyt.K,
            current_mks_unit=current_in_cgs * unyt.A,
        )
        return cls(
            name=internal.get('name', data.get('name', name)),
            mass_in_cgs=mass_in_cgs,
            length_in_cgs=length_in_cgs,
            velocity_in_cgs=velocity_in_cgs,
            current_in_cgs=current_in_cgs,
            temperature_in_cgs=temperature_in_cgs,
            unit_system=unit_system,
        )

    def to_dict(self):
        return {
            'name': self.name,
            'InternalUnitSystem': {
                'UnitMass_in_cgs': self.mass_in_cgs,
                'UnitLength_in_cgs': self.length_in_cgs,
                'UnitVelocity_in_cgs': self.velocity_in_cgs,
                'UnitCurrent_in_cgs': self.current_in_cgs,
                'UnitTemp_in_cgs': self.temperature_in_cgs,
            },
        }


class Par():
    """Apply user parameters on top of :data:`refparams` defaults.

    Parameters
    ----------
    params : dict
        User supplied run parameters. Missing keys are filled from
        :data:`refparams`.
    """

    def __init__(self,params) -> None:
            self.runparams = dict(params)
            verbose = int(params.get('verbose', refparams.get('verbose', 0)))
            code_units_value = params.get('CodeUnits', None)
            if code_units_value is None:
                raise ValueError(
                    'run parameters must define CodeUnits with an internal unit system'
                )
            missing_keys = []
            for key, value in refparams.items():
                if key in params:
                    setattr(self, key, params[key])
                else:
                    if value is not None:
                        missing_keys.append((key, value))
                    setattr(self, key, value)
            self.CodeUnits = CodeUnits.from_mapping(code_units_value)
            self.code_units = self.CodeUnits
            self.unit_system = self.CodeUnits.unit_system
            if verbose > 0 and missing_keys:
                for key, value in missing_keys:
                    print("key %s not found in params" % key)
                    print(str(value) + " is used")
