"""Default simulation parameters and parameter container."""

import numpy as np
import unyt

from radhydropy.units import CodeUnits, _as_cgs_float

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
    'cie_cooling': False,
    'cie_ion_fraction_table': None,
    'cie_cooling_table': None,
    'cie_abundance_file': None,
    'metallicity': 1.0,
    'cooling_safety_factor': 0.1,
    'cooling_temperature_floor': 100.0 * unyt.K,
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
            self.unit_system = self.CodeUnits.unit_system
            if verbose > 0 and missing_keys:
                for key, value in missing_keys:
                    print("key %s not found in params" % key)
                    print(str(value) + " is used")
