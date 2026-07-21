"""Default simulation parameters and parameter container."""

import unyt

refparams = {
    'simname':'advection1d',
    'ICfilename':'/InitialCondition.hdf5',
    'outdir':'./',
    'outfileprefix':'Output', 
    'outdeltatime':2.0*unyt.s *0.1,
    'savedir':'./',
    'coordsys':'cartesian', #
    'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
    'gamma':1.4, # for polytropic, the polytropic index
    'timesim':2.0*unyt.s, # final simulation time
    'CFL':0.1, # CFL condition for time-step
    'boundcond':'Periodic',
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
    'hydrogen_chemistry': False,
    'hydrogen_mass_fraction': 1.0,
    'hydrogen_xHI_initial': 1.0,
    'hydrogen_xHI_inflow': 1.0,
    'hydrogen_xHI_outflow': 1.0,
    'hydrogen_source_CFL': 0.1,
    'hydrogen_update_mu': False,
    'hydrogen_thermal_coupling': True,
    'hydrogen_collisional_ionization': True,
    'hydrogen_radiation_field': False,
    'hydrogen_radiation_evolution': True,
    'hydrogen_ngamma_initial': 0.0 / unyt.cm**3,
    'hydrogen_ngamma_inflow': 0.0 / unyt.cm**3,
    'hydrogen_ngamma_outflow': 0.0 / unyt.cm**3,
    'hydrogen_sigma_gamma': 1.62e-18 * unyt.cm**2,
    'hydrogen_epsilon_gamma': 0.0 * unyt.erg,
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
            for key, value in refparams.items():
                if key in params:
                    setattr(self, key, params[key])
                else:
                    print("key %s not find in params"%key)
                    print(str(value) +"is used")
                    setattr(self, key, value)
