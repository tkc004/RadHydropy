"""Default simulation parameters and parameter container."""

import numpy as np
import unyt
from radhydropy.radiation_spectrum import load_radiation_spectrum, resolve_spectrum_filename
from radhydropy.thermo_networks.pie import MetalPIETable

from radhydropy.units import CodeUnits, _as_cgs_float, code_quantity_to_cgs
from radhydropy.cosmology import EinsteinDeSitter

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
    'dark_matter_crossing_safety_factor': 0.1,
    'gravity': None,
    'gravity_potential': None,
    'gravity_coordinate': None,
    'gravity_acceleration': None,
    'selfgravity_softening': 0.0 * unyt.cm,
    'selfgravity_boundary_acceleration': 0.0 * unyt.cm / unyt.s**2,
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
    'relaxation_damping_time': None,
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
    'helium_mass_fraction': 0.0,
    'hydrogen_helium_implicit_local_update': False,
    'hydrogen_helium_coupled_implicit': True,
    'hydrogen_helium_xHeI_initial': 1.0,
    'hydrogen_helium_xHeII_initial': 0.0,
    'hydrogen_helium_xHeIII_initial': 0.0,
    'hydrogen_xHI_initial': 1.0,
    'hydrogen_xHI_inflow': 1.0,
    'hydrogen_xHI_outflow': 1.0,
    'hydrogen_source_CFL': 0.1,
    'hydrogen_source_dtmin': 0.0 * unyt.s,
    'hydrogen_update_mu': False,
    'hydrogen_thermal_coupling': True,
    'compton_cmb_enabled': False,
    'compton_cmb_redshift': 0.0,
    'cmb_temperature_0': 2.7255 * unyt.K,
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
    'radiative_transfer_boundary_flux_groups': None,
    'radiative_transfer_source_photon_rate': 0.0 / unyt.s,
    'radiative_transfer_source_photon_rate_groups': None,
    'radiation_group_edges_eV': None,
    'radiation_group_sigma_gamma': None,
    'radiation_group_epsilon_gamma': None,
    'radiation_group_sigma_gamma_HeI': None,
    'radiation_group_sigma_gamma_HeII': None,
    'radiation_group_epsilon_gamma_HeI': None,
    'radiation_group_epsilon_gamma_HeII': None,
    'star_emission_rates': None,
    'stellar_spectrum_type': 1,
    'stellar_spectrum_blackbody_temperature_K': 1.0e5,
    'ionizing_photon_energy_erg': None,
    'radiation_spectrum_filename': None,
    'radiation_spectrum_total_photon_rate': None,
    'metal_pie_enabled': False,
    'metal_pie_table_filename': None,
    'metal_pie_table': None,
    'number_of_radiation_groups': None,
    'radiative_transfer_direction': 1,
    'cosmological_expansion': False,
    'supercomoving_coordinates': False,
    'coordinate_frame': 'physical',
    'time_coordinate': 'cosmic',
    'velocity_representation': 'physical',
    'density_representation': 'physical',
    'pressure_representation': 'physical',
    'temperature_representation': 'physical',
    'cosmology_type': None,
    'cosmology_t_ref': 1.0,
    'cosmology_a_ref': 1.0,
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
            if self.cosmological_expansion:
                if self.cosmology_type not in (None, 'einstein_de_sitter', 'EinsteinDeSitter'):
                    raise ValueError(
                        "unsupported cosmology_type: %s" % self.cosmology_type
                    )
                self.cosmology = EinsteinDeSitter.from_code_units(
                    self.CodeUnits,
                    t_ref=self.cosmology_t_ref,
                    a_ref=self.cosmology_a_ref,
                )
                if self.supercomoving_coordinates:
                    self.coordinate_frame = 'comoving'
                    self.time_coordinate = 'supercomoving'
                    self.velocity_representation = 'supercomoving_peculiar'
                    self.density_representation = 'comoving'
                    self.pressure_representation = 'supercomoving'
                    self.temperature_representation = 'supercomoving'
            if params.get('radiation_spectrum_filename') is not None:
                self.load_radiation_spectrum(params.get('outdir'))
            if params.get('metal_pie_enabled', False) and params.get('metal_pie_table_filename'):
                self.load_metal_pie_table(params.get('outdir'))
            if verbose > 0 and missing_keys:
                for key, value in missing_keys:
                    print("key %s not found in params" % key)
                    print(str(value) + " is used")

    def load_radiation_spectrum(self, base_directory=None):
            """Load the configured HDF5 spectrum into runtime parameters."""
            spectrum_filename = getattr(self, 'radiation_spectrum_filename', None)
            if spectrum_filename is None:
                return
            spectrum = load_radiation_spectrum(
                resolve_spectrum_filename(spectrum_filename, base_directory)
            )
            for key, value in spectrum.items():
                setattr(self, key, value)
                self.runparams[key] = value
            if self.radiation_group_sigma_gamma is not None:
                self.radiation_group_sigma_gamma = self.radiation_group_sigma_gamma * unyt.cm**2
            if self.radiation_group_epsilon_gamma is not None:
                self.radiation_group_epsilon_gamma = self.radiation_group_epsilon_gamma * unyt.erg
            for species in ('HeI', 'HeII'):
                sigma_name = f'radiation_group_sigma_gamma_{species}'
                epsilon_name = f'radiation_group_epsilon_gamma_{species}'
                if getattr(self, sigma_name, None) is not None:
                    setattr(self, sigma_name, getattr(self, sigma_name) * unyt.cm**2)
                if getattr(self, epsilon_name, None) is not None:
                    setattr(self, epsilon_name, getattr(self, epsilon_name) * unyt.erg)
            power_unit = self.CodeUnits.energy_unit / self.CodeUnits.time_unit
            rates = np.asarray(self.star_emission_rates, dtype=float) * power_unit
            energies = np.asarray(self.ionizing_photon_energy_erg, dtype=float) * unyt.erg
            total_rate = getattr(self, 'radiation_spectrum_total_photon_rate', None)
            if total_rate is not None:
                if hasattr(total_rate, 'to_value'):
                    target_rate_s = float(total_rate.to_value(1.0 / unyt.s))
                else:
                    target_rate_s = code_quantity_to_cgs(
                        total_rate,
                        self.CodeUnits,
                        'photon_rate_per_s',
                    )
                current_rate_s = float(
                    np.sum((rates[1:] / energies).to_value(1.0 / unyt.s))
                )
                if current_rate_s <= 0.0:
                    raise ValueError('radiation spectrum has no ionizing injection rate')
                self.star_emission_rates = np.array(self.star_emission_rates, dtype=float)
                self.star_emission_rates[1:] *= target_rate_s / current_rate_s
                self.runparams['star_emission_rates'] = self.star_emission_rates
                rates = self.star_emission_rates * power_unit
            self.radiative_transfer_source_photon_rate_groups = (rates[1:] / energies).to(1.0 / unyt.s)
            self.radiative_transfer_boundary_flux_groups = np.zeros(
                self.number_of_radiation_groups
            ) / (unyt.cm**2 * unyt.s)

    def load_metal_pie_table(self, base_directory=None):
        filename = resolve_spectrum_filename(
            self.metal_pie_table_filename, base_directory
        )
        self.metal_pie_table = MetalPIETable(filename)
