"""Default simulation parameters and parameter container."""

import warnings
from dataclasses import dataclass

import numpy as np
import unyt
from radhydropy.radiation_spectrum import load_radiation_spectrum, resolve_spectrum_filename
from radhydropy.thermo_networks.pie import MetalPIETable

from radhydropy.units import CodeUnits, _as_cgs_float, code_quantity_to_cgs
from radhydropy.cosmology import EinsteinDeSitter, LambdaCDM

refparams = {
    'simname':'advection1d',
    'ICfilename':'/InitialCondition.hdf5',
    'outdir':'./',
    'outfileprefix':'Output', 
    'outdeltatime':2.0*unyt.s *0.1,
    'outputtimefilename': None,
    'savedir':'./',
    'figure_prefix': None,
    'final_cosmic_time': None,
    'gas_profile_cadence': None,
    'linear_correlation_table_filename': None,
    'minimum_temperature': None,
    'plot_exclude_outer_cells': 0,
    'smooth_dm_force_for_gas': False,
    'temperature_plot_ymin': None,
    'dm_density_bins': None,
    'coordsys':'cartesian', #
    'selfgravity': False,
    'externalgravity': False,
    'dark_matter_crossing_safety_factor': 0.1,
    # Optional approximate crossing batching.  Zero retains the exact
    # event-driven shell integrator; positive values batch crossings over this
    # fraction of each requested dark-matter interval.
    'dark_matter_crossing_batch_fraction': 0.0,
    'dark_matter_global_timestep_limit': True,
    'gravity': None,
    'gravity_potential': None,
    'gravity_coordinate': None,
    'gravity_acceleration': None,
    'selfgravity_softening': 0.0 * unyt.cm,
    'selfgravity_boundary_acceleration': 0.0 * unyt.cm / unyt.s**2,
    'cosmological_gravity': False,
    'EOStype':'polytropic', #type of equation of state (EOS): polytropic or isothermal
    'gamma':1.4, # for polytropic, the polytropic index
    'temperature':2.7*unyt.K, # default gas/background temperature
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
    # Abort with a cell/neighborhood diagnostic if a gas cell crosses this
    # physical temperature.  Set to None to disable the guard.
    'temperature_jump_error_threshold': 1.0e8,
    'order': 0,  
    'riemann_solver': 'Rusanov',
    # Minmod is robust near strong discontinuities; MC retains more
    # resolution in smooth rarefaction/contact regions while remaining TVD.
    'flux_limiter': 'minmod',
    'dual_energy': False,
    # First-stage optional passive gas angular-momentum storage.
    'gas_angular_momentum': False,
    'gas_rotational_energy': False,
    # Donor upwind plus face-local MUSCL/FCT limiting is the robust default.
    'angular_momentum_flux_scheme': 'fct',
    # Local hydro fallback threshold for nearly rotation-supported cells.
    'angular_momentum_energy_margin_fraction': 1.0e-4,
    'gravity_potential_energy': False,
    'gas_specific_angular_momentum': 0.0,
    'specific_angular_momentum_inflow': 0.0,
    'specific_angular_momentum_outflow': 0.0,
    # Bryan et al. (1995) dual-energy switching thresholds.  eta1 selects
    # the pressure estimate; eta2 controls synchronization to total energy.
    'dual_energy_eta1': 1.0e-3,
    'dual_energy_eta2': 1.0e-1,
    # If both estimates are admissible, reject a dual estimate that has
    # fallen far below conservative E-K.  This protects entropy in cells
    # where the dual flux update loses thermal energy at an under-resolved
    # shock.
    'dual_energy_consistency_factor': 1.0e-1,
    'dual_energy_entropy_limiter': True,
    # Pressure source in dual-energy mode: 'switch' selects between E-K and
    # InternalEnergy; 'internal' always selects the evolved InternalEnergy;
    # 'conservative' always selects admissible E-K.
    'dual_energy_pressure_selection': 'switch',
    # Code-unit pressure used only when both energy estimates are invalid.
    'dual_energy_pressure_floor': 1.0e-20,
    # Backward-compatible alias for the old single pressure-selection switch.
    'dual_energy_switch': 1.0e-3,
    'nogrid': None,
    'noghost':2,
    'dtmin': 2.0e-8*unyt.s,
    'dtmax': 2.0e-1*unyt.s,   
    # Numerical density threshold used only for vacuum-safe CFL and face
    # reconstruction.  Cell-centred conserved states are not floored.
    'cfl_density_floor': 0.0,
    'hydro_temperature_floor': None,
    # Conservative invariant-domain limiter for finite-volume hydro updates.
    'positivity_preserving': True,
    'positivity_density_floor': 0.0,
    'positivity_energy_floor': 0.0,
    'relaxation_damping_time': None,
    'thermochemistry_network': 'hydrogen',
    'cie_cooling': False,
    'cie_ion_fraction_table': None,
    'cie_cooling_table': None,
    'cie_abundance_file': None,
    'metallicity': 1.0,
    'cooling_safety_factor': 0.1,
    'cooling_temperature_floor': 100.0 * unyt.K,
    'pie_uvbg_implicit_tolerance': 1.0e-3,
    'pie_uvbg_implicit_max_retries': 8,
    'pie_uvbg_implicit_max_iterations': 64,
    'pie_uvbg_implicit_step_doubling': True,
    'chemistry_key': 'H',
    'hydrogen_chemistry': False,
    'hydrogen_mass_fraction': 1.0,
    'helium_mass_fraction': 0.0,
    'hydrogen_helium_coupled_implicit': True,
    'hydrogen_helium_xHeI_initial': 1.0,
    'hydrogen_helium_xHeII_initial': 0.0,
    'hydrogen_helium_xHeIII_initial': 0.0,
    'hydrogen_xHI_initial': 1.0,
    'hydrogen_xHI_inflow': 1.0,
    'hydrogen_xHI_outflow': 1.0,
    'hydrogen_source_CFL': 0.1,
    'hydrogen_source_dtmin': 0.0 * unyt.s,
    'hydrogen_source_solver': 'hybrid',
    'hydrogen_source_skip_floor_cells': False,
    'hydrogen_source_density_floor': None,
    'hydrogen_source_floor_temperature_tolerance': 1.0e-2,
    'hydrogen_hybrid_change_tolerance': 0.1,
    'hydrogen_implicit_tolerance': 1.0e-6,
    'hydrogen_implicit_absolute_temperature_tolerance': 0.0 * unyt.K,
    'hydrogen_implicit_absolute_xhi_tolerance': 0.0,
    'hydrogen_implicit_convergence_tolerance': 1.0e-3,
    'hydrogen_implicit_max_iterations': 32,
    'hydrogen_implicit_debug': False,
    'hydrogen_implicit_fallback': 'explicit',
    'hydrogen_implicit_max_refinements': 4,
    'hydrogen_split_implicit_max_subcycles': 100000,
    # Optional pressure-supported unresolved central core.  The default keeps
    # the ordinary cell-centred hydro evolution unchanged.
    'gas_core_model': 'none',
    'gas_core_radius': None,
    'hydrogen_update_mu': False,
    'hydrogen_thermal_coupling': True,
    'energy_diagnostics': False,
    'compton_cmb_enabled': False,
    'compton_cmb_redshift': 0.0,
    'cmb_temperature_0': 2.7255 * unyt.K,
    'hydrogen_recombination': True,
    'hydrogen_collisional_ionization': True,
    'hydrogen_atomic_cooling': True,
    'hydrogen_alpha_B': None,
    'hydrogen_beta': None,
    'hydrogen_radiation_field': False,
    'hydrogen_radiation_evolution': True,
    'hydrogen_ngamma_initial': 0.0 / unyt.cm**3,
    'hydrogen_ngamma_inflow': 0.0 / unyt.cm**3,
    'hydrogen_ngamma_outflow': 0.0 / unyt.cm**3,
    'hydrogen_sigma_gamma': 1.62e-18 * unyt.cm**2,
    'hydrogen_epsilon_gamma': 0.0 * unyt.erg,
    'hydrogen_photon_energy': 13.6 * unyt.eV,
    'radiation_pressure': False,
    'radiation_pressure_efficiency': 1.0,
    'radiative_transfer': False,
    'radiative_transfer_method': 'long_characteristics',
    'radiative_transfer_temporal_scheme': 'instantaneous',
    'radiative_transfer_c2ray_max_iterations': 32,
    'radiative_transfer_c2ray_tolerance': 1.0e-6,
    'radiative_transfer_c2ray_relaxation': 1.0,
    'radiative_transfer_c2ray_nonconvergence': 'warn',
    'radiative_transfer_c2ray_ode_max_iterations': 24,
    'radiative_transfer_c2ray_ode_tolerance': 1.0e-8,
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
    'metal_pie_photoheating_max_density_cm3': 50.0,
    'metal_pie_redshift': 0.0,
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
    'cosmological_background_boundary_reconstruction': False,
    'cosmology_type': None,
    'cosmology_t_ref': 1.0,
    'cosmology_a_ref': 1.0,
    'cosmology_omega_m': 0.3,
    'cosmology_omega_lambda': 0.7,
    'cosmology_hubble_ref': None,
}


@dataclass
class CosmologyParameters:
    """Structured cosmology settings with legacy model delegation."""

    type: object = None
    t_ref: float = 1.0
    a_ref: float = 1.0
    omega_m: float = 0.3
    omega_lambda: float = 0.7
    hubble_ref: object = None
    model: object = None

    def __getattr__(self, name):
        model = self.__dict__.get('model')
        if model is not None:
            return getattr(model, name)
        raise AttributeError(name)


@dataclass
class UnitsParameters:
    """Structured view of the internal code-unit system."""

    CodeUnits: object = None
    unit_system: object = None


@dataclass
class HydrodynamicsParameters:
    """Structured view of the primary gas-dynamics settings."""

    eos_type: str = 'polytropic'
    gamma: float = 1.4
    temperature: object = None
    CFL: float = 0.1
    order: int = 0
    riemann_solver: str = 'Rusanov'
    flux_limiter: str = 'minmod'
    dual_energy: bool = False
    dual_energy_pressure_selection: str = 'switch'
    dual_energy_entropy_limiter: bool = True
    positivity_preserving: bool = True
    positivity_density_floor: float = 0.0
    positivity_energy_floor: float = 0.0
    gas_angular_momentum: bool = False
    gas_rotational_energy: bool = False
    angular_momentum_flux_scheme: str = 'fct'


@dataclass
class BoundaryParameters:
    """Structured view of boundary-condition and reservoir settings."""

    condition: str = 'Periodic'
    inflow_velocity: object = None
    inflow_density: object = None
    inflow_temperature: object = None
    inflow_mu: float = 1.0
    outflow_velocity: object = None
    outflow_density: object = None
    outflow_temperature: object = None
    outflow_mu: float = 1.0
    cosmological_background_reconstruction: bool = False


@dataclass
class TimestepParameters:
    """Structured view of hydro and source timestep controls."""

    dtmin: object = None
    dtmax: object = None
    cfl_density_floor: float = 0.0
    hydro_temperature_floor: object = None
    cooling_safety_factor: float = 0.1
    hydrogen_source_CFL: float = 0.1
    hydrogen_source_dtmin: object = None
    relaxation_damping_time: object = None


@dataclass
class ThermochemistryParameters:
    """Structured view of cooling, chemistry, and thermal source settings."""

    network: str = 'hydrogen'
    cie_cooling: bool = False
    cie_ion_fraction_table: object = None
    cie_cooling_table: object = None
    cie_abundance_file: object = None
    metallicity: float = 1.0
    cooling_safety_factor: float = 0.1
    cooling_temperature_floor: object = None
    hydrogen_chemistry: bool = False
    hydrogen_recombination: bool = True
    hydrogen_collisional_ionization: bool = True
    hydrogen_atomic_cooling: bool = True
    hydrogen_thermal_coupling: bool = True
    hydrogen_update_mu: bool = False
    compton_cmb_enabled: bool = False
    metal_pie_enabled: bool = False
    pie_uvbg_implicit_tolerance: float = 1.0e-3
    pie_uvbg_implicit_max_retries: int = 8
    pie_uvbg_implicit_max_iterations: int = 64
    pie_uvbg_implicit_step_doubling: bool = True
    hydrogen_source_solver: str = 'hybrid'


@dataclass
class GravityParameters:
    """Structured gravity settings with delegation to a live gravity model."""

    selfgravity: bool = False
    externalgravity: bool = False
    potential: object = None
    coordinate: object = None
    acceleration: object = None
    selfgravity_softening: object = None
    selfgravity_boundary_acceleration: object = None
    cosmological: bool = False
    potential_energy: bool = False
    model: object = None

    def __getattr__(self, name):
        model = self.__dict__.get('model')
        if model is not None:
            return getattr(model, name)
        raise AttributeError(name)


@dataclass
class OutputParameters:
    """Structured view of snapshot destinations and scheduling settings."""

    directory: str = './'
    savedir: str = './'
    filename_prefix: str = 'Output'
    cadence: object = None
    time_list_filename: object = None


@dataclass
class SimulationParameters:
    """Structured view of run identity and coordinate representations."""

    name: str = 'advection1d'
    initial_condition_filename: object = None
    coordinate_system: str = 'cartesian'
    final_time: object = None
    current_time: object = None
    box_size: object = None
    cosmological_expansion: bool = False
    supercomoving_coordinates: bool = False
    coordinate_frame: str = 'physical'
    time_coordinate: str = 'cosmic'
    velocity_representation: str = 'physical'
    density_representation: str = 'physical'
    pressure_representation: str = 'physical'
    temperature_representation: str = 'physical'


@dataclass
class DiagnosticsParameters:
    """Structured view of runtime safety and energy diagnostics."""

    verbose: int = 0
    energy_diagnostics: bool = False
    temperature_jump_error_threshold: object = None
    temperature_plot_ymin: object = None
    plot_exclude_outer_cells: int = 0


@dataclass
class MeshParameters:
    """Structured view of mesh geometry settings."""

    ghost_cells: int = 2
    area: object = None
    grid_cells: object = None


@dataclass
class ChemistryParameters:
    """Structured view of chemical species and ionization defaults."""

    key: str = 'H'
    hydrogen_mass_fraction: float = 1.0
    helium_mass_fraction: float = 0.0
    hydrogen_xHI_initial: float = 1.0
    hydrogen_xHI_inflow: float = 1.0
    hydrogen_xHI_outflow: float = 1.0
    helium_xHeI_initial: float = 1.0
    helium_xHeII_initial: float = 0.0
    helium_xHeIII_initial: float = 0.0
    update_mu: bool = False
    helium_coupled_implicit: bool = True
    source_skip_floor_cells: bool = False
    source_density_floor: object = None
    source_floor_temperature_tolerance: float = 1.0e-2
    hybrid_change_tolerance: float = 0.1
    implicit_tolerance: float = 1.0e-6
    implicit_convergence_tolerance: float = 1.0e-3
    implicit_max_iterations: int = 32
    implicit_fallback: str = 'explicit'
    implicit_max_refinements: int = 4
    split_implicit_max_subcycles: int = 100000
    implicit_absolute_temperature_tolerance: object = None
    implicit_absolute_xhi_tolerance: float = 0.0
    implicit_debug: bool = False
    alpha_B: object = None
    beta: object = None


@dataclass
class AngularMomentumParameters:
    """Structured view of gas angular-momentum transport settings."""

    enabled: bool = False
    rotational_energy: bool = False
    flux_scheme: str = 'fct'
    energy_margin_fraction: float = 1.0e-4
    specific_angular_momentum: object = 0.0
    inflow: object = 0.0
    outflow: object = 0.0


@dataclass
class DarkMatterParameters:
    """Structured view of live dark-matter shell integration settings."""

    crossing_safety_factor: float = 0.1
    crossing_batch_fraction: float = 0.0
    global_timestep_limit: bool = True
    density_bins: object = None


@dataclass
class DualEnergyParameters:
    """Structured view of dual-energy pressure and synchronization controls."""

    enabled: bool = False
    eta1: float = 1.0e-3
    eta2: float = 1.0e-1
    consistency_factor: float = 1.0e-1
    entropy_limiter: bool = True
    pressure_selection: str = 'switch'
    pressure_floor: float = 1.0e-20
    switch: float = 1.0e-3


@dataclass
class PositivityParameters:
    """Structured view of invariant-domain limiting controls."""

    enabled: bool = True
    density_floor: float = 0.0
    energy_floor: float = 0.0


@dataclass
class RadiationParameters:
    """Structured view of the radiation and metal-PIE settings."""

    spectrum_filename: object = None
    spectrum_total_photon_rate: object = None
    radiative_transfer: bool = False
    radiation_pressure: bool = False
    metal_pie_enabled: bool = False
    metal_pie_table_filename: object = None
    metal_pie_table: object = None
    number_of_radiation_groups: object = None
    group_edges_eV: object = None
    group_sigma_gamma: object = None
    group_epsilon_gamma: object = None
    group_sigma_gamma_HeI: object = None
    group_sigma_gamma_HeII: object = None
    group_epsilon_gamma_HeI: object = None
    group_epsilon_gamma_HeII: object = None
    radiative_transfer_method: str = 'long_characteristics'
    radiative_transfer_temporal_scheme: str = 'instantaneous'
    radiative_transfer_direction: int = 1
    boundary_flux: object = None
    source_photon_rate: object = None
    boundary_flux_groups: object = None
    source_photon_rate_groups: object = None
    c2ray_max_iterations: int = 32
    c2ray_tolerance: float = 1.0e-6
    c2ray_relaxation: float = 1.0
    c2ray_nonconvergence: str = 'warn'
    c2ray_ode_max_iterations: int = 24
    c2ray_ode_tolerance: float = 1.0e-8
    radiation_pressure_efficiency: float = 1.0
    compton_cmb_enabled: bool = False
    compton_cmb_redshift: float = 0.0
    cmb_temperature_0: object = None
    hydrogen_radiation_field: bool = False
    hydrogen_radiation_evolution: bool = True
    hydrogen_ngamma_initial: object = None
    hydrogen_ngamma_inflow: object = None
    hydrogen_ngamma_outflow: object = None
    hydrogen_sigma_gamma: object = None
    hydrogen_epsilon_gamma: object = None
    hydrogen_photon_energy: object = None
    stellar_spectrum_type: int = 1
    stellar_spectrum_blackbody_temperature_K: float = 1.0e5
    star_emission_rates: object = None
    ionizing_photon_energy_erg: object = None
    metal_pie_photoheating_max_density_cm3: float = 50.0
    metal_pie_redshift: float = 0.0


class ParameterDefaultWarning(UserWarning):
    """Warning emitted when :class:`Par` supplies a missing default."""


class Par:
    """Apply user parameters on top of :data:`refparams` defaults.

    Parameters
    ----------
    params : dict
        User supplied run parameters. Missing keys are filled from
        :data:`refparams`.
    """

    def __init__(self, params) -> None:
        params = self._validate_mapping(params)
        params = self._flatten_nested_parameters(params)
        self._validate_keys(params)
        self.runparams = dict(params)
        self._parameter_values = {}
        missing_keys = self._apply_defaults(params)
        self._initialize_parameter_groups()
        self._initialize_units(params)
        self._configure_cosmology()
        self._load_optional_physics(params)
        self._sync_simulation_parameters()
        self._warn_defaulted_parameters(missing_keys)

    @staticmethod
    def _validate_mapping(params):
        if not hasattr(params, 'items'):
            raise TypeError('run parameters must be supplied as a mapping')
        return params

    @staticmethod
    def _flatten_nested_parameters(params):
        """Translate the nested YAML shape into internal input names."""
        if not any(isinstance(params.get(group), dict) for group in (
            'simulation', 'mesh', 'hydrodynamics', 'boundary', 'timestep',
            'units', 'radiation', 'chemistry', 'thermochemistry',
        )):
            return params
        flattened = dict(params)
        groups = {
            'simulation': {
                'name': 'simname',
                'initial_condition_filename': 'ICfilename',
                'coordinate_system': 'coordsys',
                'final_time': 'timesim',
                'box_size': 'boxsize',
                'current_time': 'time',
            },
            'mesh': {'grid_cells': 'nogrid', 'ghost_cells': 'noghost', 'area': 'area'},
            'hydrodynamics': {
                'eos_type': 'EOStype', 'gamma': 'gamma', 'temperature': 'temperature',
                'CFL': 'CFL', 'order': 'order', 'riemann_solver': 'riemann_solver',
                'positivity_preserving': 'positivity_preserving',
                'dual_energy': 'dual_energy',
                'cfl_density_floor': 'cfl_density_floor',
                'temperature_jump_error_threshold': 'temperature_jump_error_threshold',
            },
            'boundary': {
                'condition': 'boundcond', 'inflow_velocity': 'vel_inflow',
                'inflow_density': 'rho_inflow', 'inflow_temperature': 'temp_inflow',
                'inflow_mu': 'mu_inflow', 'outflow_velocity': 'vel_outflow',
                'outflow_density': 'rho_outflow', 'outflow_temperature': 'temp_outflow',
                'outflow_mu': 'mu_outflow',
            },
            'timestep': {
                'dtmin': 'dtmin', 'dtmax': 'dtmax',
                'hydrogen_source_CFL': 'hydrogen_source_CFL',
                'hydrogen_source_dtmin': 'hydrogen_source_dtmin',
            },
            'units': {'CodeUnits': 'CodeUnits'},
            'output': {
                'directory': 'outdir', 'savedir': 'savedir',
                'filename_prefix': 'outfileprefix', 'cadence': 'outdeltatime',
                'time_list_filename': 'outputtimefilename',
            },
            'diagnostics': {'verbose': 'verbose'},
            'radiation': {
                'direction': 'radiative_transfer_direction',
                'boundary_flux': 'radiative_transfer_boundary_flux',
                'source_photon_rate': 'radiative_transfer_source_photon_rate',
                'radiative_transfer': 'radiative_transfer',
                'method': 'radiative_transfer_method',
                'temporal_scheme': 'radiative_transfer_temporal_scheme',
                'c2ray_max_iterations': 'radiative_transfer_c2ray_max_iterations',
                'c2ray_tolerance': 'radiative_transfer_c2ray_tolerance',
                'c2ray_relaxation': 'radiative_transfer_c2ray_relaxation',
                'c2ray_nonconvergence': 'radiative_transfer_c2ray_nonconvergence',
                'hydrogen_radiation_field': 'hydrogen_radiation_field',
                'hydrogen_radiation_evolution': 'hydrogen_radiation_evolution',
                'hydrogen_ngamma_initial': 'hydrogen_ngamma_initial',
                'hydrogen_ngamma_inflow': 'hydrogen_ngamma_inflow',
                'hydrogen_ngamma_outflow': 'hydrogen_ngamma_outflow',
                'hydrogen_sigma_gamma': 'hydrogen_sigma_gamma',
                'hydrogen_epsilon_gamma': 'hydrogen_epsilon_gamma',
            },
            'chemistry': {
                'hydrogen_mass_fraction': 'hydrogen_mass_fraction',
                'metallicity': 'metallicity',
                'hydrogen_xHI_initial': 'hydrogen_xHI_initial',
                'hydrogen_xHI_inflow': 'hydrogen_xHI_inflow',
                'hydrogen_xHI_outflow': 'hydrogen_xHI_outflow',
                'hydrogen_update_mu': 'hydrogen_update_mu',
                'hydrogen_thermal_coupling': 'hydrogen_thermal_coupling',
                'hydrogen_recombination': 'hydrogen_recombination',
                'hydrogen_collisional_ionization': 'hydrogen_collisional_ionization',
                'hydrogen_alpha_B': 'hydrogen_alpha_B',
                'hydrogen_beta': 'hydrogen_beta',
            },
            'thermochemistry': {
                'network': 'thermochemistry_network',
                'hydrogen_chemistry': 'hydrogen_chemistry',
                'hydrogen_thermal_coupling': 'hydrogen_thermal_coupling',
                'hydrogen_recombination': 'hydrogen_recombination',
                'hydrogen_collisional_ionization': 'hydrogen_collisional_ionization',
                'cie_cooling': 'cie_cooling',
                'cooling_safety_factor': 'cooling_safety_factor',
                'cooling_temperature_floor': 'cooling_temperature_floor',
            },
        }
        for group, names in groups.items():
            values = params.get(group, {})
            if isinstance(values, dict):
                for nested_name, input_name in names.items():
                    if nested_name in values:
                        flattened[input_name] = values[nested_name]
            flattened.pop(group, None)
        return flattened

    @staticmethod
    def _validate_keys(params):
        unknown_keys = sorted(set(params) - set(refparams))
        if unknown_keys:
            formatted = ', '.join(repr(key) for key in unknown_keys)
            raise ValueError(f'unknown run parameter(s): {formatted}')

    def _apply_defaults(self, params):
        missing_keys = []
        nested_keys = {
            'nogrid', 'noghost', 'coordsys', 'timesim', 'gamma', 'CFL',
            'boundcond', 'vel_inflow', 'vel_outflow', 'rho_inflow',
            'rho_outflow', 'temp_inflow', 'temp_outflow', 'mu_inflow',
            'mu_outflow', 'dtmin', 'dtmax', 'CodeUnits',
            'radiative_transfer_boundary_flux',
            'radiative_transfer_source_photon_rate',
            'radiative_transfer_direction',
        }
        for key, default in refparams.items():
            value = params.get(key, default)
            if key not in nested_keys:
                setattr(self, key, value)
            self._parameter_values[key] = value
            if key not in params and default is not None:
                missing_keys.append((key, default))
        return missing_keys

    def _parameter(self, name, default=None):
        return self._parameter_values.get(name, default)

    def _initialize_parameter_groups(self):
        self._sync_hydrodynamics_parameters()
        self._sync_boundary_parameters()
        self._sync_timestep_parameters()
        self._sync_thermochemistry_parameters()
        self._sync_gravity_parameters()
        self._sync_output_parameters()
        self._sync_diagnostics_parameters()
        self._sync_mesh_parameters()
        self._sync_chemistry_parameters()
        self._sync_angular_momentum_parameters()
        self._sync_dark_matter_parameters()
        self._sync_dual_energy_parameters()
        self._sync_positivity_parameters()
        self.cosmology = CosmologyParameters(
            type=self.cosmology_type,
            t_ref=self.cosmology_t_ref,
            a_ref=self.cosmology_a_ref,
            omega_m=self.cosmology_omega_m,
            omega_lambda=self.cosmology_omega_lambda,
            hubble_ref=self.cosmology_hubble_ref,
        )
        self._sync_radiation_parameters()

    def _sync_hydrodynamics_parameters(self):
        self.hydrodynamics = HydrodynamicsParameters(
            eos_type=self.EOStype,
            gamma=self._parameter('gamma'),
            temperature=self.temperature,
            CFL=self._parameter('CFL'),
            order=self.order,
            riemann_solver=self.riemann_solver,
            flux_limiter=self.flux_limiter,
            dual_energy=self.dual_energy,
            dual_energy_pressure_selection=self.dual_energy_pressure_selection,
            dual_energy_entropy_limiter=self.dual_energy_entropy_limiter,
            positivity_preserving=self.positivity_preserving,
            positivity_density_floor=self.positivity_density_floor,
            positivity_energy_floor=self.positivity_energy_floor,
            gas_angular_momentum=self.gas_angular_momentum,
            gas_rotational_energy=self.gas_rotational_energy,
            angular_momentum_flux_scheme=self.angular_momentum_flux_scheme,
        )

    def _sync_boundary_parameters(self):
        self.boundary = BoundaryParameters(
            condition=self._parameter('boundcond'),
            inflow_velocity=self._parameter('vel_inflow'),
            inflow_density=self._parameter('rho_inflow'),
            inflow_temperature=self._parameter('temp_inflow'),
            inflow_mu=self._parameter('mu_inflow'),
            outflow_velocity=self._parameter('vel_outflow'),
            outflow_density=self._parameter('rho_outflow'),
            outflow_temperature=self._parameter('temp_outflow'),
            outflow_mu=self._parameter('mu_outflow'),
            cosmological_background_reconstruction=(
                self.cosmological_background_boundary_reconstruction
            ),
        )

    def _sync_timestep_parameters(self):
        self.timestep = TimestepParameters(
            dtmin=self._parameter('dtmin'),
            dtmax=self._parameter('dtmax'),
            cfl_density_floor=self.cfl_density_floor,
            hydro_temperature_floor=self.hydro_temperature_floor,
            cooling_safety_factor=self.cooling_safety_factor,
            hydrogen_source_CFL=self.hydrogen_source_CFL,
            hydrogen_source_dtmin=self.hydrogen_source_dtmin,
            relaxation_damping_time=self.relaxation_damping_time,
        )

    def _sync_thermochemistry_parameters(self):
        self.thermochemistry = ThermochemistryParameters(
            network=self.thermochemistry_network,
            cie_cooling=self.cie_cooling,
            cie_ion_fraction_table=self.cie_ion_fraction_table,
            cie_cooling_table=self.cie_cooling_table,
            cie_abundance_file=self.cie_abundance_file,
            metallicity=self.metallicity,
            cooling_safety_factor=self.cooling_safety_factor,
            cooling_temperature_floor=self.cooling_temperature_floor,
            hydrogen_chemistry=self.hydrogen_chemistry,
            hydrogen_recombination=self.hydrogen_recombination,
            hydrogen_collisional_ionization=self.hydrogen_collisional_ionization,
            hydrogen_atomic_cooling=self.hydrogen_atomic_cooling,
            hydrogen_thermal_coupling=self.hydrogen_thermal_coupling,
            hydrogen_update_mu=self.hydrogen_update_mu,
            compton_cmb_enabled=self.compton_cmb_enabled,
            metal_pie_enabled=self.metal_pie_enabled,
            pie_uvbg_implicit_tolerance=self.pie_uvbg_implicit_tolerance,
            pie_uvbg_implicit_max_retries=self.pie_uvbg_implicit_max_retries,
            pie_uvbg_implicit_max_iterations=self.pie_uvbg_implicit_max_iterations,
            pie_uvbg_implicit_step_doubling=self.pie_uvbg_implicit_step_doubling,
            hydrogen_source_solver=self.hydrogen_source_solver,
        )

    def _sync_gravity_parameters(self):
        model = self.gravity if self.gravity is not None else None
        self.gravity = GravityParameters(
            selfgravity=self.selfgravity,
            externalgravity=self.externalgravity,
            potential=self.gravity_potential,
            coordinate=self.gravity_coordinate,
            acceleration=self.gravity_acceleration,
            selfgravity_softening=self.selfgravity_softening,
            selfgravity_boundary_acceleration=self.selfgravity_boundary_acceleration,
            cosmological=self.cosmological_gravity,
            potential_energy=self.gravity_potential_energy,
            model=model,
        )

    def _sync_output_parameters(self):
        self.output = OutputParameters(
            directory=self.outdir,
            savedir=self.savedir,
            filename_prefix=self.outfileprefix,
            cadence=self.outdeltatime,
            time_list_filename=self.outputtimefilename,
        )

    def _sync_simulation_parameters(self):
        self.simulation = SimulationParameters(
            name=self.simname,
            initial_condition_filename=self.ICfilename,
            coordinate_system=self._parameter('coordsys'),
            final_time=self._parameter('timesim'),
            current_time=getattr(self, 'time', None),
            box_size=getattr(self, 'boxsize', None),
            cosmological_expansion=self.cosmological_expansion,
            supercomoving_coordinates=self.supercomoving_coordinates,
            coordinate_frame=self.coordinate_frame,
            time_coordinate=self.time_coordinate,
            velocity_representation=self.velocity_representation,
            density_representation=self.density_representation,
            pressure_representation=self.pressure_representation,
            temperature_representation=self.temperature_representation,
        )

    def _sync_diagnostics_parameters(self):
        self.diagnostics = DiagnosticsParameters(
            verbose=self.verbose,
            energy_diagnostics=self.energy_diagnostics,
            temperature_jump_error_threshold=self.temperature_jump_error_threshold,
            temperature_plot_ymin=getattr(self, 'temperature_plot_ymin', None),
            plot_exclude_outer_cells=getattr(self, 'plot_exclude_outer_cells', 0),
        )

    def _sync_mesh_parameters(self):
        self.mesh = MeshParameters(
            ghost_cells=self._parameter('noghost'),
            area=self.area,
            grid_cells=self._parameter('nogrid'),
        )

    def _sync_chemistry_parameters(self):
        self.chemistry = ChemistryParameters(
            key=self.chemistry_key,
            hydrogen_mass_fraction=self.hydrogen_mass_fraction,
            helium_mass_fraction=self.helium_mass_fraction,
            hydrogen_xHI_initial=self.hydrogen_xHI_initial,
            hydrogen_xHI_inflow=self.hydrogen_xHI_inflow,
            hydrogen_xHI_outflow=self.hydrogen_xHI_outflow,
            helium_xHeI_initial=self.hydrogen_helium_xHeI_initial,
            helium_xHeII_initial=self.hydrogen_helium_xHeII_initial,
            helium_xHeIII_initial=self.hydrogen_helium_xHeIII_initial,
            update_mu=self.hydrogen_update_mu,
            helium_coupled_implicit=self.hydrogen_helium_coupled_implicit,
            source_skip_floor_cells=self.hydrogen_source_skip_floor_cells,
            source_density_floor=self.hydrogen_source_density_floor,
            source_floor_temperature_tolerance=(
                self.hydrogen_source_floor_temperature_tolerance
            ),
            hybrid_change_tolerance=self.hydrogen_hybrid_change_tolerance,
            implicit_tolerance=self.hydrogen_implicit_tolerance,
            implicit_convergence_tolerance=(
                self.hydrogen_implicit_convergence_tolerance
            ),
            implicit_max_iterations=self.hydrogen_implicit_max_iterations,
            implicit_fallback=self.hydrogen_implicit_fallback,
            implicit_max_refinements=self.hydrogen_implicit_max_refinements,
            split_implicit_max_subcycles=(
                self.hydrogen_split_implicit_max_subcycles
            ),
            implicit_absolute_temperature_tolerance=(
                self.hydrogen_implicit_absolute_temperature_tolerance
            ),
            implicit_absolute_xhi_tolerance=(
                self.hydrogen_implicit_absolute_xhi_tolerance
            ),
            implicit_debug=self.hydrogen_implicit_debug,
            alpha_B=self.hydrogen_alpha_B,
            beta=self.hydrogen_beta,
        )

    def _sync_angular_momentum_parameters(self):
        self.angular_momentum = AngularMomentumParameters(
            enabled=self.gas_angular_momentum,
            rotational_energy=self.gas_rotational_energy,
            flux_scheme=self.angular_momentum_flux_scheme,
            energy_margin_fraction=self.angular_momentum_energy_margin_fraction,
            specific_angular_momentum=self.gas_specific_angular_momentum,
            inflow=self.specific_angular_momentum_inflow,
            outflow=self.specific_angular_momentum_outflow,
        )

    def _sync_dark_matter_parameters(self):
        self.dark_matter_config = DarkMatterParameters(
            crossing_safety_factor=self.dark_matter_crossing_safety_factor,
            crossing_batch_fraction=self.dark_matter_crossing_batch_fraction,
            global_timestep_limit=self.dark_matter_global_timestep_limit,
            density_bins=getattr(self, 'dm_density_bins', None),
        )

    def _sync_dual_energy_parameters(self):
        self.dual_energy_config = DualEnergyParameters(
            enabled=self.dual_energy,
            eta1=self.dual_energy_eta1,
            eta2=self.dual_energy_eta2,
            consistency_factor=self.dual_energy_consistency_factor,
            entropy_limiter=self.dual_energy_entropy_limiter,
            pressure_selection=self.dual_energy_pressure_selection,
            pressure_floor=self.dual_energy_pressure_floor,
            switch=self.dual_energy_switch,
        )

    def _sync_positivity_parameters(self):
        self.positivity = PositivityParameters(
            enabled=self.positivity_preserving,
            density_floor=self.positivity_density_floor,
            energy_floor=self.positivity_energy_floor,
        )

    def _sync_radiation_parameters(self):
        self.radiation = RadiationParameters(
            spectrum_filename=self.radiation_spectrum_filename,
            spectrum_total_photon_rate=self.radiation_spectrum_total_photon_rate,
            radiative_transfer=self.radiative_transfer,
            radiation_pressure=self.radiation_pressure,
            metal_pie_enabled=self.metal_pie_enabled,
            metal_pie_table_filename=self.metal_pie_table_filename,
            metal_pie_table=self.metal_pie_table,
            number_of_radiation_groups=self.number_of_radiation_groups,
            group_edges_eV=self.radiation_group_edges_eV,
            group_sigma_gamma=self.radiation_group_sigma_gamma,
            group_epsilon_gamma=self.radiation_group_epsilon_gamma,
            group_sigma_gamma_HeI=self.radiation_group_sigma_gamma_HeI,
            group_sigma_gamma_HeII=self.radiation_group_sigma_gamma_HeII,
            group_epsilon_gamma_HeI=self.radiation_group_epsilon_gamma_HeI,
            group_epsilon_gamma_HeII=self.radiation_group_epsilon_gamma_HeII,
            radiative_transfer_method=self.radiative_transfer_method,
            radiative_transfer_temporal_scheme=self.radiative_transfer_temporal_scheme,
            radiative_transfer_direction=self._parameter('radiative_transfer_direction'),
            boundary_flux=self._parameter('radiative_transfer_boundary_flux'),
            source_photon_rate=self._parameter('radiative_transfer_source_photon_rate'),
            boundary_flux_groups=self.radiative_transfer_boundary_flux_groups,
            source_photon_rate_groups=self.radiative_transfer_source_photon_rate_groups,
            c2ray_max_iterations=self.radiative_transfer_c2ray_max_iterations,
            c2ray_tolerance=self.radiative_transfer_c2ray_tolerance,
            c2ray_relaxation=self.radiative_transfer_c2ray_relaxation,
            c2ray_nonconvergence=self.radiative_transfer_c2ray_nonconvergence,
            c2ray_ode_max_iterations=self.radiative_transfer_c2ray_ode_max_iterations,
            c2ray_ode_tolerance=self.radiative_transfer_c2ray_ode_tolerance,
            radiation_pressure_efficiency=self.radiation_pressure_efficiency,
            compton_cmb_enabled=self.compton_cmb_enabled,
            compton_cmb_redshift=self.compton_cmb_redshift,
            cmb_temperature_0=self.cmb_temperature_0,
            hydrogen_radiation_field=self.hydrogen_radiation_field,
            hydrogen_radiation_evolution=self.hydrogen_radiation_evolution,
            hydrogen_ngamma_initial=self.hydrogen_ngamma_initial,
            hydrogen_ngamma_inflow=self.hydrogen_ngamma_inflow,
            hydrogen_ngamma_outflow=self.hydrogen_ngamma_outflow,
            hydrogen_sigma_gamma=self.hydrogen_sigma_gamma,
            hydrogen_epsilon_gamma=self.hydrogen_epsilon_gamma,
            hydrogen_photon_energy=self.hydrogen_photon_energy,
            stellar_spectrum_type=self.stellar_spectrum_type,
            stellar_spectrum_blackbody_temperature_K=(
                self.stellar_spectrum_blackbody_temperature_K
            ),
            star_emission_rates=self.star_emission_rates,
            ionizing_photon_energy_erg=self.ionizing_photon_energy_erg,
            metal_pie_photoheating_max_density_cm3=(
                self.metal_pie_photoheating_max_density_cm3
            ),
            metal_pie_redshift=self.metal_pie_redshift,
        )

    def _initialize_units(self, params):
        code_units_value = self._parameter('CodeUnits')
        if code_units_value is None:
            raise ValueError(
                'run parameters must define CodeUnits with an internal unit system'
            )
        self.set_code_units(CodeUnits.from_mapping(code_units_value))

    def set_code_units(self, code_units):
        """Update the unit system stored in the nested units group."""
        self.unit_system = code_units.unit_system
        self.units = UnitsParameters(
            CodeUnits=code_units,
            unit_system=self.unit_system,
        )

    def _configure_cosmology(self):
        if not self.cosmological_expansion:
            return
        supported = (
            None, 'einstein_de_sitter', 'EinsteinDeSitter',
            'lambda_cdm', 'LambdaCDM', 'lcdm',
        )
        if self.cosmology_type not in supported:
            raise ValueError(f'unsupported cosmology_type: {self.cosmology_type}')

        is_eds = self.cosmology_type in (
            None, 'einstein_de_sitter', 'EinsteinDeSitter'
        )
        cosmology_class = EinsteinDeSitter if is_eds else LambdaCDM
        cosmology_kwargs = {
            't_ref': self.cosmology_t_ref,
            'a_ref': self.cosmology_a_ref,
        }
        if cosmology_class is LambdaCDM:
            cosmology_kwargs.update({
                'omega_m': self.cosmology_omega_m,
                'omega_lambda': self.cosmology_omega_lambda,
                'hubble_ref': self.cosmology_hubble_ref,
            })
        self.cosmology.model = cosmology_class.from_code_units(
            self.units.CodeUnits, **cosmology_kwargs
        )
        if self.supercomoving_coordinates:
            self.coordinate_frame = 'comoving'
            self.time_coordinate = 'supercomoving'
            self.velocity_representation = 'supercomoving_peculiar'
            self.density_representation = 'comoving'
            self.pressure_representation = 'supercomoving'
            self.temperature_representation = 'supercomoving'

    def _load_optional_physics(self, params):
        if params.get('radiation_spectrum_filename') is not None:
            self.load_radiation_spectrum(params.get('outdir'))
        if params.get('metal_pie_enabled', False) and params.get('metal_pie_table_filename'):
            self.load_metal_pie_table(params.get('outdir'))
        if (
            self.metal_pie_enabled
            and self.metal_pie_table is not None
            and self.metal_pie_table.is_hm12_uv_background
            and self.radiative_transfer
        ):
            raise ValueError(
                'HM12 PIE UV-background tables require '
                'radiative_transfer: false in the first implementation'
            )
        self._sync_radiation_parameters()
        self._sync_thermochemistry_parameters()

    def _warn_defaulted_parameters(self, missing_keys):
        if int(self.verbose) <= 0:
            return
        for key, value in missing_keys:
            warnings.warn(
                f'run parameter {key!r} was not provided; using default {value!r}',
                ParameterDefaultWarning,
                stacklevel=3,
            )

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
            power_unit = self.units.CodeUnits.energy_unit / self.units.CodeUnits.time_unit
            rates = np.asarray(self.star_emission_rates, dtype=float) * power_unit
            energies = np.asarray(self.ionizing_photon_energy_erg, dtype=float) * unyt.erg
            total_rate = getattr(self, 'radiation_spectrum_total_photon_rate', None)
            if total_rate is not None:
                if hasattr(total_rate, 'to_value'):
                    target_rate_s = float(total_rate.to_value(1.0 / unyt.s))
                else:
                    target_rate_s = code_quantity_to_cgs(
                        total_rate,
                        self.units.CodeUnits,
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
            self._sync_radiation_parameters()

    def load_metal_pie_table(self, base_directory=None):
        filename = resolve_spectrum_filename(
            self.metal_pie_table_filename, base_directory
        )
        self.metal_pie_table = MetalPIETable(filename)
        self._sync_radiation_parameters()

    def set_cosmology_model(self, model):
        """Update the cosmology model while retaining structured settings."""
        self.cosmology = CosmologyParameters(
            type=self.cosmology_type,
            t_ref=self.cosmology_t_ref,
            a_ref=self.cosmology_a_ref,
            omega_m=self.cosmology_omega_m,
            omega_lambda=self.cosmology_omega_lambda,
            hubble_ref=self.cosmology_hubble_ref,
            model=model,
        )
