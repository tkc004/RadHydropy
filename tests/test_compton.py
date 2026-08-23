import numpy as np
from types import SimpleNamespace

from radhydropy.thermo_networks.compton import cmb_compton_rate
from radhydropy.thermo_networks.hydrogen import (
    _coupled_implicit_source_update,
    _fast_source_state,
    _fast_sync_state_to_fluid,
    apply_thermochemistry_fast,
    ionization_fraction_rate,
    thermal_rate,
)
from radhydropy.thermo_networks.hydrogen_helium import _rates
from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS
import radhydropy.chemistry_species.hydrogen as hydrogen_species
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.units import CodeUnits


def _implicit_hydrogen_state(temperature, xhi, recombination, collisional,
                              atomic_cooling):
    mu = float(hydrogen_species.mean_molecular_weight_mu(
        np.array([xhi]), hydrogen_mass_fraction=1.0
    )[0])
    specific_energy = np.array([
        1.5 * BOLTZMANN_CONSTANT_CGS * temperature
        / (mu * PROTON_MASS_CGS)
    ])
    return {
        'rho_g_cm3': np.array([PROTON_MASS_CGS * 1.0e-2]),
        'temperature_K': np.array([temperature]),
        'xHI': np.array([xhi]),
        'hydrogen_mass_fraction': 1.0,
        'gamma': 5.0 / 3.0,
        'mu': np.array([mu]),
        'hydrogen_update_mu': True,
        'recombination': recombination,
        'collisional_ionization': collisional,
        'atomic_cooling': atomic_cooling,
        'sigma_gamma_cm2': 0.0,
        'epsilon_gamma_erg': 0.0,
        'compton_cmb_enabled': False,
        'compton_cmb_redshift': 0.0,
        'cmb_temperature_0_K': 2.7255,
        'alpha_B_cm3_s': None,
        'beta_cm3_s': None,
        'thermal_coupling': True,
        'specific_energy_erg_g': specific_energy.copy(),
        'specific_total_energy_erg_g': specific_energy.copy(),
        'specific_kinetic_energy_erg_g': np.zeros(1),
    }


def test_coupled_implicit_source_evolves_recombination_and_energy_together():
    state = _implicit_hydrogen_state(
        temperature=1.0e4,
        xhi=0.5,
        recombination=True,
        collisional=False,
        atomic_cooling=False,
    )
    old_energy = state['specific_energy_erg_g'].copy()
    assert _coupled_implicit_source_update(
        state, 1.0e11, tolerance=1.0e-8, max_iterations=32
    )
    assert state['xHI'][0] > 0.5
    np.testing.assert_allclose(state['specific_energy_erg_g'], old_energy)
    assert 0.0 < state['xHI'][0] < 1.0
    assert np.isfinite(state['temperature_K'][0])


def test_coupled_implicit_source_handles_collisional_ionization():
    state = _implicit_hydrogen_state(
        temperature=1.0e6,
        xhi=0.99,
        recombination=False,
        collisional=True,
        atomic_cooling=False,
    )
    assert _coupled_implicit_source_update(
        state, 1.0e9, tolerance=1.0e-8, max_iterations=32
    )
    assert state['xHI'][0] < 0.99
    assert 0.0 < state['xHI'][0] < 1.0
    assert state['temperature_K'][0] > 0.0


def test_coupled_implicit_source_satisfies_both_backward_euler_residuals():
    state = _implicit_hydrogen_state(
        temperature=1.0e5,
        xhi=0.5,
        recombination=True,
        collisional=True,
        atomic_cooling=True,
    )
    old_energy = state['specific_energy_erg_g'].copy()
    old_xhi = state['xHI'].copy()
    dt = 1.0e9
    assert _coupled_implicit_source_update(
        state, dt, tolerance=1.0e-8, max_iterations=32
    )
    thermal = thermal_rate(state, None)
    chemistry = ionization_fraction_rate(state, None)
    energy_residual = (
        state['specific_energy_erg_g'] - old_energy
        - dt * thermal / state['rho_g_cm3']
    ) / old_energy
    xhi_residual = state['xHI'] - old_xhi - dt * chemistry
    assert np.max(np.abs(energy_residual)) < 1.0e-7
    assert np.max(np.abs(xhi_residual)) < 1.0e-7


def test_fast_source_dispatches_to_coupled_implicit_solver():
    units = CodeUnits.from_mapping(
        {
            'UnitMass_in_cgs': 1.0e33,
            'UnitLength_in_cgs': 1.0e18,
            'UnitVelocity_in_cgs': 1.0e5,
            'UnitCurrent_in_cgs': 1.0,
            'UnitTemp_in_cgs': 1.0,
        }
    )
    temperature = 1.0e4
    xhi = 0.5
    mu = float(hydrogen_species.mean_molecular_weight_mu(
        np.array([xhi]), hydrogen_mass_fraction=1.0
    )[0])
    specific_energy = 1.5 * BOLTZMANN_CONSTANT_CGS * temperature / (
        mu * PROTON_MASS_CGS
    )
    energy_code = specific_energy * 1.0e33 / units.energy_unit.to_value('erg')
    par = SimpleNamespace(
        CodeUnits=units,
        noghost=0,
        nogrid=1,
        gamma=5.0 / 3.0,
        hydrogen_chemistry=True,
        hydrogen_mass_fraction=1.0,
        hydrogen_source_CFL=0.1,
        hydrogen_source_dtmin=0.0,
        hydrogen_source_solver='coupled_implicit',
        hydrogen_implicit_tolerance=1.0e-7,
        hydrogen_implicit_max_iterations=32,
        hydrogen_implicit_fallback='error',
        hydrogen_recombination=True,
        hydrogen_collisional_ionization=False,
        hydrogen_atomic_cooling=False,
        hydrogen_update_mu=True,
        hydrogen_thermal_coupling=True,
        compton_cmb_enabled=False,
        hydrogen_radiation_field=False,
        radiative_transfer=False,
        radiative_transfer_direction=1,
        hydrogen_photon_energy=13.6,
    )
    fluid = SimpleNamespace(
        rho=np.array([1.0e-3]),
        temp=np.array([temperature]),
        vel=np.array([0.0]),
        Mass=np.array([1.0]),
        Energy=np.array([energy_code]),
        pre=np.array([1.0]),
        xHI=np.array([xhi]),
        mu=np.array([mu]),
        eos=SimpleNamespace(gamma=5.0 / 3.0),
    )
    fluid.SetHydrogenMu = lambda hydrogen_mass_fraction=1.0: None
    fluid.SetPressure = lambda: None
    mesh = SimpleNamespace(
        boundary=np.array([0.0, 1.0]),
        xdelta=np.array([1.0]),
        vol=np.array([1.0]),
    )
    result = apply_thermochemistry_fast(1.0e-4, mesh, fluid, par)
    assert result['source_steps'] == 1
    assert fluid.xHI[0] > xhi
    assert np.isfinite(fluid.temp[0])


def test_cmb_compton_source_is_opt_in_and_has_expected_sign():
    temperature = np.array([1.0, 1.0e4])
    electrons = np.ones(2)

    disabled = cmb_compton_rate(temperature, electrons, redshift=10.0)
    enabled = cmb_compton_rate(
        temperature,
        electrons,
        enabled=True,
        redshift=10.0,
    )

    assert np.all(disabled == 0.0)
    assert enabled[0] > 0.0
    assert enabled[1] < 0.0


def test_hydrogen_thermal_rate_includes_optional_compton_source():
    state = {
        "rho_g_cm3": np.array([PROTON_MASS_CGS]),
        "temperature_K": np.array([1.0]),
        "xHI": np.array([0.0]),
        "hydrogen_mass_fraction": 1.0,
        "recombination": False,
        "collisional_ionization": False,
        "sigma_gamma_cm2": 0.0,
        "epsilon_gamma_erg": 0.0,
        "compton_cmb_enabled": True,
        "compton_cmb_redshift": 10.0,
        "cmb_temperature_0_K": 2.7255,
    }
    rate = thermal_rate(state, None)
    state["compton_cmb_enabled"] = False
    background_rate = thermal_rate(state, None)
    expected = cmb_compton_rate(
        state["temperature_K"],
        np.array([1.0]),
        enabled=True,
        redshift=10.0,
    )
    np.testing.assert_allclose(rate - background_rate, expected)


def test_hydrogen_thermal_rate_can_disable_atomic_cooling():
    state = {
        "rho_g_cm3": np.array([PROTON_MASS_CGS]),
        "temperature_K": np.array([1.0e5]),
        "xHI": np.array([0.5]),
        "hydrogen_mass_fraction": 1.0,
        "recombination": True,
        "collisional_ionization": True,
        "atomic_cooling": False,
        "sigma_gamma_cm2": 0.0,
        "epsilon_gamma_erg": 0.0,
        "compton_cmb_enabled": False,
        "compton_cmb_redshift": 0.0,
        "cmb_temperature_0_K": 2.7255,
    }
    assert np.allclose(thermal_rate(state, None), 0.0)


def test_hydrogen_helium_thermal_rate_uses_electron_density():
    state = {
        "rho_g_cm3": np.array([PROTON_MASS_CGS]),
        "hydrogen_mass_fraction": 0.7,
        "helium_mass_fraction": 0.28,
        "temperature_K": np.array([1.0]),
        "xHI": np.array([0.0]),
        "xHeI": np.array([0.0]),
        "xHeIII": np.array([1.0]),
        "sigma_gamma_cm2": {
            "HI": np.zeros(1), "HeI": np.zeros(1), "HeII": np.zeros(1)
        },
        "epsilon_gamma_erg": {
            "HI": np.zeros(1), "HeI": np.zeros(1), "HeII": np.zeros(1)
        },
        "compton_cmb_enabled": True,
        "compton_cmb_redshift": 10.0,
        "cmb_temperature_0_K": 2.7255,
    }
    ngamma = np.zeros((1, 1))
    _, _, _, rate = _rates(state, ngamma)
    expected_ne = state["ne_cm3"].copy()
    expected = cmb_compton_rate(
        state["temperature_K"],
        expected_ne,
        enabled=True,
        redshift=10.0,
    )
    state["compton_cmb_enabled"] = False
    _, _, _, background_rate = _rates(state, ngamma)
    np.testing.assert_allclose(rate - background_rate, expected)


def test_fast_source_state_round_trips_supercomoving_temperature():
    units = CodeUnits.from_mapping(
        {
            "UnitMass_in_cgs": 1.0e33,
            "UnitLength_in_cgs": 1.0e18,
            "UnitVelocity_in_cgs": 1.0e5,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        }
    )
    cosmology = EinsteinDeSitter.from_code_units(units)
    cosmic_time = 0.01418666885
    tau = float(cosmology.supercomoving_time(cosmic_time))
    scale_factor = float(cosmology.scale_factor(cosmic_time))
    physical_temperature = 275.2755

    par = SimpleNamespace(
        CodeUnits=units,
        noghost=0,
        nogrid=1,
        gamma=5.0 / 3.0,
        hydrogen_mass_fraction=0.76,
        hydrogen_source_CFL=0.1,
        hydrogen_source_dtmin=0.0,
        hydrogen_recombination=False,
        hydrogen_collisional_ionization=False,
        hydrogen_thermal_coupling=False,
        hydrogen_update_mu=False,
        compton_cmb_enabled=True,
        compton_cmb_redshift=100.0,
        cmb_temperature_0=2.7255,
        supercomoving_coordinates=True,
        cosmology=cosmology,
        fluid_time=tau,
    )
    mesh = SimpleNamespace(
        boundary=np.array([0.0, 1.0]),
        xdelta=np.array([1.0]),
        vol=np.array([4.0 * np.pi / 3.0]),
    )
    fluid = SimpleNamespace(
        rho=np.array([scale_factor**3]),
        temp=np.array([physical_temperature * scale_factor**2]),
        vel=np.array([0.0]),
        Mass=np.array([1.0]),
        Energy=np.array([1.0]),
        xHI=np.array([0.9998]),
        mu=np.array([1.0 / (0.76 * (2.0 - 0.9998))]),
        eos=SimpleNamespace(gamma=5.0 / 3.0),
        time=tau,
    )

    state = _fast_source_state(mesh, fluid, par)
    assert np.isclose(state["temperature_K"][0], physical_temperature)
    density_unit_cgs = units.mass_in_cgs / units.length_in_cgs**3
    assert np.isclose(state["rho_g_cm3"][0], density_unit_cgs)
    _fast_sync_state_to_fluid(state, fluid, par)
    assert np.isclose(fluid.temp[0], physical_temperature * scale_factor**2)
