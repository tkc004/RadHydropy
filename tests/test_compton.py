import numpy as np
import pytest
from types import SimpleNamespace
from tests.parameter_fixtures import parameter_namespace

from radhydropy.thermo_networks.compton import cmb_compton_rate
from radhydropy.thermo_networks.hydrogen import (
    _coupled_implicit_source_update,
    _fast_source_state,
    _fast_sync_state_to_fluid,
    _fast_update_temperature_from_energy,
    _split_implicit_source_state_update,
    _source_stiffness_groups,
    apply_thermochemistry_fast,
    ionization_fraction_rate,
    thermal_rate,
)
from radhydropy.thermo_networks.hydrogen_helium import _rates
from radhydropy.constants import (
    BOLTZMANN_CONSTANT_CGS,
    PROTON_MASS_CGS,
    SPEED_OF_LIGHT_CGS,
)
import radhydropy.chemistry_species.hydrogen as hydrogen_species
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.units import CodeUnits


def _implicit_hydrogen_state(temperature, xhi, recombination, collisional,
                              atomic_cooling, density_factor=1.0):
    mu = float(hydrogen_species.mean_molecular_weight_mu(
        np.array([xhi]), hydrogen_mass_fraction=1.0
    )[0])
    specific_energy = np.array([
        1.5 * BOLTZMANN_CONSTANT_CGS * temperature
        / (mu * PROTON_MASS_CGS)
    ])
    return {
        'rho_g_cm3': np.array([PROTON_MASS_CGS * density_factor]),
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


def _explicit_reference_update(state, dt_s, steps):
    """Small-forward-step reference integration for the local source ODE."""
    reference = {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in state.items()
    }
    sub_dt = dt_s / steps
    for _ in range(steps):
        thermal = thermal_rate(reference, None)
        chemistry = ionization_fraction_rate(reference, None)
        reference['specific_energy_erg_g'] = np.maximum(
            reference['specific_energy_erg_g']
            + sub_dt * thermal / reference['rho_g_cm3'],
            1.0e-30,
        )
        reference['xHI'] = np.clip(
            reference['xHI'] + sub_dt * chemistry,
            1.0e-12,
            1.0 - 1.0e-12,
        )
        reference['specific_total_energy_erg_g'] = (
            reference['specific_energy_erg_g']
            + reference['specific_kinetic_energy_erg_g']
        )
        _fast_update_temperature_from_energy(reference)
    return reference


def _source_test_problem(
    solver='coupled_implicit',
    fallback='explicit',
    supercomoving=False,
):
    units = CodeUnits.from_mapping(
        {
            'UnitMass_in_cgs': 1.0e33,
            'UnitLength_in_cgs': 1.0e18,
            'UnitVelocity_in_cgs': 1.0e5,
            'UnitCurrent_in_cgs': 1.0,
            'UnitTemp_in_cgs': 1.0,
        }
    )
    cosmology = EinsteinDeSitter.from_code_units(units)
    cosmic_time = 0.01418666885
    tau = float(cosmology.supercomoving_time(cosmic_time))
    scale_factor = float(cosmology.scale_factor(cosmic_time))
    physical_density = 1.0e-24
    physical_temperature = 1.0e4
    xhi = 0.5
    mu = float(hydrogen_species.mean_molecular_weight_mu(
        np.array([xhi]), hydrogen_mass_fraction=1.0
    )[0])
    physical_specific_energy = (
        1.5 * BOLTZMANN_CONSTANT_CGS * physical_temperature
        / (mu * PROTON_MASS_CGS)
    )
    density_unit = units.mass_in_cgs / units.length_in_cgs**3
    density_factor = scale_factor**3 if supercomoving else 1.0
    temperature_factor = scale_factor**2 if supercomoving else 1.0
    fluid_density = physical_density / density_unit * density_factor
    fluid_temperature = physical_temperature * temperature_factor
    fluid_energy = (
        physical_specific_energy * units.mass_in_cgs
        / units.energy_unit.to_value('erg') * temperature_factor
    )
    par = parameter_namespace(
        CodeUnits=units,
        noghost=0,
        nogrid=1,
        gamma=5.0 / 3.0,
        hydrogen_chemistry=True,
        hydrogen_mass_fraction=1.0,
        hydrogen_source_CFL=0.1,
        hydrogen_source_dtmin=0.0,
        hydrogen_source_solver=solver,
        hydrogen_implicit_tolerance=1.0e-7,
        hydrogen_implicit_max_iterations=32,
        hydrogen_implicit_fallback=fallback,
        hydrogen_recombination=True,
        hydrogen_collisional_ionization=False,
        hydrogen_atomic_cooling=False,
        hydrogen_update_mu=False,
        hydrogen_thermal_coupling=True,
        compton_cmb_enabled=False,
        hydrogen_radiation_field=False,
        radiative_transfer=False,
        radiative_transfer_direction=1,
        hydrogen_photon_energy=13.6,
        supercomoving_coordinates=supercomoving,
        cosmology=cosmology,
        fluid_time=tau,
    )
    fluid = SimpleNamespace(
        rho=np.array([fluid_density]),
        temp=np.array([fluid_temperature]),
        vel=np.array([0.0]),
        Mass=np.array([1.0]),
        Energy=np.array([fluid_energy]),
        pre=np.array([1.0]),
        xHI=np.array([xhi]),
        mu=np.array([mu]),
        eos=SimpleNamespace(gamma=5.0 / 3.0),
    )
    mesh = SimpleNamespace(
        boundary=np.array([0.0, 1.0]),
        xdelta=np.array([1.0]),
        vol=np.array([1.0]),
    )
    return units, par, fluid, mesh, scale_factor


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
    state['beta_cm3_s'] = np.array([1.0e-12])
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


def test_coupled_implicit_chemistry_limits():
    recombination = _implicit_hydrogen_state(
        temperature=1.0e4,
        xhi=0.5,
        recombination=True,
        collisional=False,
        atomic_cooling=False,
    )
    recombination['alpha_B_cm3_s'] = np.array([1.0e-12])
    assert _coupled_implicit_source_update(
        recombination, 1.0e11, tolerance=1.0e-8, max_iterations=32
    )
    assert recombination['xHI'][0] > 0.5

    collisional = _implicit_hydrogen_state(
        temperature=1.0e6,
        xhi=0.5,
        recombination=False,
        collisional=True,
        atomic_cooling=False,
    )
    collisional['beta_cm3_s'] = np.array([1.0e-12])
    assert _coupled_implicit_source_update(
        collisional, 1.0e11, tolerance=1.0e-8, max_iterations=32
    )
    assert collisional['xHI'][0] < 0.5

    no_chemistry = _implicit_hydrogen_state(
        temperature=1.0e5,
        xhi=0.5,
        recombination=False,
        collisional=False,
        atomic_cooling=False,
    )
    original_xhi = no_chemistry['xHI'].copy()
    original_energy = no_chemistry['specific_energy_erg_g'].copy()
    assert _coupled_implicit_source_update(
        no_chemistry, 1.0e15, tolerance=1.0e-8, max_iterations=32
    )
    np.testing.assert_allclose(no_chemistry['xHI'], original_xhi)
    np.testing.assert_allclose(
        no_chemistry['specific_energy_erg_g'],
        original_energy,
    )


def test_coupled_implicit_compton_thermal_limit_has_correct_direction():
    cold = _implicit_hydrogen_state(
        temperature=1.0,
        xhi=0.0,
        recombination=False,
        collisional=False,
        atomic_cooling=False,
    )
    cold['compton_cmb_enabled'] = True
    cold['compton_cmb_redshift'] = 10.0
    assert _coupled_implicit_source_update(
        cold, 1.0e13, tolerance=1.0e-8, max_iterations=32
    )
    assert cold['temperature_K'][0] > 1.0

    hot = _implicit_hydrogen_state(
        temperature=1.0e5,
        xhi=0.0,
        recombination=False,
        collisional=False,
        atomic_cooling=False,
    )
    hot['compton_cmb_enabled'] = True
    hot['compton_cmb_redshift'] = 10.0
    assert _coupled_implicit_source_update(
        hot, 1.0e13, tolerance=1.0e-8, max_iterations=32
    )
    assert hot['temperature_K'][0] < 1.0e5

    cooling = _implicit_hydrogen_state(
        temperature=1.0e5,
        xhi=0.5,
        recombination=True,
        collisional=True,
        atomic_cooling=True,
    )
    old_energy = cooling['specific_energy_erg_g'].copy()
    assert _coupled_implicit_source_update(
        cooling, 1.0e8, tolerance=1.0e-8, max_iterations=32
    )
    assert cooling['specific_energy_erg_g'][0] < old_energy[0]


def test_trust_region_leaves_cold_floor_under_stiff_compton_heating():
    """Regress the late-time cosmological virial-shock source state."""
    temperature_floor = 2.6939889101121835e-5
    xhi = 0.9418750059757972
    n_hydrogen = 0.005471103569641479
    dt_s = 218254737078.46677
    state = _implicit_hydrogen_state(
        temperature=temperature_floor,
        xhi=xhi,
        recombination=True,
        collisional=True,
        atomic_cooling=True,
    )
    state.update({
        'rho_g_cm3': np.array([
            n_hydrogen * PROTON_MASS_CGS / 0.76
        ]),
        'hydrogen_mass_fraction': 0.76,
        'specific_energy_erg_g': np.array([2682.3951472516123]),
        'specific_total_energy_erg_g': np.array([2682.3951472516123]),
        'compton_cmb_enabled': True,
        'compton_cmb_redshift': 10.71686171470456,
        'temperature_floor_K': temperature_floor,
        'temperature_floor_tolerance': 1.0e-2,
        'active': np.array([True]),
    })
    state['mu'] = hydrogen_species.mean_molecular_weight_mu(
        state['xHI'], hydrogen_mass_fraction=0.76
    )
    _fast_update_temperature_from_energy(state)
    old_energy = state['specific_energy_erg_g'].copy()
    old_xhi = state['xHI'].copy()

    assert _coupled_implicit_source_update(
        state,
        dt_s,
        tolerance=1.0e-4,
        max_iterations=32,
        trust_region=True,
    )
    assert state['temperature_K'][0] > temperature_floor
    energy_residual = (
        state['specific_energy_erg_g'] - old_energy
        - dt_s * thermal_rate(state, None) / state['rho_g_cm3']
    ) / old_energy
    chemistry_residual = (
        state['xHI'] - old_xhi
        - dt_s * ionization_fraction_rate(state, None)
    )
    assert np.max(np.abs(energy_residual)) < 1.0e-4
    assert np.max(np.abs(chemistry_residual)) < 1.0e-4


def test_stiff_source_cell_isolated_from_quiet_cells():
    state = _implicit_hydrogen_state(
        temperature=1.0e4,
        xhi=0.5,
        recombination=True,
        collisional=False,
        atomic_cooling=False,
    )
    for key, value in list(state.items()):
        if isinstance(value, np.ndarray) and value.shape == (1,):
            state[key] = np.repeat(value, 8)
    state['active'] = np.ones(8, dtype=bool)
    state['rho_g_cm3'][-1] *= 1.0e8

    groups = _source_stiffness_groups(state, 1.0e12)

    assert len(groups) == 2
    np.testing.assert_array_equal(groups[-1], np.array([7]))


def test_coupled_implicit_matches_small_step_reference():
    implicit = _implicit_hydrogen_state(
        temperature=1.0e5,
        xhi=0.5,
        recombination=True,
        collisional=True,
        atomic_cooling=True,
    )
    reference = _explicit_reference_update(implicit, 1.0e6, 10000)
    assert _coupled_implicit_source_update(
        implicit, 1.0e6, tolerance=1.0e-8, max_iterations=32
    )
    np.testing.assert_allclose(implicit['xHI'], reference['xHI'], rtol=2.0e-4)
    np.testing.assert_allclose(
        implicit['temperature_K'], reference['temperature_K'], rtol=2.0e-4
    )


def test_coupled_implicit_enforces_energy_and_fraction_bounds():
    for temperature, xhi, recombination, collisional in (
        (1.0e6, 0.0, False, True),
        (1.0e4, 1.0, True, False),
    ):
        state = _implicit_hydrogen_state(
            temperature=temperature,
            xhi=xhi,
            recombination=recombination,
            collisional=collisional,
            atomic_cooling=False,
        )
        state['alpha_B_cm3_s'] = np.array([1.0e-12])
        state['beta_cm3_s'] = np.array([1.0e-12])
        assert _coupled_implicit_source_update(
            state, 1.0e15, tolerance=1.0e-7, max_iterations=64
        )
        assert np.all(np.isfinite(state['specific_energy_erg_g']))
        assert np.all(state['specific_energy_erg_g'] > 0.0)
        assert np.all(np.isfinite(state['temperature_K']))
        assert np.all(state['temperature_K'] > 0.0)
        assert np.all(state['xHI'] >= 0.0)
        assert np.all(state['xHI'] <= 1.0)


def test_coupled_implicit_reaches_fixed_field_equilibrium():
    state = _implicit_hydrogen_state(
        temperature=1.0e4,
        xhi=0.99,
        recombination=True,
        collisional=False,
        atomic_cooling=False,
    )
    state['rho_g_cm3'][:] = PROTON_MASS_CGS
    state['alpha_B_cm3_s'] = np.array([1.0e-12])
    state['sigma_gamma_cm2'] = 1.0e-18
    ngamma = np.array([1.0e-12 / (SPEED_OF_LIGHT_CGS * 1.0e-18)])
    for _ in range(40):
        assert _coupled_implicit_source_update(
            state,
            1.0e12,
            ngamma=ngamma,
            tolerance=1.0e-8,
            max_iterations=32,
        )
    chemistry = ionization_fraction_rate(state, ngamma)
    assert abs(chemistry[0]) < 1.0e-15
    assert 0.0 < state['xHI'][0] < 1.0


def test_coupled_implicit_fallback_to_explicit():
    _, par, fluid, mesh, _ = _source_test_problem(
        solver='coupled_implicit', fallback='explicit'
    )
    par.hydrogen_implicit_max_iterations = 0
    result = apply_thermochemistry_fast(1.0, mesh, fluid, par)
    assert result['source_steps'] > 1
    assert np.isfinite(fluid.temp[0])


def test_coupled_implicit_error_fallback_raises():
    _, par, fluid, mesh, _ = _source_test_problem(
        solver='coupled_implicit', fallback='error'
    )
    par.hydrogen_implicit_max_iterations = 0
    with pytest.raises(RuntimeError, match='did not converge'):
        apply_thermochemistry_fast(1.0, mesh, fluid, par)


def test_coupled_implicit_uses_converged_half_step_pair():
    _, implicit_par, implicit_fluid, implicit_mesh, _ = _source_test_problem(
        solver='coupled_implicit', fallback='error'
    )
    implicit_par.hydrogen_implicit_max_refinements = 12
    implicit_result = apply_thermochemistry_fast(
        1.0, implicit_mesh, implicit_fluid, implicit_par
    )
    _, explicit_par, explicit_fluid, explicit_mesh, _ = _source_test_problem(
        solver='explicit'
    )
    explicit_result = apply_thermochemistry_fast(
        1.0, explicit_mesh, explicit_fluid, explicit_par
    )
    assert implicit_result['source_steps'] >= 2
    assert explicit_result['source_steps'] > 1


def test_trust_region_source_result_preserves_selected_solver():
    _, par, fluid, mesh, _ = _source_test_problem(
        solver='trust_region', fallback='error'
    )
    par.hydrogen_implicit_max_refinements = 12
    result = apply_thermochemistry_fast(1.0, mesh, fluid, par)
    assert result['source_solver'] == 'trust_region'


def test_coupled_implicit_supercomoving_matches_physical_source_update():
    _, physical_par, physical_fluid, physical_mesh, scale_factor = (
        _source_test_problem(supercomoving=False)
    )
    _, supercomoving_par, supercomoving_fluid, supercomoving_mesh, _ = (
        _source_test_problem(supercomoving=True)
    )
    apply_thermochemistry_fast(
        1.0e-4, physical_mesh, physical_fluid, physical_par
    )
    apply_thermochemistry_fast(
        1.0e-4 / scale_factor**2,
        supercomoving_mesh,
        supercomoving_fluid,
        supercomoving_par,
    )
    physical_temperature = physical_fluid.temp[0]
    supercomoving_temperature = supercomoving_fluid.temp[0] / scale_factor**2
    np.testing.assert_allclose(
        supercomoving_temperature, physical_temperature, rtol=1.0e-10
    )
    np.testing.assert_allclose(
        supercomoving_fluid.xHI, physical_fluid.xHI, rtol=1.0e-10
    )


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
    par = parameter_namespace(
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
    assert result['source_steps'] >= 2
    assert fluid.xHI[0] > xhi
    assert np.isfinite(fluid.temp[0])


def test_split_implicit_source_includes_compton_and_atomic_cooling():
    _, par, fluid, mesh, _ = _source_test_problem(solver='split_implicit')
    par.hydrogen_atomic_cooling = True
    par.compton_cmb_enabled = True
    par.compton_cmb_redshift = 10.0

    result = apply_thermochemistry_fast(1.0e-4, mesh, fluid, par)

    assert result['source_solver'] == 'split_implicit'
    assert result['source_steps'] >= 1
    assert np.isfinite(fluid.temp[0])
    assert np.isfinite(fluid.xHI[0])


def test_split_implicit_matches_coupled_solvers_for_identical_source_state():
    results = {}
    for solver in ('split_implicit', 'coupled_implicit', 'trust_region'):
        _, par, fluid, mesh, _ = _source_test_problem(
            solver=solver, fallback='error'
        )
        par.hydrogen_atomic_cooling = True
        par.hydrogen_collisional_ionization = True
        par.compton_cmb_enabled = True
        par.compton_cmb_redshift = 10.0
        par.hydrogen_update_mu = True
        fluid.SetHydrogenMu = lambda hydrogen_mass_fraction=1.0: None
        fluid.SetPressure = lambda: None
        result = apply_thermochemistry_fast(1.0e-4, mesh, fluid, par)
        results[solver] = (
            float(fluid.temp[0]),
            float(fluid.xHI[0]),
            float(fluid.Energy[0]),
            result,
        )

    split = np.asarray(results['split_implicit'][:3])
    for solver in ('coupled_implicit', 'trust_region'):
        reference = np.asarray(results[solver][:3])
        np.testing.assert_allclose(split, reference, rtol=1.0e-6, atol=1.0e-10)
        assert results[solver][3]['source_solver'] in (
            'coupled_implicit', 'trust_region'
        )
    assert results['split_implicit'][3]['source_solver'] == 'split_implicit'


def test_hybrid_source_uses_coupled_implicit_for_small_change():
    _, par, fluid, mesh, _ = _source_test_problem(solver='hybrid')
    par.hydrogen_hybrid_change_tolerance = 0.1

    result = apply_thermochemistry_fast(1.0e-8, mesh, fluid, par)

    assert result['source_solver'] == 'coupled_implicit'


def test_hybrid_source_uses_implicit_for_large_explicit_change():
    _, par, fluid, mesh, _ = _source_test_problem(solver='hybrid')
    par.hydrogen_hybrid_change_tolerance = 0.0
    par.hydrogen_implicit_fallback = 'error'

    result = apply_thermochemistry_fast(1.0e-4, mesh, fluid, par)

    assert result['source_solver'] == 'coupled_implicit'
    assert result['relative_change'] > 0.0


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

    par = parameter_namespace(
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
