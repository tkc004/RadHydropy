import numpy as np
import pytest
import h5py
import tempfile
import unyt
from pathlib import Path
from types import SimpleNamespace
from tests.parameter_fixtures import parameter_namespace

from radhydropy.cosmology import EinsteinDeSitter, LambdaCDM
from radhydropy.cosmological_variables import (
    physical_density,
    physical_temperature,
    physical_velocity,
    supercomoving_scale,
    to_supercomoving_density,
    to_supercomoving_temperature,
    to_supercomoving_velocity,
)
from radhydropy.units import CodeUnits
from radhydropy.params import Par
from radhydropy.solver import Solver
import radhydropy.io as rio


def code_units():
    return CodeUnits.from_mapping({
        "UnitMass_in_cgs": 1.0e33,
        "UnitLength_in_cgs": 1.0e18,
        "UnitVelocity_in_cgs": 1.0e5,
        "UnitCurrent_in_cgs": 1.0,
        "UnitTemp_in_cgs": 1.0,
    })


def test_einstein_de_sitter_background_relations():
    cosmology = EinsteinDeSitter.from_code_units(code_units())
    assert np.isclose(cosmology.scale_factor(8.0), 4.0)
    assert np.isclose(cosmology.hubble(2.0), 1.0 / 3.0)
    assert np.isclose(
        cosmology.background_density(2.0),
        1.0 / (6.0 * np.pi * cosmology.gravitational_constant * 4.0),
    )


def test_einstein_de_sitter_rejects_zero_time():
    with pytest.raises(ValueError):
        EinsteinDeSitter().scale_factor(0.0)


def test_lambda_cdm_reference_normalization_and_round_trip():
    cosmology = LambdaCDM.from_code_units(
        code_units(), t_ref=2.0, a_ref=1.5, omega_m=0.3, omega_lambda=0.7,
    )
    assert np.isclose(cosmology.scale_factor(2.0), 1.5)
    assert np.isclose(
        cosmology.background_density(2.0),
        3.0 * cosmology._hubble_ref**2 * 0.3
        / (8.0 * np.pi * cosmology.gravitational_constant),
    )
    for time in (0.5, 2.0, 8.0):
        tau = cosmology.supercomoving_time(time)
        assert np.isclose(cosmology.cosmic_time_from_supercomoving(tau), time)
    assert np.isclose(cosmology.cosmic_time_from_scale_factor(1.5), 2.0)


def test_supercomoving_time_round_trip():
    cosmology = EinsteinDeSitter()
    for time in (1.0, 2.0, 8.0):
        tau = cosmology.supercomoving_time(time)
        assert np.isclose(cosmology.cosmic_time_from_supercomoving(tau), time)


def test_supercomoving_physical_conversions():
    cosmology = EinsteinDeSitter()
    tau = cosmology.supercomoving_time(2.0)
    assert np.isclose(cosmology.physical_density(4.0, tau), 1.0)
    assert np.isclose(
        cosmology.physical_pressure(2.0 ** (10.0 / 3.0), tau, 5.0 / 3.0),
        1.0,
    )


def test_supercomoving_variable_round_trip():
    cosmology = EinsteinDeSitter()
    tau = cosmology.supercomoving_time(2.0)
    class Par:
        time = tau
    par = Par()
    par.cosmology = cosmology
    a, hubble = supercomoving_scale(par)
    gamma = 5.0 / 3.0
    radius = np.array([2.0])
    physical_rho = np.array([4.0])
    physical_temp = np.array([10.0])
    physical_vel = np.array([hubble * a * radius[0] + 3.0 / a])
    rho = to_supercomoving_density(physical_rho, a)
    temp = to_supercomoving_temperature(physical_temp, a, gamma)
    velocity = to_supercomoving_velocity(physical_vel, radius, a, hubble)
    assert np.allclose(physical_density(rho, a), physical_rho)
    assert np.allclose(physical_temperature(temp, a, gamma), physical_temp)
    assert np.allclose(physical_velocity(velocity, radius, a, hubble), physical_vel)


def test_supercomoving_specific_angular_momentum_is_scale_factor_invariant():
    cosmology = EinsteinDeSitter()
    tau = cosmology.supercomoving_time(2.0)
    class Par:
        time = tau
    par = Par()
    par.cosmology = cosmology
    a, _ = supercomoving_scale(par)

    comoving_radius = np.array([2.0, 4.0])
    physical_tangential_velocity = np.array([3.0, -1.5])
    physical_radius = a * comoving_radius
    supercomoving_tangential_velocity = a * physical_tangential_velocity

    physical_j = physical_radius * physical_tangential_velocity
    supercomoving_j = comoving_radius * supercomoving_tangential_velocity
    assert np.allclose(supercomoving_j, physical_j)


def test_supercomoving_rotational_energy_density_scales_as_a5():
    cosmology = EinsteinDeSitter()
    tau = cosmology.supercomoving_time(2.0)

    class Par:
        time = tau

    par = Par()
    par.cosmology = cosmology
    a, _ = supercomoving_scale(par)
    x = np.array([1.5, 3.0])
    physical_radius = a * x
    physical_density_value = np.array([2.0, 5.0])
    physical_tangential_velocity = np.array([0.75, -1.25])
    rho_sc = to_supercomoving_density(physical_density_value, a)
    j = physical_radius * physical_tangential_velocity

    mesh = SimpleNamespace(coordsys='spherical', coordinate=x)
    fluid = SimpleNamespace(rho_code=rho_sc, specific_angular_momentum_code=j)
    options = SimpleNamespace(gas_rotational_energy=True, gas_angular_momentum=True)
    energy_sc = Solver()._rotational_energy_density(mesh, fluid, options)
    energy_phys = 0.5 * physical_density_value * physical_tangential_velocity**2

    # Comoving density and radius make this a^5 times the physical density.
    assert np.allclose(energy_sc, a**5 * energy_phys)


def test_supercomoving_centrifugal_source_has_expected_scale_factor():
    cosmology = EinsteinDeSitter()
    tau = cosmology.supercomoving_time(2.0)

    class Par:
        time = tau
        nogrid = 1
        noghost = 0
        mesh = SimpleNamespace(ghost_cells=0, grid_cells=1)
        gas_rotational_energy = True
        gas_angular_momentum = True
        energy_diagnostics = True

    par = Par()
    par.cosmology = cosmology
    a, _ = supercomoving_scale(par)
    x = 2.0
    physical_radius = a * x
    physical_tangential_velocity = 1.25
    j = physical_radius * physical_tangential_velocity
    mass = 3.0
    dt = 0.125
    mesh = SimpleNamespace(
        coordsys='spherical', coordinate=np.array([x]), vol=np.array([1.0]),
    )
    fluid = SimpleNamespace(
        rho_code=np.array([mass]), Mass_code=np.array([mass]), Mom_code=np.array([0.0]),
        Energy_code=np.array([7.0]), AngularMomentum_code=np.array([mass * j]),
    )

    solver = Solver()
    solver.ApplyGravity(dt, mesh, fluid, par)
    supercomoving_acceleration = j**2 / x**3
    physical_acceleration = j**2 / physical_radius**3

    assert fluid.Mom_code[0] == pytest.approx(mass * supercomoving_acceleration * dt)
    assert supercomoving_acceleration == pytest.approx(a**3 * physical_acceleration)
    # Rotational energy is already in total energy, so source work is a
    # diagnostic rather than a second direct energy update.
    assert fluid.Energy_code[0] == pytest.approx(7.0)
    assert solver.last_centrifugal_work_by_cell[0] != 0.0


def test_cosmological_angular_momentum_evolution_and_restart():
    units = code_units()
    cosmology = EinsteinDeSitter.from_code_units(units)
    cosmic_times = (1.0, 2.0, 8.0)
    x = np.array([1.0, 2.0])
    j = np.array([0.6, -0.35])
    physical_density_at_a1 = np.array([2.0, 3.0])
    physical_tangential_velocity_at_a1 = j / x

    rotational_energy_ratios = []
    centrifugal_accelerations = []
    for cosmic_time in cosmic_times:
        scale_factor = cosmology.scale_factor(cosmic_time)
        physical_density = physical_density_at_a1 / scale_factor**3
        physical_tangential_velocity = j / (scale_factor * x)
        rho_sc = to_supercomoving_density(physical_density, scale_factor)
        j_from_velocity = x * (
            scale_factor * physical_tangential_velocity
        )
        energy_sc = 0.5 * rho_sc * (j / x)**2
        energy_phys = 0.5 * physical_density * physical_tangential_velocity**2

        np.testing.assert_allclose(j_from_velocity, j)
        rotational_energy_ratios.append(energy_sc / energy_phys)
        centrifugal_accelerations.append(j**2 / x**3)

    for cosmic_time, ratio in zip(cosmic_times, rotational_energy_ratios):
        assert np.allclose(
            ratio,
            cosmology.scale_factor(cosmic_time)**5,
        )
    np.testing.assert_allclose(
        centrifugal_accelerations[0],
        centrifugal_accelerations[1],
    )
    np.testing.assert_allclose(
        centrifugal_accelerations[1],
        centrifugal_accelerations[2],
    )

    tau_initial = cosmology.supercomoving_time(cosmic_times[0])
    tau_restart = cosmology.supercomoving_time(cosmic_times[1])
    scale_initial = cosmology.scale_factor(cosmic_times[0])
    rho_sc = to_supercomoving_density(
        physical_density_at_a1 / scale_initial**3,
        scale_initial,
    )
    volume = (
        4.0 * np.pi / 3.0
        * ((x + 0.5)**3 - (x - 0.5)**3)
    ) * units.volume_unit
    specific_quantity = j * units.length_unit**2 / units.time_unit
    radius_quantity = x * units.length_unit
    rho_quantity = rho_sc * units.density_unit
    angular_quantity = rho_quantity * specific_quantity * volume
    rotational_specific_energy = (
        0.5 * (specific_quantity / radius_quantity)**2
    )
    par = parameter_namespace(
        coordsys='spherical', nogrid=2, noghost=0,
        CodeUnits=units, time=tau_initial,
        boxsize=3.0 * units.length_unit,
        cosmological_expansion=True, supercomoving_coordinates=True,
        cosmology=cosmology, cosmology_type='einstein_de_sitter',
        cosmology_t_ref=1.0, cosmology_a_ref=1.0,
        coordinate_frame='comoving', time_coordinate='supercomoving',
        velocity_representation='supercomoving_peculiar',
        density_representation='comoving', pressure_representation='supercomoving',
        temperature_representation='supercomoving', gamma=5.0 / 3.0,
        gas_angular_momentum=True, gas_rotational_energy=True,
    )
    mesh = SimpleNamespace(
        boundary=np.array([0.5, 1.5, 2.5]) * units.length_unit,
    )
    fluid = SimpleNamespace(
        rho_code=rho_quantity,
        vel_code=np.zeros(2) * units.velocity_unit,
        temp_code=np.ones(2) * units.temperature_unit,
        mu=np.ones(2),
        specific_angular_momentum_code=specific_quantity,
        Mass_code=rho_quantity * volume,
        AngularMomentum_code=angular_quantity,
        Energy_code=rho_quantity * rotational_specific_energy * volume,
    )
    sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)

    with tempfile.TemporaryDirectory() as directory:
        filename = Path(directory) / 'angular_momentum_restart.hdf5'
        rio.writehdf5(sim, filename)
        loaded_par = parameter_namespace()
        loaded_mesh = SimpleNamespace()
        loaded_fluid = SimpleNamespace()
        rio.readhdf5(loaded_par, loaded_mesh, loaded_fluid, filename)

        np.testing.assert_allclose(
            loaded_fluid.specific_angular_momentum_code, j
        )
        np.testing.assert_allclose(
            loaded_fluid.AngularMomentum_code,
            np.asarray(angular_quantity.to_value(
                units.mass_unit * units.length_unit**2 / units.time_unit
            )),
        )
        assert loaded_par.time == pytest.approx(tau_initial)

        # Continue from the reloaded state at a later supercomoving time.
        loaded_par.time = tau_restart
        loaded_fluid.time_code = tau_restart
        restart_scale = cosmology.scale_factor(cosmic_times[1])
        restart_rho_code = np.asarray(loaded_fluid.rho_code, dtype=float)
        restart_j = np.asarray(loaded_fluid.specific_angular_momentum_code, dtype=float)
        restart_energy = 0.5 * restart_rho_code * (restart_j / x)**2
        physical_restart_energy = (
            0.5 * (physical_density_at_a1 / restart_scale**3)
            * (j / (restart_scale * x))**2
        )
        np.testing.assert_allclose(restart_j, j)
        np.testing.assert_allclose(
            restart_energy / physical_restart_energy,
            restart_scale**5,
        )
        np.testing.assert_allclose(restart_j**2 / x**3, j**2 / x**3)


def test_cosmology_header_round_trip_and_supercomoving_input_output():
    units = code_units()
    cosmology = EinsteinDeSitter.from_code_units(units)
    tau = cosmology.supercomoving_time(2.0)
    par = parameter_namespace(
        coordsys='cartesian', nogrid=2, noghost=0,
        CodeUnits=units, time=tau, boxsize=2.0,
        cosmological_expansion=True, supercomoving_coordinates=True,
        cosmology=cosmology, cosmology_type='einstein_de_sitter',
        cosmology_t_ref=1.0, cosmology_a_ref=1.0,
        coordinate_frame='comoving', time_coordinate='supercomoving',
        velocity_representation='supercomoving_peculiar',
        density_representation='comoving',
        pressure_representation='supercomoving',
        temperature_representation='supercomoving', gamma=5.0 / 3.0,
    )
    mesh = SimpleNamespace(boundary=np.array([0.0, 1.0, 2.0]))
    fluid = SimpleNamespace(
        rho_code=np.ones(2) * 4.0, vel_code=np.ones(2) * 2.0,
        temp_code=np.ones(2) * 3.0, mu=np.ones(2), time=tau,
    )
    sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
    with tempfile.TemporaryDirectory() as directory:
        filename = Path(directory) / 'supercomoving.hdf5'
        rio.writehdf5(sim, filename)
        with h5py.File(filename, 'r') as handle:
            header = handle['Header']
            assert header.attrs['CosmologyType'] == 'einstein_de_sitter'
            assert header.attrs['TimeCoordinate'] == 'supercomoving'
            assert header.attrs['ScaleFactor'] == pytest.approx(2.0 ** (2.0 / 3.0))
        loaded = parameter_namespace()
        rio.readhdf5(loaded, SimpleNamespace(), SimpleNamespace(), filename)
        assert loaded.cosmological_expansion
        assert loaded.supercomoving_coordinates
        assert loaded.cosmology.type_name == 'einstein_de_sitter'


def test_lambda_cdm_header_round_trip():
    units = code_units()
    cosmology = LambdaCDM.from_code_units(
        units, t_ref=2.0, a_ref=1.0, omega_m=0.3, omega_lambda=0.7,
        hubble_ref=0.4,
    )
    tau = cosmology.supercomoving_time(2.0)
    par = parameter_namespace(
        coordsys='cartesian', nogrid=1, noghost=0,
        CodeUnits=units, time=tau, boxsize=1.0,
        cosmological_expansion=True, supercomoving_coordinates=True,
        cosmology=cosmology, cosmology_type='lambda_cdm',
        cosmology_t_ref=2.0, cosmology_a_ref=1.0,
        coordinate_frame='comoving', time_coordinate='supercomoving',
        velocity_representation='supercomoving_peculiar',
        density_representation='comoving', pressure_representation='supercomoving',
        temperature_representation='supercomoving', gamma=5.0 / 3.0,
    )
    mesh = SimpleNamespace(boundary=np.array([0.0, 1.0]))
    fluid = SimpleNamespace(rho_code=np.ones(1), vel_code=np.zeros(1), temp_code=np.ones(1), mu=np.ones(1), time=tau)
    sim = SimpleNamespace(par=par, mesh=mesh, fluid=fluid)
    with tempfile.TemporaryDirectory() as directory:
        filename = Path(directory) / 'lambda_cdm.hdf5'
        rio.writehdf5(sim, filename)
        loaded = parameter_namespace()
        rio.readhdf5(loaded, SimpleNamespace(), SimpleNamespace(), filename)
        assert loaded.cosmology.type_name == 'lambda_cdm'
        assert loaded.cosmology.omega_m == pytest.approx(0.3)
        assert loaded.cosmology.omega_lambda == pytest.approx(0.7)
        assert loaded.cosmology._hubble_ref == pytest.approx(0.4)


def test_par_constructs_lambda_cdm_from_parameters():
    units = {
        "name": "test",
        "InternalUnitSystem": {
            "UnitMass_in_cgs": 1.0e33,
            "UnitLength_in_cgs": 1.0e18,
            "UnitVelocity_in_cgs": 1.0e5,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        },
    }
    par = Par({
        "CodeUnits": units,
        "cosmological_expansion": True,
        "cosmology_type": "lambda_cdm",
        "cosmology_omega_m": 0.3,
        "cosmology_omega_lambda": 0.7,
        "cosmology_hubble_ref": 0.4,
    })
    assert par.cosmology.type_name == "lambda_cdm"
    assert par.cosmology._hubble_ref == pytest.approx(0.4)
    assert par.cosmology.type == "lambda_cdm"
    assert par.cosmology.model.type_name == "lambda_cdm"
    assert par.units.CodeUnits is not None
    assert par.units.unit_system is par.unit_system
    assert par.hydrodynamics.eos_type == "polytropic"
    assert par.hydrodynamics.gamma == pytest.approx(1.4)
    assert par.hydrodynamics.temperature == par.temperature
    assert par.hydrodynamics.dual_energy is False
    assert par.boundary.condition == "Periodic"
    assert par.boundary.inflow_density == 1.0 * unyt.g / unyt.cm**3
    assert par.timestep.dtmin == 2.0e-8 * unyt.s
    assert par.timestep.dtmax == 2.0e-1 * unyt.s
    assert par.timestep.cooling_safety_factor == pytest.approx(0.1)
    assert par.timestep.relaxation_damping_time == par.relaxation_damping_time
    assert par.thermochemistry.network == "hydrogen"
    assert par.thermochemistry.cie_cooling is False
    assert par.thermochemistry.metallicity == pytest.approx(1.0)
    assert par.thermochemistry.hydrogen_atomic_cooling is True
    assert par.gravity.selfgravity is False
    assert par.gravity.model is None
    assert par.gravity.potential_energy is False
    assert par.output.directory == par.outdir
    assert par.output.savedir == par.savedir
    assert par.output.filename_prefix == par.outfileprefix
    assert par.simulation.name == par.simname
    assert par.simulation.coordinate_system == "cartesian"
    assert par.simulation.final_time == 2.0 * unyt.s
    assert par.diagnostics.verbose == par.verbose
    assert par.diagnostics.energy_diagnostics is False
    assert par.mesh.ghost_cells == 2
    assert par.mesh.area == par.area
    assert par.chemistry.key == "H"
    assert par.chemistry.hydrogen_mass_fraction == pytest.approx(1.0)
    assert par.chemistry.hydrogen_xHI_initial == pytest.approx(1.0)
    assert par.chemistry.helium_coupled_implicit is True
    assert par.chemistry.implicit_max_iterations == 32
    assert par.chemistry.implicit_fallback == "explicit"
    assert par.chemistry.alpha_B is None
    assert par.chemistry.beta is None
    assert par.angular_momentum.enabled is False
    assert par.angular_momentum.flux_scheme == "fct"
    assert par.angular_momentum.inflow == par.specific_angular_momentum_inflow
    assert par.dark_matter_config.crossing_safety_factor == pytest.approx(0.1)
    assert par.dark_matter_config.global_timestep_limit is True
    assert par.radiation.radiative_transfer_method == "long_characteristics"
    assert par.radiation.radiation_pressure_efficiency == pytest.approx(1.0)
    assert par.radiation.c2ray_max_iterations == 32
    assert par.radiation.c2ray_nonconvergence == "warn"
    assert par.radiation.compton_cmb_enabled is False
    assert par.radiation.hydrogen_radiation_evolution is True
    assert par.radiation.hydrogen_ngamma_initial == par.hydrogen_ngamma_initial
    assert par.dual_energy_config.enabled is False
    assert par.dual_energy_config.pressure_selection == "switch"
    assert par.positivity.enabled is True
    assert par.positivity.density_floor == pytest.approx(0.0)


def test_par_gravity_assignment_updates_nested_model():
    par = Par({"CodeUnits": code_units()})
    model = SimpleNamespace(acceleration_on_mesh=lambda *args: None)
    par.gravity.model = model
    assert par.gravity.model is model
    assert par.gravity.acceleration_on_mesh is model.acceleration_on_mesh


def test_par_rejects_unknown_run_parameters():
    with pytest.raises(ValueError, match="unknown run parameter.*typo"):
        Par({"CodeUnits": code_units(), "typo": True})


def test_par_warns_when_verbose_defaults_are_used():
    with pytest.warns(UserWarning, match="run parameter 'gamma'.*default"):
        Par({"CodeUnits": code_units(), "verbose": 1})
