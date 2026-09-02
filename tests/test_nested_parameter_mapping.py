"""Regression tests for nested YAML settings reaching the runtime parameter object."""

import pytest
import unyt
import numpy as np

from radhydropy.params import Par
from radhydropy.rsim import Rsim


CODE_UNITS = {
    "name": "nested_mapping_test_units",
    "InternalUnitSystem": {
        "UnitMass_in_cgs": 1.0,
        "UnitLength_in_cgs": 1.0,
        "UnitVelocity_in_cgs": 1.0,
        "UnitCurrent_in_cgs": 1.0,
        "UnitTemp_in_cgs": 1.0,
    },
}


def test_nested_runtime_settings_are_copied_to_par():
    par = Par({
        "simulation": {"initial_time": 1.25},
        "hydrodynamics": {"hydro_integrator": "ssprk2"},
        "timestep": {
            "chemistry_timestep": 0.2,
            "evolution_timestep": 0.3,
            "output_interval": 0.4,
            "crossing_safety_factor": 0.5,
            "supercomoving_timestep": 0.6,
        },
        "output": {"time_interval": 0.7},
        "radiation": {
            "radiation_pressure": True,
            "radiation_pressure_efficiency": 0.8,
            "radiation_pressure_source_luminosity": 9.0,
            "radiative_transfer_c2ray_ode_max_iterations": 19,
            "radiative_transfer_c2ray_ode_tolerance": 2.0e-7,
        },
        "thermochemistry": {
            "absolute_tolerance": 2.0e-11,
            "relative_tolerance": 2.0e-4,
            "explicit_tolerance": 0.2,
            "hydrogen_initial_collisional_equilibrium": True,
            "hydrogen_photon_energy": 21.0,
            "pie_uvbg_implicit_max_iterations": 17,
            "pie_uvbg_implicit_max_retries": 3,
            "pie_uvbg_implicit_step_doubling": False,
            "pie_uvbg_implicit_tolerance": 4.0e-4,
        },
        "units": {"CodeUnits": CODE_UNITS},
    })

    assert par.initial_time == pytest.approx(1.25)
    assert par.hydro_integrator == "ssprk2"
    assert par.chemistry_timestep == pytest.approx(0.2)
    assert par.evolution_timestep == pytest.approx(0.3)
    assert par.output_interval == pytest.approx(0.4)
    assert par.crossing_safety_factor == pytest.approx(0.5)
    assert par.supercomoving_timestep == pytest.approx(0.6)
    assert par.time_interval == pytest.approx(0.7)
    assert par.radiation_pressure is True
    assert par.radiation_pressure_efficiency == pytest.approx(0.8)
    assert par.radiation_pressure_source_luminosity == pytest.approx(9.0)
    assert par.radiative_transfer_c2ray_ode_max_iterations == 19
    assert par.radiative_transfer_c2ray_ode_tolerance == pytest.approx(2.0e-7)
    assert par.absolute_tolerance == pytest.approx(2.0e-11)
    assert par.relative_tolerance == pytest.approx(2.0e-4)
    assert par.explicit_tolerance == pytest.approx(0.2)
    assert par.hydrogen_initial_collisional_equilibrium is True
    assert par.hydrogen_photon_energy == pytest.approx(21.0)
    assert par.pie_uvbg_implicit_max_iterations == 17
    assert par.pie_uvbg_implicit_max_retries == 3
    assert par.pie_uvbg_implicit_step_doubling is False
    assert par.pie_uvbg_implicit_tolerance == pytest.approx(4.0e-4)


def test_nested_group_aliases_are_not_dropped():
    par = Par({
        "chemistry": {
            "hydrogen_chemistry": True,
            "hydrogen_recombination": False,
            "hydrogen_collisional_ionization": False,
            "hydrogen_thermal_coupling": False,
            "hydrogen_update_mu": True,
            "hydrogen_xHI_initial": 0.25,
            "hydrogen_ngamma_initial": 3.0,
            "hydrogen_source_CFL": 11.0,
        },
        "thermochemistry": {
            "hydrogen_radiation_field": True,
            "hydrogen_radiation_evolution": False,
        },
        "units": {"CodeUnits": CODE_UNITS},
    })

    assert par.hydrogen_chemistry is True
    assert par.hydrogen_recombination is False
    assert par.hydrogen_collisional_ionization is False
    assert par.hydrogen_thermal_coupling is False
    assert par.hydrogen_update_mu is True
    assert par.hydrogen_xHI_initial == pytest.approx(0.25)
    assert par.hydrogen_ngamma_initial == pytest.approx(3.0)
    assert par.hydrogen_source_CFL == pytest.approx(11.0)
    assert par.hydrogen_radiation_field is True
    assert par.hydrogen_radiation_evolution is False


def test_nested_unitful_settings_are_converted_to_code_units():
    config = {
        "simulation": {
            "final_time": 2.0e13 * unyt.s,
            "current_time": 1.0e13 * unyt.s,
            "box_size": 3.0e18 * unyt.cm,
        },
        "mesh": {"grid_cells": 4, "ghost_cells": 2, "area": 2.0e36 * unyt.cm**2},
        "hydrodynamics": {
            "positivity_density_floor": 4.0e-21 * unyt.g / unyt.cm**3,
        },
        "boundary": {
            "inflow_density": 3.0e-21 * unyt.g / unyt.cm**3,
            "inflow_velocity": 2.0e5 * unyt.cm / unyt.s,
            "inflow_temperature": 400.0 * unyt.K,
            "outflow_density": 4.0e-21 * unyt.g / unyt.cm**3,
            "outflow_velocity": 3.0e5 * unyt.cm / unyt.s,
            "outflow_temperature": 500.0 * unyt.K,
        },
        "timestep": {
            "dtmin": 1.0e13 * unyt.s,
            "dtmax": 2.0e13 * unyt.s,
            "hydrogen_source_dtmin": 3.0e13 * unyt.s,
        },
        "output": {"cadence": 4.0e13 * unyt.s},
        "gravity": {
            "cosmology_t_ref": 5.0e13 * unyt.s,
            "cosmology_hubble_ref": 2.0e-13 / unyt.s,
            "gas_core_radius": 6.0e18 * unyt.cm,
        },
        "radiation": {
            "radiative_transfer_boundary_flux":
                5.0 / (unyt.cm**2 * unyt.s),
            "radiative_transfer_source_photon_rate": 7.0e-13 / unyt.s,
            "radiation_spectrum_total_photon_rate": 8.0e-13 / unyt.s,
            "radiative_transfer_boundary_flux_groups":
                np.array([1.0e-13, 2.0e-13]) / (unyt.cm**2 * unyt.s),
            "radiative_transfer_source_photon_rate_groups":
                np.array([3.0e-13, 4.0e-13]) / unyt.s,
            "hydrogen_ngamma_initial": 9.0e-18 / unyt.cm**3,
            "hydrogen_ngamma_inflow": 1.0e-17 / unyt.cm**3,
            "hydrogen_ngamma_outflow": 1.1e-17 / unyt.cm**3,
            "radiation_group_sigma_gamma":
                np.array([4.0e-18, 5.0e-18]) * unyt.cm**2,
            "radiation_group_epsilon_gamma":
                np.array([6.0, 7.0]) * unyt.eV,
            "hydrogen_sigma_gamma": 4.0e-18 * unyt.cm**2,
            "hydrogen_epsilon_gamma": 8.0 * unyt.eV,
        },
        "thermochemistry": {
            "hydrogen_alpha_B": 2.0e-13 * unyt.cm**3 / unyt.s,
            "hydrogen_beta": 3.0e-13 * unyt.cm**3 / unyt.s,
            "hydrogen_implicit_absolute_temperature_tolerance": 10.0 * unyt.K,
            "cmb_temperature_0": 2.0 * unyt.K,
        },
        "units": {
            "CodeUnits": {
                "InternalUnitSystem": {
                    "UnitMass_in_cgs": 1.0e33,
                    "UnitLength_in_cgs": 1.0e18,
                    "UnitVelocity_in_cgs": 1.0e5,
                    "UnitCurrent_in_cgs": 1.0,
                    "UnitTemp_in_cgs": 1.0,
                },
            },
        },
    }

    sim = Rsim(config)
    sim.ConvertParametersToCodeUnits()
    assert sim.par.simulation.final_time == pytest.approx(2.0)
    assert sim.par.simulation.current_time == pytest.approx(1.0)
    assert sim.par.simulation.box_size == pytest.approx(3.0)
    assert sim.par.area == pytest.approx(2.0)
    assert sim.par.boundary.inflow_density == pytest.approx(3.0)
    assert sim.par.boundary.inflow_velocity == pytest.approx(2.0)
    assert sim.par.boundary.inflow_temperature == pytest.approx(400.0)
    assert sim.par.boundary.outflow_density == pytest.approx(4.0)
    assert sim.par.boundary.outflow_velocity == pytest.approx(3.0)
    assert sim.par.boundary.outflow_temperature == pytest.approx(500.0)
    assert sim.par.positivity_density_floor == pytest.approx(4.0)
    assert sim.par.timestep.dtmin == pytest.approx(1.0)
    assert sim.par.timestep.dtmax == pytest.approx(2.0)
    assert sim.par.timestep.hydrogen_source_dtmin == pytest.approx(3.0)
    assert sim.par.output.cadence == pytest.approx(4.0)
    assert sim.par.cosmology_t_ref == pytest.approx(5.0)
    assert sim.par.cosmology_hubble_ref == pytest.approx(2.0)
    assert sim.par.gas_core_radius == pytest.approx(6.0)
    assert sim.par.radiation.boundary_flux == pytest.approx(5.0e49)
    assert sim.par.radiation.source_photon_rate == pytest.approx(7.0)
    assert sim.par.radiation.spectrum_total_photon_rate == pytest.approx(8.0)
    assert np.allclose(sim.par.radiation.boundary_flux_groups, [1.0e36, 2.0e36])
    assert np.allclose(sim.par.radiation.source_photon_rate_groups, [3.0, 4.0])
    assert sim.par.radiation.hydrogen_ngamma_initial == pytest.approx(9.0e36)
    assert sim.par.radiation.hydrogen_ngamma_inflow == pytest.approx(1.0e37)
    assert sim.par.radiation.hydrogen_ngamma_outflow == pytest.approx(1.1e37)
    assert np.allclose(sim.par.radiation.group_sigma_gamma, [4.0e-54, 5.0e-54])
    assert np.allclose(
        sim.par.radiation.group_epsilon_gamma,
        (np.array([6.0, 7.0]) * unyt.eV).to_value(unyt.erg) / 1.0e43,
    )
    assert sim.par.radiation.hydrogen_sigma_gamma == pytest.approx(4.0e-54)
    assert sim.par.radiation.hydrogen_epsilon_gamma == pytest.approx(
        (8.0 * unyt.eV).to_value(unyt.erg) / 1.0e43
    )
    assert sim.par.hydrogen_alpha_B == pytest.approx(2.0e-54)
    assert sim.par.hydrogen_beta == pytest.approx(3.0e-54)
    assert sim.par.hydrogen_implicit_absolute_temperature_tolerance == pytest.approx(10.0)
    assert sim.par.cmb_temperature_0 == pytest.approx(2.0)
