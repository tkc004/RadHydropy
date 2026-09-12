"""Verify hydrogen cooling preserves rotational energy in a rotating gas cell.

This is a source-only companion to the rotating-collapse example.  Mass,
angular momentum, radius, and radial velocity are held fixed; only the
thermal chemistry source is applied.
"""

from pathlib import Path
import copy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "example"))

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.units import CodeUnits, quantity_to_value
import example_utils as eu
from cosmological_initial_condition import build_initial_condition

from cosmological_rotating_collapse1d import spherical_centers, DEFAULT_CONFIG


def main(output_root=None):
    config = eu.load_nested_example_config(DEFAULT_CONFIG)
    initial_condition = config["initial_condition"]
    config["par"].setdefault("thermochemistry", {}).update({
        'hydrogen_chemistry': True, 'hydrogen_thermal_coupling': True,
        'hydrogen_update_mu': False, 'hydrogen_recombination': True,
        'hydrogen_collisional_ionization': True, 'hydrogen_atomic_cooling': True,
        'hydrogen_radiation_field': False,
        'hydrogen_source_solver': 'coupled_implicit',
        'hydrogen_implicit_fallback': 'error',
        'cooling_temperature_floor': {'value': 1.0e-3, 'unit': 'K'},
    })
    if output_root is not None:
        config["par"]["output"]["directory"] = str(output_root)

    units = CodeUnits.from_mapping(config["par"]['units']['CodeUnits'])
    config["par"]["simulation"]["final_time"] = 1.0 * units.time_unit
    config["par"]["thermochemistry"]["cooling_temperature_floor"] = 1.0e-3 * units.temperature_unit
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=quantity_to_value(config["par"]['cosmology']['cosmology_t_ref'], units.time_unit),
        a_ref=float(config["par"]['cosmology']['cosmology_a_ref']),
    )
    output_dir = ROOT / config["par"]['output']['directory'] / 'hydrogen_source_rotation'
    output_dir.mkdir(parents=True, exist_ok=True)
    config["par"]['simulation'] = {**config["par"]['simulation'], 'initial_condition_filename': str(output_dir / 'InitialCondition.hdf5')}
    config["par"]['output'] = {**config["par"]['output'], 'directory': str(output_dir), 'filename_prefix': 'Output'}

    count = int(config["par"]["mesh"]["grid_cells"])
    time_cosmic_code = quantity_to_value(initial_condition["time_cosmic"], units.time_unit)
    scale_factor = float(cosmology.scale_factor(time_cosmic_code))
    hubble = float(cosmology.hubble(time_cosmic_code))
    boundary_comoving_code = np.linspace(
        float(initial_condition["radius_inner_comoving"].to_value(units.length_unit)),
        float(initial_condition["radius_outer_comoving"].to_value(units.length_unit)),
        count + 1,
    )
    x_comoving_code = spherical_centers(boundary_comoving_code)
    volume_comoving_code = 4.0 * np.pi / 3.0 * (
        boundary_comoving_code[1:]**3 - boundary_comoving_code[:-1]**3
    )
    rho_background = float(cosmology.background_density(time_cosmic_code))
    overdensity = float(initial_condition["overdensity"])
    inside = x_comoving_code < float(
        initial_condition["radius_perturbation_comoving"].to_value(units.length_unit)
    )
    rho_comoving_code = rho_background * (1.0 + overdensity * inside) * scale_factor**3
    enclosed_mass = np.cumsum(rho_comoving_code * volume_comoving_code)
    physical_radius_code = scale_factor * x_comoving_code
    specific_angular_momentum_code = float(initial_condition["high_rotation_factor"]) * np.sqrt(
        cosmology.gravitational_constant * enclosed_mass * physical_radius_code
    )
    initial_config = {
        "par": config["par"],
        "initial_condition": {
            **initial_condition,
            "box_size_comoving": initial_condition["radius_outer_comoving"],
            "time_cosmic": time_cosmic_code * units.time_unit,
        },
        "example": {},
        "_code_cosmology": cosmology,
        "_boundary_comoving_code": boundary_comoving_code,
        "_x_comoving_code": x_comoving_code,
        "_volume_comoving_code": volume_comoving_code,
        "_rho_comoving_code": rho_comoving_code,
        "_vel_supercomoving_code": -scale_factor**2 * hubble * overdensity * x_comoving_code / 3.0,
        "_temp_supercomoving_code": np.full(
            count,
            float(initial_condition["temperature_proper"].to_value(units.temperature_unit)) * scale_factor**2,
        ),
        "_mu_dimensionless": np.full(count, float(initial_condition["mean_molecular_weight"])),
        "_specific_angular_momentum_code": specific_angular_momentum_code,
        "_initial_tau_supercomoving_code": float(
            cosmology.supercomoving_time(time_cosmic_code)
        ),
    }
    initial = build_initial_condition(initial_config)
    rio.writehdf5(initial, config["par"]['simulation']['initial_condition_filename'])
    sim = rio.loadhdf5(
        config,
        config["par"]["simulation"]["initial_condition_filename"],
    )
    sim.SetMesh()
    sim.SetFluid()
    sim.fluid.SetFluidTime(sim.par.tau_supercomoving_code)
    sim.SetInitFluid()
    initial_tau = np.asarray(sim.par.tau_supercomoving_code, dtype=float)
    sim.par.tau_supercomoving_code = initial_tau.copy()
    sim.par.simulation.tau_supercomoving_code = initial_tau.copy()
    sim.fluid.SetFluidTime(initial_tau)
    if not (
        np.allclose(sim.par.tau_supercomoving_code, initial_tau)
        and np.allclose(sim.par.simulation.tau_supercomoving_code, initial_tau)
        and np.allclose(np.asarray(sim.fluid.tau_supercomoving_code, dtype=float), initial_tau)
    ):
        raise RuntimeError("cosmological startup clocks disagree after SetInitFluid")
    sim.par.set_cosmology_model(cosmology)

    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    active = slice(first, last)
    mass_before = np.asarray(sim.fluid.Mass_code[active], dtype=float).copy()
    angular_before = np.asarray(sim.fluid.AngularMomentum_code[active], dtype=float).copy()
    energy_before = np.asarray(sim.fluid.Energy_code[active], dtype=float).copy()
    radius_comoving_code = np.abs(np.asarray(sim.mesh.x_comoving_code[active], dtype=float))
    rotational_before = np.zeros_like(mass_before)
    valid = (mass_before > 0.0) & (radius_comoving_code > 0.0)
    rotational_before[valid] = (
        0.5 * angular_before[valid]**2
        / (mass_before[valid] * radius_comoving_code[valid]**2)
    )

    sim.ApplyThermochemistrySources(1.0e-3)
    sim._synchronize_thermochemistry_internal_energy()

    mass_after = np.asarray(sim.fluid.Mass_code[active], dtype=float)
    angular_after = np.asarray(sim.fluid.AngularMomentum_code[active], dtype=float)
    energy_after = np.asarray(sim.fluid.Energy_code[active], dtype=float)
    momentum_after = np.asarray(sim.fluid.Mom_code[active], dtype=float)
    rotational_after = np.zeros_like(mass_after)
    valid = (mass_after > 0.0) & (radius_comoving_code > 0.0)
    rotational_after[valid] = (
        0.5 * angular_after[valid]**2
        / (mass_after[valid] * radius_comoving_code[valid]**2)
    )
    kinetic_before = 0.5 * np.asarray(sim.fluid.Mom_code[active], dtype=float)**2 / mass_before
    kinetic_after = 0.5 * momentum_after**2 / mass_after
    thermal_change = (energy_after - kinetic_after - rotational_after) - (
        energy_before - kinetic_before - rotational_before
    )
    total_change = energy_after - energy_before

    np.testing.assert_allclose(mass_after, mass_before, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(angular_after, angular_before, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(rotational_after, rotational_before, rtol=1.0e-12, atol=1.0e-14)
    np.testing.assert_allclose(total_change, thermal_change, rtol=1.0e-10, atol=1.0e-14)

    rio._writehdf5(sim, output_dir / 'Output_final.hdf5')
    print('rotating hydrogen source-energy check passed')
    print('maximum |d E_rot| = %s' % np.max(np.abs(rotational_after - rotational_before)))
    print('maximum |d E_total - d E_thermal| = %s' % np.max(np.abs(total_change - thermal_change)))
    return sim


if __name__ == '__main__':
    main()
