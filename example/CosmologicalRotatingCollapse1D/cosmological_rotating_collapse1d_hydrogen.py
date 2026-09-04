"""Verify hydrogen cooling preserves rotational energy in a rotating gas cell.

This is a source-only companion to the rotating-collapse example.  Mass,
angular momentum, radius, and radial velocity are held fixed; only the
thermal chemistry source is applied.
"""

from pathlib import Path
import copy
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "example"))

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu

from cosmological_rotating_collapse1d import InitialCondition, DEFAULT_CONFIG


def main(output_root=None):
    config = eu.load_nested_example_config(DEFAULT_CONFIG)
    runtime = config["par"]
    icparams = config["initial_condition"]
    runtime = copy.deepcopy(runtime)
    runtime["thermochemistry"] = {
        **runtime.get("thermochemistry", {}),
        'hydrogen_chemistry': True, 'hydrogen_thermal_coupling': True,
        'hydrogen_update_mu': False, 'hydrogen_recombination': True,
        'hydrogen_collisional_ionization': True, 'hydrogen_atomic_cooling': True,
        'hydrogen_radiation_field': False,
        'hydrogen_source_solver': 'coupled_implicit',
        'hydrogen_implicit_fallback': 'error',
        'cooling_temperature_floor': {'value': 1.0e-3, 'unit': 'K'},
    }
    runtime["simulation"] = {**runtime["simulation"], "final_time": 1.0}
    if output_root is not None:
        runtime["output"] = {**runtime["output"], "directory": str(output_root), "savedir": str(output_root)}

    units = CodeUnits.from_mapping(runtime['units']['CodeUnits'])
    runtime["thermochemistry"]["cooling_temperature_floor"] = 1.0e-3 * units.temperature_unit
    cosmology = EinsteinDeSitter.from_code_units(
        units,
        t_ref=float(runtime['gravity']['cosmology_t_ref']),
        a_ref=float(runtime['gravity']['cosmology_a_ref']),
    )
    output_dir = ROOT / runtime['output']['directory'] / 'hydrogen_source_rotation'
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime['simulation'] = {**runtime['simulation'], 'initial_condition_filename': str(output_dir / 'InitialCondition.hdf5')}
    runtime['output'] = {**runtime['output'], 'directory': str(output_dir), 'savedir': str(output_dir), 'filename_prefix': 'Output'}

    initial = InitialCondition(
        icparams, runtime, float(icparams['high_rotation_factor']),
        units, cosmology,
    )
    rio.writehdf5(initial, runtime['simulation']['initial_condition_filename'])
    sim = Rsim(runtime)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.fluid.SetFluidTime(sim.par.time)
    sim.SetInitFluid()
    sim.par.set_cosmology_model(cosmology)

    first = int(sim.par.mesh.ghost_cells)
    last = first + int(sim.par.mesh.grid_cells)
    active = slice(first, last)
    mass_before = np.asarray(sim.fluid.Mass_code[active], dtype=float).copy()
    angular_before = np.asarray(sim.fluid.AngularMomentum_code[active], dtype=float).copy()
    energy_before = np.asarray(sim.fluid.Energy_code[active], dtype=float).copy()
    radius = np.abs(np.asarray(sim.mesh.coordinate[active], dtype=float))
    rotational_before = np.zeros_like(mass_before)
    valid = (mass_before > 0.0) & (radius > 0.0)
    rotational_before[valid] = (
        0.5 * angular_before[valid]**2
        / (mass_before[valid] * radius[valid]**2)
    )

    sim.ApplyThermochemistrySources(1.0e-3)
    sim._synchronize_thermochemistry_internal_energy()

    mass_after = np.asarray(sim.fluid.Mass_code[active], dtype=float)
    angular_after = np.asarray(sim.fluid.AngularMomentum_code[active], dtype=float)
    energy_after = np.asarray(sim.fluid.Energy_code[active], dtype=float)
    momentum_after = np.asarray(sim.fluid.Mom_code[active], dtype=float)
    rotational_after = np.zeros_like(mass_after)
    valid = (mass_after > 0.0) & (radius > 0.0)
    rotational_after[valid] = (
        0.5 * angular_after[valid]**2
        / (mass_after[valid] * radius[valid]**2)
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

    rio.writehdf5(sim, output_dir / 'Output_final.hdf5')
    print('rotating hydrogen source-energy check passed')
    print('maximum |d E_rot| = %s' % np.max(np.abs(rotational_after - rotational_before)))
    print('maximum |d E_total - d E_thermal| = %s' % np.max(np.abs(total_change - thermal_change)))
    return sim


if __name__ == '__main__':
    main()
