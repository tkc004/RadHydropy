"""Test homogeneous Hubble flow against the standalone cosmology tool."""

from pathlib import Path
import copy
import sys

import numpy as np
import unyt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = PROJECT_ROOT / "example" / "CosmologicalDensityEvolution1D"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT / "example"))
sys.path.insert(0, str(EXAMPLE_ROOT))

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter as CodeEdS
from radhydropy.cosmology import LambdaCDM as CodeLambdaCDM
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
from cosmology import EinsteinDeSitter as PhysicalEdS
from cosmology import LambdaCDM as PhysicalLambdaCDM
from cosmological_density_evolution1d import (
    CODE_TIME_S,
    CODE_LENGTH_CM,
    CODE_VELOCITY_CM_S,
    SECONDS_PER_GYR,
    density_msun_mpc3_to_cgs,
    make_units,
    unit_mapping,
)
from cosmological_initial_condition import build_initial_condition
import example_utils as eu


OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
CONFIG_FILE = Path(__file__).with_name("cosmological_hubble_flow1d.yaml")


def code_values(value, unit):
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(unit), dtype=float)
    raw = np.asarray(value, dtype=float)
    unit_value = float(unit)
    if unit_value != 1.0 and np.max(np.abs(raw), initial=0.0) > 1.0e6:
        raw = raw / unit_value
    return raw


def run():
    config = eu.load_nested_example_config(CONFIG_FILE)

    example = config["example"]
    units = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    time_unit_gyr = CODE_TIME_S / SECONDS_PER_GYR
    initial_scale_factor = float(example["initial_scale_factor"])
    final_scale_factor = float(example["final_scale_factor"])
    density_unit = float(units.density_unit.to_value("g/cm**3"))
    cases = [
        ("EdS", PhysicalEdS(h0=70.0), CodeEdS),
        ("LCDM_0p3_0p7", PhysicalLambdaCDM(h0=70.0, omega_m=0.3, omega_lambda=0.7), CodeLambdaCDM),
        ("LCDM_0p7_0p3", PhysicalLambdaCDM(h0=70.0, omega_m=0.7, omega_lambda=0.3), CodeLambdaCDM),
    ]
    histories = []

    for label, physical, code_class in cases:
        initial_time_gyr = float(physical.cosmic_time_from_scale_factor(initial_scale_factor))
        final_time_gyr = float(physical.cosmic_time_from_scale_factor(final_scale_factor))
        initial_time = initial_time_gyr / time_unit_gyr
        final_time = final_time_gyr / time_unit_gyr
        if code_class is CodeEdS:
            code_cosmology = code_class.from_code_units(
                units, t_ref=physical.age_0 / time_unit_gyr, a_ref=1.0
            )
            cosmology_type = "einstein_de_sitter"
            cosmology_parameters = {}
        else:
            code_cosmology = code_class.from_code_units(
                units, t_ref=physical.age_0 / time_unit_gyr, a_ref=1.0,
                omega_m=physical.omega_m, omega_lambda=physical.omega_lambda,
                hubble_ref=physical.hubble_0_gyr * time_unit_gyr,
            )
            cosmology_type = "lambda_cdm"
            cosmology_parameters = {
                "cosmology_omega_m": physical.omega_m,
                "cosmology_omega_lambda": physical.omega_lambda,
                "cosmology_hubble_ref": physical.hubble_0_gyr * time_unit_gyr,
            }
        initial_tau = float(code_cosmology.supercomoving_time(initial_time))
        final_tau = float(code_cosmology.supercomoving_time(final_time))
        rho_proper_code = density_msun_mpc3_to_cgs(
            physical.critical_density(initial_time_gyr)
        ) / density_unit
        case_config = copy.deepcopy(config)
        case_config["_code_cosmology"] = code_cosmology
        case_config["initial_condition"] = {
            "box_size_comoving": 4.0 * units.length_unit,
            "time_cosmic": initial_time * units.time_unit,
        }
        case_config["_rho_comoving_code"] = np.full(
            int(config["par"]["mesh"]["grid_cells"]),
            rho_proper_code * initial_scale_factor**3,
        )
        case_config["_temp_supercomoving_code"] = np.ones(
            int(config["par"]["mesh"]["grid_cells"])
        )
        case_config["_vel_supercomoving_code"] = np.zeros(
            int(config["par"]["mesh"]["grid_cells"])
        )
        initial = build_initial_condition(case_config)
        output_dir = OUTPUT_ROOT / label
        output_dir.mkdir(parents=True, exist_ok=True)
        ic_filename = output_dir / "InitialCondition.hdf5"
        rio.writehdf5(initial, ic_filename)
        case_config["par"]["simulation"].update(
            name=f"CosmologicalHubbleFlow1D_{label}",
            initial_condition_filename=str(ic_filename),
            final_time=final_tau * units.time_unit,
        )
        case_config["par"]["output"].update(directory=str(output_dir))
        case_config["par"]["gravity"].update(
            cosmology_type=cosmology_type,
            cosmology_t_ref=physical.age_0 / time_unit_gyr,
            cosmology_a_ref=1.0,
            **cosmology_parameters,
        )
        case_config["par"]["output"]["cadence"] = (final_tau - initial_tau) * units.time_unit
        sim = Rsim(case_config["par"])
        rio.readhdf5(sim.par, sim.mesh, sim.fluid, sim.par.simulation.initial_condition_filename)
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
        sim.par.cosmology = code_cosmology
        sim.Run(outputtime=0)

        final_tau_sim = float(
            np.asarray(sim.fluid.tau_supercomoving_code, dtype=float).flat[0]
        )
        time_cosmic_code, final_a, final_hubble = code_cosmology.background_state_from_supercomoving(final_tau_sim)
        first = int(sim.par.mesh.ghost_cells)
        last = first + int(sim.par.mesh.grid_cells)
        x_comoving_code = sim.mesh.x_comoving_code[first:last]
        vel_supercomoving_code = sim.fluid.vel_supercomoving_code[first:last]
        positions = code_values(x_comoving_code, units.length_unit)
        peculiar = code_values(vel_supercomoving_code, units.velocity_unit)
        proper_velocity_code = final_hubble * final_a * positions + peculiar / final_a
        expected_hubble_code = (
            physical.hubble(final_time_gyr)
            / 3.0856775814913673e19
            * CODE_TIME_S
        )
        expected_velocity_code = expected_hubble_code * final_scale_factor * positions
        velocity_error = float(np.max(np.abs(proper_velocity_code - expected_velocity_code)))
        peculiar_error = float(np.max(np.abs(peculiar)))
        proper_velocity_kms = proper_velocity_code * CODE_VELOCITY_CM_S / 1.0e5
        expected_velocity_kms = expected_velocity_code * CODE_VELOCITY_CM_S / 1.0e5
        radius_proper_kpc = positions * CODE_LENGTH_CM / float((1.0 * unyt.kpc).to_value('cm'))
        histories.append({
            "label": label,
            "physical": physical,
            "radius_proper_kpc": radius_proper_kpc,
            "velocity_proper_km_s": proper_velocity_kms,
            "expected_velocity_proper_km_s": expected_velocity_kms,
        })
        print(
            f"{label}: a={final_a:.12g}, H={final_hubble:.12g}, "
            f"max|v_peculiar|={peculiar_error:.6e}, "
            f"max|u-u_Hubble|={velocity_error:.6e}"
        )
        if not np.isclose(final_a, final_scale_factor, rtol=2.0e-8):
            raise RuntimeError(f"{label}: scale factor disagrees")
        if peculiar_error > 2.0e-12:
            raise RuntimeError(f"{label}: homogeneous peculiar velocity changed")
        if velocity_error > 2.0e-8:
            raise RuntimeError(f"{label}: Hubble velocity disagrees")
    figure = OUTPUT_ROOT / "CosmologicalHubbleFlow1D.jpg"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, (flow_axis, error_axis) = plt.subplots(2, 1, figsize=(8.0, 7.0))
    for history in histories:
        physical = history["physical"]
        scale_factors = np.linspace(initial_scale_factor, final_scale_factor, 200)
        times_gyr = np.asarray([
            physical.cosmic_time_from_scale_factor(float(scale_factor))
            for scale_factor in scale_factors
        ])
        hubble = np.asarray([physical.hubble(float(time_gyr)) for time_gyr in times_gyr])
        radius_proper_mpc = history["radius_proper_kpc"][[-1]][0] / 1000.0
        velocity_proper_km_s = hubble * scale_factors * radius_proper_mpc
        flow_axis.plot(scale_factors, velocity_proper_km_s, label=history["label"])
        relative_error = np.abs(
            (history["velocity_proper_km_s"] - history["expected_velocity_proper_km_s"])
            / np.maximum(np.abs(history["expected_velocity_proper_km_s"]), 1.0e-30)
        )
        error_axis.plot(history["radius_proper_kpc"], relative_error, marker="o", label=history["label"])
    flow_axis.set_ylabel("proper Hubble velocity [km/s]")
    flow_axis.set_title("Homogeneous Hubble-flow evolution")
    flow_axis.grid(alpha=0.25)
    flow_axis.legend(frameon=False)
    error_axis.set_xlabel("radius [kpc]")
    error_axis.set_ylabel("relative velocity error")
    error_axis.set_yscale("log")
    error_axis.grid(alpha=0.25, which="both")
    error_axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"figure = {figure}")
    return OUTPUT_ROOT


if __name__ == "__main__":
    run()
