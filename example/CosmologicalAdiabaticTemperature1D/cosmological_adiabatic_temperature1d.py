"""Compare homogeneous adiabatic temperature evolution with cosmology tools."""

from pathlib import Path
import copy
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = PROJECT_ROOT / "tools"
DENSITY_EXAMPLE_ROOT = PROJECT_ROOT / "example" / "CosmologicalDensityEvolution1D"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "example"))
sys.path.insert(0, str(DENSITY_EXAMPLE_ROOT))

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter as CodeEdS
from radhydropy.cosmology import LambdaCDM as CodeLambdaCDM
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
from cosmology import EinsteinDeSitter as PhysicalEdS
from cosmology import LambdaCDM as PhysicalLambdaCDM
from cosmological_density_evolution1d import (
    CODE_TIME_S,
    SECONDS_PER_GYR,
    make_initial_condition,
    make_units,
    unit_mapping,
)
import example_utils as eu


OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
CONFIG_FILE = Path(__file__).with_name("cosmological_adiabatic_temperature1d.yaml")


def run():
    config = eu.load_nested_example_config(CONFIG_FILE)
    base_runtime = config["par"]
    example = config["example"]
    units = CodeUnits.from_mapping(base_runtime["units"]["CodeUnits"])
    time_unit_gyr = CODE_TIME_S / SECONDS_PER_GYR
    initial_scale_factor = float(example["initial_scale_factor"])
    final_scale_factor = float(example["final_scale_factor"])
    initial_temperature = float(example["initial_temperature"])
    cases = [
        ("EdS", PhysicalEdS(h0=70.0), CodeEdS),
        ("LCDM_0p3_0p7", PhysicalLambdaCDM(h0=70.0, omega_m=0.3, omega_lambda=0.7), CodeLambdaCDM),
        ("LCDM_0p7_0p3", PhysicalLambdaCDM(h0=70.0, omega_m=0.7, omega_lambda=0.3), CodeLambdaCDM),
    ]
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
        initial = make_initial_condition(
            units, code_cosmology, initial_time, 1.0,
            initial_scale_factor,
        )
        # For gamma=5/3, T_tilde = T*a^2.  The stored temperature is therefore
        # constant for homogeneous adiabatic expansion.
        initial.fluid.temp[:] = initial_temperature * initial_scale_factor**2
        output_dir = OUTPUT_ROOT / label
        output_dir.mkdir(parents=True, exist_ok=True)
        ic_filename = output_dir / "InitialCondition.hdf5"
        rio.writehdf5(initial, ic_filename)
        runparams = copy.deepcopy(base_runtime)
        runparams["simulation"].update(
            name=f"CosmologicalAdiabaticTemperature1D_{label}",
            initial_condition_filename=str(ic_filename),
            final_time=final_tau * units.time_unit,
        )
        runparams["output"].update(directory=str(output_dir), savedir=str(output_dir))
        runparams["gravity"].update(
            cosmology_type=cosmology_type,
            cosmology_t_ref=physical.age_0 / time_unit_gyr,
            cosmology_a_ref=1.0,
            **cosmology_parameters,
        )
        runparams["output"]["cadence"] = (final_tau - initial_tau) * units.time_unit
        sim = Rsim(runparams)
        sim.Callreadhdf5()
        sim.SetMesh()
        sim.SetFluid()
        sim.fluid.SetFluidTime(sim.par.time)
        sim.SetInitFluid()
        sim.par.cosmology = code_cosmology
        sim.Run(outputtime=0)
        final_tau_sim = float(np.asarray(sim.fluid.time, dtype=float).flat[0])
        _, final_a, _ = code_cosmology.background_state_from_supercomoving(final_tau_sim)
        stored_temperature = float(np.mean(sim.fluid.temp))
        measured_temperature = stored_temperature / final_a**2
        expected_temperature = initial_temperature * (initial_scale_factor / final_scale_factor) ** 2
        print(
            f"{label}: a={final_a:.12g}, T_stored={stored_temperature:.12g}, "
            f"T_physical={measured_temperature:.12g} K, "
            f"analytic={expected_temperature:.12g} K, "
            f"relative_error={(measured_temperature - expected_temperature) / expected_temperature:.6e}"
        )
        if not np.isclose(final_a, final_scale_factor, rtol=2.0e-8):
            raise RuntimeError(f"{label}: scale factor disagrees")
        if not np.isclose(measured_temperature, expected_temperature, rtol=2.0e-8):
            raise RuntimeError(f"{label}: adiabatic temperature disagrees")
    return OUTPUT_ROOT


if __name__ == "__main__":
    run()
