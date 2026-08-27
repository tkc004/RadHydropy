"""Test homogeneous Hubble flow against the standalone cosmology tool."""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = PROJECT_ROOT / "example" / "CosmologicalDensityEvolution1D"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(EXAMPLE_ROOT))

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter as CodeEdS
from radhydropy.cosmology import LambdaCDM as CodeLambdaCDM
from radhydropy.rsim import Rsim
from cosmology import EinsteinDeSitter as PhysicalEdS
from cosmology import LambdaCDM as PhysicalLambdaCDM
from cosmological_density_evolution1d import (
    CODE_TIME_S,
    SECONDS_PER_GYR,
    density_msun_mpc3_to_cgs,
    make_initial_condition,
    make_units,
    unit_mapping,
)


OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def code_values(value, unit):
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(unit), dtype=float)
    raw = np.asarray(value, dtype=float)
    unit_value = float(unit)
    if unit_value != 1.0 and np.max(np.abs(raw), initial=0.0) > 1.0e6:
        raw = raw / unit_value
    return raw


def run():
    units = make_units()
    time_unit_gyr = CODE_TIME_S / SECONDS_PER_GYR
    initial_scale_factor = 1.0 / 101.0
    final_scale_factor = 1.0 / 2.0
    density_unit = float(units.density_unit.to_value("g/cm**3"))
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
        initial_density = density_msun_mpc3_to_cgs(
            physical.critical_density(initial_time_gyr)
        ) / density_unit
        initial = make_initial_condition(
            units, code_cosmology, initial_time, initial_density,
            initial_scale_factor,
        )
        output_dir = OUTPUT_ROOT / label
        output_dir.mkdir(parents=True, exist_ok=True)
        ic_filename = output_dir / "InitialCondition.hdf5"
        rio.writehdf5(initial, ic_filename)
        runparams = {
            "simname": f"CosmologicalHubbleFlow1D_{label}",
            "ICfilename": str(ic_filename),
            "outdir": str(output_dir),
            "savedir": str(output_dir),
            "outfileprefix": "Output",
            "coordsys": "cartesian",
            "nogrid": 4,
            "EOStype": "polytropic",
            "gamma": 5.0 / 3.0,
            "cosmological_expansion": True,
            "supercomoving_coordinates": True,
            "cosmological_gravity": False,
            "selfgravity": False,
            "externalgravity": False,
            "cosmology_type": cosmology_type,
            "cosmology_t_ref": physical.age_0 / time_unit_gyr,
            "cosmology_a_ref": 1.0,
            **cosmology_parameters,
            "timesim": final_tau * units.time_unit,
            "outdeltatime": (final_tau - initial_tau) * units.time_unit,
            "dtmin": 1.0e-8 * units.time_unit,
            "dtmax": 1.0 * units.time_unit,
            "CFL": 0.1,
            "boundcond": "Periodic",
            "order": 1,
            "CodeUnits": unit_mapping(),
        }
        sim = Rsim(runparams)
        sim.Callreadhdf5()
        sim.SetMesh()
        sim.SetFluid()
        sim.fluid.SetFluidTime(sim.par.time)
        sim.SetInitFluid()
        sim.Run(outputtime=0)

        final_tau_sim = float(np.asarray(sim.fluid.time, dtype=float).flat[0])
        cosmic_time, final_a, final_hubble = code_cosmology.background_state_from_supercomoving(final_tau_sim)
        coordinate = sim.mesh.coordinate[sim.par.noghost:sim.par.noghost + sim.par.nogrid]
        velocity = sim.fluid.vel[sim.par.noghost:sim.par.noghost + sim.par.nogrid]
        positions = code_values(coordinate, units.length_unit)
        peculiar = code_values(velocity, units.velocity_unit)
        proper_velocity_code = final_hubble * final_a * positions + peculiar / final_a
        expected_hubble_code = (
            physical.hubble(final_time_gyr)
            / 3.0856775814913673e19
            * CODE_TIME_S
        )
        expected_velocity_code = expected_hubble_code * final_scale_factor * positions
        velocity_error = float(np.max(np.abs(proper_velocity_code - expected_velocity_code)))
        peculiar_error = float(np.max(np.abs(peculiar)))
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
    return OUTPUT_ROOT


if __name__ == "__main__":
    run()
