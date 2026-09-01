"""Compare uniform-gas density evolution with the standalone cosmology tool."""

from pathlib import Path
import copy
import sys

import numpy as np
import unyt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = PROJECT_ROOT / "tools"
EXAMPLE_ROOT = PROJECT_ROOT / "example"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

import radhydropy.io as rio
from radhydropy.cosmology import EinsteinDeSitter as CodeEdS
from radhydropy.cosmology import LambdaCDM as CodeLambdaCDM
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu
from cosmology import EinsteinDeSitter as PhysicalEdS
from cosmology import LambdaCDM as PhysicalLambdaCDM


OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
CONFIG_FILE = Path(__file__).with_name("cosmological_density_evolution1d.yaml")
CODE_LENGTH_CM = 1.0e24
CODE_VELOCITY_CM_S = 1.0e7
CODE_TIME_S = CODE_LENGTH_CM / CODE_VELOCITY_CM_S
SECONDS_PER_GYR = 365.25 * 24.0 * 3600.0 * 1.0e9


def density_msun_mpc3_to_cgs(density):
    return float(density) * float((1.0 * unyt.Msun).to_value("g")) / float(
        (1.0 * unyt.Mpc).to_value("cm")
    ) ** 3


def make_units():
    return CodeUnits.from_mapping(unit_mapping())


def unit_mapping():
    return {
        "name": "cosmological_density_evolution_unit_system",
        "InternalUnitSystem": {
            "UnitMass_in_cgs": 1.0e33,
            "UnitLength_in_cgs": CODE_LENGTH_CM,
            "UnitVelocity_in_cgs": CODE_VELOCITY_CM_S,
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        },
    }


def make_initial_condition(
    units, code_cosmology, initial_time, density_code, initial_scale_factor
):
    class State:
        pass

    state = State()
    state.par = State()
    state.mesh = State()
    state.fluid = State()
    count = 4
    boxsize = 4.0
    state.par.CodeUnits = units
    state.par.units = State()
    state.par.units.CodeUnits = units
    state.par.unit_system = units.unit_system
    state.par.nogrid = count
    state.par.coordsys = "cartesian"
    state.par.boxsize = np.asarray([boxsize])
    initial_tau = float(code_cosmology.supercomoving_time(initial_time))
    state.par.time = np.asarray([initial_tau])
    state.par.simulation = State()
    state.par.simulation.current_time = state.par.time
    state.par.simulation.box_size = state.par.boxsize
    state.par.simulation.coordinate_system = "cartesian"
    state.par.mesh = State()
    state.par.mesh.grid_cells = count
    state.par.mesh.ghost_cells = 0
    state.par.hydrodynamics = State()
    state.par.hydrodynamics.gamma = 5.0 / 3.0
    state.par.cosmological_expansion = True
    state.par.supercomoving_coordinates = True
    state.par.cosmological_gravity = False
    state.par.selfgravity = False
    state.par.externalgravity = False
    state.par.cosmology = code_cosmology
    state.par.cosmology_type = code_cosmology.type_name
    state.par.cosmology_t_ref = code_cosmology.t_ref
    state.par.cosmology_a_ref = code_cosmology.a_ref
    state.par.coordinate_frame = "comoving"
    state.par.time_coordinate = "supercomoving"
    state.par.velocity_representation = "supercomoving_peculiar"
    state.par.density_representation = "comoving"
    state.par.pressure_representation = "supercomoving"
    state.par.temperature_representation = "supercomoving"

    boundary = np.linspace(0.0, boxsize, count + 1)
    state.mesh.boundary = boundary
    state.mesh.coordinate = 0.5 * (boundary[1:] + boundary[:-1])
    state.mesh.xdelta = np.full(count, boxsize / count)
    state.mesh.area = np.ones(count)
    state.mesh.vol = np.full(count, boxsize / count)
    state.fluid.rho = np.full(count, density_code * initial_scale_factor**3)
    state.fluid.vel = np.zeros(count)
    state.fluid.temp = np.full(count, 1.0)
    state.fluid.mu = np.ones(count)
    state.fluid.time = np.asarray([initial_tau])
    return state


def run():
    config = eu.load_nested_example_config(CONFIG_FILE)
    base_runtime = config["par"]
    example = config["example"]
    units = CodeUnits.from_mapping(base_runtime["units"]["CodeUnits"])
    time_unit_gyr = CODE_TIME_S / SECONDS_PER_GYR
    initial_scale_factor = float(example["initial_scale_factor"])
    final_scale_factor = float(example["final_scale_factor"])
    cases = [
        ("EdS", PhysicalEdS(h0=70.0), CodeEdS),
        ("LCDM_0p3_0p7", PhysicalLambdaCDM(h0=70.0, omega_m=0.3, omega_lambda=0.7), CodeLambdaCDM),
        ("LCDM_0p7_0p3", PhysicalLambdaCDM(h0=70.0, omega_m=0.7, omega_lambda=0.3), CodeLambdaCDM),
    ]
    density_unit = float(units.density_unit.to_value("g/cm**3"))
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
        initial_density_cgs = density_msun_mpc3_to_cgs(
            physical.critical_density(initial_time_gyr)
        )
        density_code = initial_density_cgs / density_unit
        initial = make_initial_condition(
            units, code_cosmology, initial_time, density_code, initial_scale_factor
        )
        output_dir = OUTPUT_ROOT / label
        output_dir.mkdir(parents=True, exist_ok=True)
        ic_filename = output_dir / "InitialCondition.hdf5"
        rio.writehdf5(initial, ic_filename)
        runparams = copy.deepcopy(base_runtime)
        runparams["simulation"].update(
            name=f"CosmologicalDensityEvolution1D_{label}",
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
        runparams["output"]["cadence"] = (final_tau - initial_tau) / 2.0 * units.time_unit
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
        measured_density = float(np.mean(sim.fluid.rho)) / final_a**3
        expected_density = density_code * (initial_scale_factor / final_scale_factor) ** 3
        expected_critical = density_msun_mpc3_to_cgs(
            physical.critical_density(final_time_gyr)
        ) / density_unit
        relative_error = (measured_density - expected_density) / expected_density
        print(f"{label}: a={final_a:.12g}, gas={measured_density:.12g}, "
              f"analytic={expected_density:.12g}, critical(z=1)={expected_critical:.12g}, "
              f"relative_error={relative_error:.6e}")
        if not np.isclose(final_a, final_scale_factor, rtol=2.0e-8):
            raise RuntimeError(f"{label}: scale factors disagree")
        if not np.isclose(measured_density, expected_density, rtol=2.0e-8):
            raise RuntimeError(f"{label}: gas density does not follow analytic a^-3 evolution")
    return OUTPUT_ROOT


if __name__ == "__main__":
    run()
