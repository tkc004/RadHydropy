"""Adiabatic gas collapse from the z=100 LCDM correlation-function IC."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import unyt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))

import plot_entropy_evolution as entropy_plotter
import plot_halo_energy_accounting as energy_plotter
import virial_shock_tools as et
from cosmological_gas_correlation_runtime import CosmologicalRunCallbacks
from cosmological_gas_correlation_support import (
    _dark_matter_energy_state,
    _energy_audit_state,
    _energy_cell_state,
    _instantaneous_source_diagnostics,
    _pad_energy_history,
    _pad_profile_history,
    plot_baryon_fraction_evolution,
    plot_baryon_normalized_density_comparison,
    plot_dark_matter_density_evolution,
    plot_density_evolution,
    plot_mass_history,
    plot_radius_history,
    plot_specific_angular_momentum_evolution,
    plot_temperature_density_evolution,
    plot_temperature_evolution,
    plot_velocity_evolution,
)
from diagnostics import CosmologicalVirialShockDiagnostics
from example_utils import load_nested_example_config
from physics import CosmologicalVirialShockPhysics

from radhydropy.constants import PROTON_MASS_CGS
from radhydropy.cosmology import EinsteinDeSitter, LambdaCDM
from radhydropy.rsim.core import Rsim
from radhydropy.thermo_networks.pie import MetalPIETable
from radhydropy.units import CodeUnits, quantity_to_value

DEFAULT_CONFIG = Path(__file__).with_name(
    "cosmological_gas_correlation_z100.yaml",
)


def load_correlation_table(config_filename, config):
    filename = Path(config["example"]["linear_correlation_table_filename"])
    if not filename.is_absolute():
        filename = Path(config_filename).resolve().parent / filename
    return et.load_lcdm_correlation_table(filename)


def run(
    config_filename=DEFAULT_CONFIG,
    final_time_override=None,
    output_suffix=None,
    riemann_solver=None,
    dual_energy_entropy_limiter=None,
    dual_energy=None,
    cfl=None,
):
    config_filename = Path(config_filename).resolve()
    config = load_nested_example_config(config_filename)

    initial_condition = config["initial_condition"]
    example = config["example"]
    simulation = config["par"]["simulation"]
    hydro = config["par"].setdefault("hydrodynamics", {})
    cosmology_config = config["par"]["cosmology"]
    output = config["par"]["output"]
    thermo = config["par"].setdefault("thermochemistry", {})
    # These are plot/source-driver settings consumed by this workflow, not
    # Rsim runtime parameters.  Keep them out of the object passed to Rsim.
    configured_minimum_temperature = example.get("minimum_temperature")
    configured_temperature_plot_ymin = example.get("temperature_plot_ymin")
    # This workflow always produces energy-balance plots and per-cell energy
    # histories, so make the required diagnostics the example default.
    config["par"].setdefault("energy_diagnostics", True)
    if riemann_solver is not None:
        hydro["riemann_solver"] = riemann_solver
    if dual_energy_entropy_limiter is not None:
        hydro["dual_energy_entropy_limiter"] = bool(
            dual_energy_entropy_limiter,
        )
    if dual_energy is not None:
        hydro["dual_energy"] = bool(dual_energy)
    if cfl is not None:
        hydro["CFL"] = float(cfl)
    if thermo.get("compton_only", False):
        thermo.update(
            {
                "hydrogen_recombination": False,
                "hydrogen_collisional_ionization": False,
                "hydrogen_atomic_cooling": False,
                "compton_cmb_enabled": True,
            }
        )
    units = CodeUnits.from_mapping(config["par"]["units"]["CodeUnits"])
    if cosmology_config.get("cosmology_type") == "lambda_cdm":
        cosmology = LambdaCDM.from_code_units(
            units,
            t_ref=quantity_to_value(cosmology_config["cosmology_t_ref"], units.time_unit),
            a_ref=float(cosmology_config["cosmology_a_ref"]),
            omega_m=float(cosmology_config["cosmology_omega_m"]),
            omega_lambda=float(cosmology_config["cosmology_omega_lambda"]),
            hubble_ref=float(cosmology_config["cosmology_hubble_ref"]),
        )
    else:
        cosmology = EinsteinDeSitter.from_code_units(
            units,
            t_ref=quantity_to_value(cosmology_config["cosmology_t_ref"], units.time_unit),
            a_ref=float(cosmology_config["cosmology_a_ref"]),
        )
    correlation_table = load_correlation_table(config_filename, config)
    config["_code_unit_system"] = units
    config["_cosmology"] = cosmology
    config["_correlation_table"] = correlation_table
    output_dir = Path(output["directory"])
    figure_prefix = str(
        example.get("figure_prefix", "CosmologicalGasCorrelationZ100"),
    )
    if output_suffix:
        output_dir = output_dir.with_name(output_dir.name + str(output_suffix))
        figure_prefix += str(output_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    ic_filename = output_dir / "InitialCondition.hdf5"
    config["par"]["output"]["directory"] = str(output_dir)
    config["par"]["simulation"]["initial_condition_filename"] = str(ic_filename)
    if thermo.get("metal_pie_table_filename"):
        table_filename = Path(thermo["metal_pie_table_filename"])
        if not table_filename.is_absolute():
            table_filename = config_filename.parent / table_filename
        config["par"]["metal_pie_table"] = MetalPIETable(table_filename)

    initial_writer = et.build_initial_condition(config)
    initial = initial_writer.simulation
    if bool(hydro.get("gas_angular_momentum", False)):
        initial.par.gas_angular_momentum = True
        initial.fluid.specific_angular_momentum_code = np.full(
            initial.par.mesh.grid_cells,
            float(hydro.get("gas_specific_angular_momentum", 0.0)),
        )
    config["_dark_matter_softening"] = config["par"]["dark_matter"]["softening"]
    dm = et.make_dark_matter(config)
    # Serialize the live shell state with the IC so RunAll can restore both
    # the mutable solver object and its typed analysis view.
    initial.par.dark_matter = dm
    initial_writer.write(ic_filename)

    baryon_fraction = float(initial_condition["baryon_fraction"])
    gas_mass_comoving_code = float(
        np.sum(initial.fluid.rho_comoving_code * initial.mesh.volume_comoving_code)
    )
    dm_mass_comoving_code = float(np.sum(dm.mass))
    measured_fraction = gas_mass_comoving_code / max(
        gas_mass_comoving_code + dm_mass_comoving_code, 1.0e-30
    )
    if not np.isclose(measured_fraction, baryon_fraction, rtol=0.02):
        raise RuntimeError(
            "initial gas/total mass fraction does not match baryon_fraction",
        )
    initial_time = quantity_to_value(initial_condition["time_cosmic"], units.time_unit)
    temperature_proper_cgs_K = (
        float(np.median(initial.fluid.temp_supercomoving_code))
        / float(
            cosmology.scale_factor(initial_time),
        )
        ** 2
    )
    cmb_temperature_0_cgs_K = float(
        initial_condition["cmb_temperature_0"].to_value(unyt.K),
    )
    expected_temperature = cmb_temperature_0_cgs_K * (
        1.0 / float(cosmology.scale_factor(initial_time))
    )
    if not np.isclose(temperature_proper_cgs_K, expected_temperature, rtol=1.0e-8):
        raise RuntimeError("initial gas temperature is not the z=100 CMB temperature")

    initial_a = float(cosmology.scale_factor(initial_time))
    minimum_temperature = configured_minimum_temperature
    if minimum_temperature is not None:
        if hasattr(minimum_temperature, "to_value"):
            minimum_temperature = float(minimum_temperature.to_value("K"))
        else:
            minimum_temperature = float(minimum_temperature)

    transition_redshift = example.get(
        "thermochemistry_transition_redshift",
        thermo.get("thermochemistry_transition_redshift"),
    )
    transition_tau = None
    if transition_redshift is not None:
        transition_redshift = float(transition_redshift)
        transition_scale_factor = 1.0 / (1.0 + transition_redshift)
        transition_time = float(
            cosmology.cosmic_time_from_scale_factor(transition_scale_factor),
        )
        transition_tau = float(cosmology.supercomoving_time(transition_time))

    final_time = (
        float(final_time_override)
        if final_time_override is not None
        else quantity_to_value(simulation["final_time"], units.time_unit)
    )
    target_tau = float(cosmology.supercomoving_time(final_time))
    config["par"]["simulation"]["final_time"] = target_tau
    runner = Rsim(config["par"])
    physics = CosmologicalVirialShockPhysics(
        runner,
        cosmology,
        initial_condition,
        baryon_fraction,
        initial_a,
        cmb_temperature_0_cgs_K,
        temperature_proper_cgs_K,
        minimum_temperature=minimum_temperature,
        transition_redshift=transition_redshift,
    )
    diagnostics = CosmologicalVirialShockDiagnostics(
        config,
        initial_condition,
        runner,
        dm,
        units,
        cosmology,
        _instantaneous_source_diagnostics,
        _energy_cell_state,
        _dark_matter_energy_state,
        _energy_audit_state,
        {
            "output_dir": output_dir,
            "figure_prefix": figure_prefix,
            "example": example,
            "hydro": hydro,
            "units": units,
            "initial_condition": initial_condition,
            "baryon_fraction": baryon_fraction,
            "configured_temperature_plot_ymin": configured_temperature_plot_ymin,
            "minimum_temperature": minimum_temperature,
            "measured_fraction": measured_fraction,
            "plot_mass_history": plot_mass_history,
            "plot_radius_history": plot_radius_history,
            "plot_temperature_evolution": plot_temperature_evolution,
            "plot_specific_angular_momentum_evolution": plot_specific_angular_momentum_evolution,
            "plot_density_evolution": plot_density_evolution,
            "plot_temperature_density_evolution": plot_temperature_density_evolution,
            "plot_velocity_evolution": plot_velocity_evolution,
            "plot_baryon_fraction_evolution": plot_baryon_fraction_evolution,
            "plot_dark_matter_density_evolution": plot_dark_matter_density_evolution,
            "plot_baryon_normalized_density_comparison": plot_baryon_normalized_density_comparison,
            "entropy_plotter": entropy_plotter,
            "energy_plotter": energy_plotter,
            "PROTON_MASS_CGS": PROTON_MASS_CGS,
            "quantity_to_value": quantity_to_value,
            "_pad_energy_history": _pad_energy_history,
            "_pad_profile_history": _pad_profile_history,
        },
    )
    callbacks = CosmologicalRunCallbacks(
        config_filename,
        thermo,
        hydro,
        cosmology,
        baryon_fraction,
        physics,
        diagnostics,
        initial_time,
        initial_a,
    )
    runner.RunAll(
        mode="hydro_sources",
        before_step_callback=callbacks.before_step,
        history_callback=callbacks.history,
        snapshot_callback=callbacks.snapshot,
        stop_condition=lambda current_sim: False,
    )
    return diagnostics.finalize(runner)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--final-time",
        type=float,
        default=None,
        help=(
            "override final cosmic time in simulation code units for a short "
            "debug run; output is reported in Gyr"
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="append a suffix to the output directory and figure prefix",
    )
    parser.add_argument(
        "--riemann-solver",
        choices=("Rusanov", "HLLC"),
        default=None,
        help="override the configured Riemann solver",
    )
    parser.add_argument(
        "--disable-dual-energy-entropy-limiter",
        action="store_true",
        help="disable the experimental dual-energy entropy limiter",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=None,
        help="override the configured CFL number for this run",
    )
    args = parser.parse_args()
    run(
        args.config,
        final_time_override=args.final_time,
        output_suffix=args.output_suffix,
        riemann_solver=args.riemann_solver,
        dual_energy_entropy_limiter=(False if args.disable_dual_energy_entropy_limiter else None),
        cfl=args.cfl,
    )
