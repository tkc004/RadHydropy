"""20 pc dynamic Stromgren sphere with direct radiation pressure."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "example"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).resolve().with_name(
    "dynamic_stromgren_sphere_photoheating20pc_radiation_pressure1d.yaml"
)


def _radiation_impulse(sim, source_result, dt):
    absorbed = source_result.get("absorbed_photon_rate")
    energies = source_result.get("photon_energy_cgs_erg")
    if absorbed is None or energies is None:
        return 0.0
    absorbed = np.asarray(absorbed, dtype=float)
    if absorbed.ndim == 1:
        absorbed = absorbed[None, :]
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    interior = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    code = CodeUnits.from_mapping(sim.par.CodeUnits)
    volume_cgs_cm3 = np.asarray(sim.mesh.vol[interior], dtype=float) * float(
        (1.0 * code.volume_unit).to_value(unyt.cm**3)
    )
    dt_s = float(np.asarray(dt)) * float((1.0 * code.time_unit).to_value(unyt.s))
    absorbed_energy_rate = np.sum(absorbed * energies[:, None], axis=0)
    return float(
        source_result.get("direction", 1)
        * np.sum(absorbed_energy_rate * volume_cgs_cm3 * dt_s)
        / unyt.c.to_value(unyt.cm / unyt.s)
    )


def _total_radial_momentum(sim):
    interior = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    code = CodeUnits.from_mapping(sim.par.CodeUnits)
    momentum_cgs = float((1.0 * code.momentum_unit).to_value(unyt.g * unyt.cm / unyt.s))
    return float(np.sum(np.asarray(sim.fluid.Mom_code[interior], dtype=float)) * momentum_cgs)


def _pressure_diagnostics(sim, source_result):
    """Estimate radiation and photoheated-gas pressures in cgs units.

    The radiation pressure is the absorbed luminosity divided by the area of
    the ionization front and by ``c``.  The gas pressure is volume-weighted
    over the ionized region, using ``1 - xHI`` as the ionization weight.
    """
    interior = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    code = CodeUnits.from_mapping(sim.par.CodeUnits)
    volume_cgs_cm3 = np.asarray(sim.mesh.vol[interior], dtype=float) * float(
        (1.0 * code.volume_unit).to_value(unyt.cm**3)
    )
    pressure_cgs = et._to_pressure(sim.fluid.pre_code[interior], sim.par)
    ionized_weight = np.clip(1.0 - np.asarray(sim.fluid.xHI[interior], dtype=float), 0.0, 1.0)
    weighted_volume = float(np.sum(volume_cgs_cm3 * ionized_weight))
    gas_pressure = (
        float(np.sum(pressure_cgs * volume_cgs_cm3 * ionized_weight) / weighted_volume)
        if weighted_volume > 0.0
        else 0.0
    )

    absorbed = source_result.get("absorbed_photon_rate") if source_result else None
    energies = source_result.get("photon_energy_cgs_erg") if source_result else None
    radiation_pressure = 0.0
    if absorbed is not None and energies is not None:
        absorbed = np.asarray(absorbed, dtype=float)
        if absorbed.ndim == 1:
            absorbed = absorbed[None, :]
        energies = np.atleast_1d(np.asarray(energies, dtype=float))
        absorbed_luminosity = float(
            np.sum(np.sum(absorbed * energies[:, None], axis=0) * volume_cgs_cm3)
        )
        front_kpc = et.ionization_front_position(sim.mesh, sim.fluid, sim.par)
        front_cm = front_kpc * float((1.0 * unyt.kpc).to_value(unyt.cm))
        if front_cm > 0.0:
            radiation_pressure = absorbed_luminosity / (
                4.0 * np.pi * front_cm**2 * unyt.c.to_value(unyt.cm / unyt.s)
            )
    return radiation_pressure, gas_pressure


def main(config_filename=DEFAULT_CONFIG):
    config = eu.load_nested_example_config(config_filename)
    par = config['par']
    output = par['output']
    eu.clean_previous_outputs(output)
    Path(output['directory']).mkdir(parents=True, exist_ok=True)
    Path(output['savedir']).mkdir(parents=True, exist_ok=True)
    et.write_initial_condition(config)

    sim = Rsim(par)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()

    momentum_history = {
        "time_s": [0.0],
        "gas_momentum": [_total_radial_momentum(sim)],
        "radiation_momentum": [0.0],
    }
    radiation_pressure, gas_pressure = _pressure_diagnostics(sim, None)
    pressure_history = {
        "time_s": [0.0],
        "radiation_pressure": [radiation_pressure],
        "gas_pressure": [gas_pressure],
    }
    radiation_momentum = 0.0

    def step_backend(dt=None, mode="hydro_sources", advect_chemistry=True):
        nonlocal radiation_momentum
        result = sim.Step(
            dt=dt,
            mode=mode,
            advect_chemistry=advect_chemistry,
        )
        if sim.last_source_result is not None:
            radiation_momentum += _radiation_impulse(
                sim,
                sim.last_source_result,
                sim.last_source_dt,
            )
        code = CodeUnits.from_mapping(sim.par.CodeUnits)
        time_s = float(np.asarray(sim.fluid.time_code)) * float(
            (1.0 * code.time_unit).to_value(unyt.s)
        )
        momentum_history["time_s"].append(time_s)
        momentum_history["gas_momentum"].append(_total_radial_momentum(sim))
        momentum_history["radiation_momentum"].append(radiation_momentum)
        radiation_pressure, gas_pressure = _pressure_diagnostics(sim, sim.last_source_result)
        pressure_history["time_s"].append(time_s)
        pressure_history["radiation_pressure"].append(radiation_pressure)
        pressure_history["gas_pressure"].append(gas_pressure)
        return result

    sim.RunAll(
        outputtime=0,
        mode="hydro_sources",
        step_backend=step_backend,
    )

    outputfilenames = et.output_files(output['directory'], output['filename_prefix'])
    history = et.load_history_from_outputs(outputfilenames, config)
    out_par, out_mesh, out_fluid = et.load_output_state(outputfilenames[-1], config)
    figure_stem = "DynamicStromgrenSpherePhotoheating20pcRadiationPressure1D"
    et.save_plot(
        out_mesh,
        out_fluid,
        out_par,
        config,
        Path(output['savedir']) / f"{figure_stem}.jpg",
    )
    et.save_front_plot(
        history,
        config,
        Path(output['savedir']) / f"{figure_stem}_IFront.jpg",
    )

    time_myr = np.asarray(momentum_history["time_s"]) / (1.0 * unyt.Myr).to_value(unyt.s)
    momentum_unit = unyt.g * unyt.cm / unyt.s
    gas = np.asarray(momentum_history["gas_momentum"])
    radiation = np.asarray(momentum_history["radiation_momentum"])
    momentum_figure = Path(output['savedir']) / f"{figure_stem}_Momentum.jpg"
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(time_myr, gas, label="total gas radial momentum")
    plt.plot(time_myr, radiation, "--", label="absorbed photon momentum")
    plt.xlabel("time [Myr]")
    plt.ylabel(f"momentum [{momentum_unit}]" )
    plt.legend()
    plt.tight_layout()
    plt.savefig(momentum_figure, dpi=180)
    plt.close()

    pressure_time_myr = np.asarray(pressure_history["time_s"]) / (
        1.0 * unyt.Myr
    ).to_value(unyt.s)
    radiation_pressure = np.asarray(pressure_history["radiation_pressure"])
    gas_pressure = np.asarray(pressure_history["gas_pressure"])
    pressure_ratio = np.zeros_like(radiation_pressure)
    nonzero_gas_pressure = gas_pressure > 0.0
    pressure_ratio[nonzero_gas_pressure] = (
        radiation_pressure[nonzero_gas_pressure]
        / gas_pressure[nonzero_gas_pressure]
    )
    pressure_figure = Path(output['savedir']) / f"{figure_stem}_PressureRatio.jpg"
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.5), sharex=True)
    axes[0].plot(pressure_time_myr, radiation_pressure, label="effective radiation pressure")
    axes[0].plot(pressure_time_myr, gas_pressure, label="ionized-gas thermal pressure")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"pressure [dyn cm$^{-2}$]")
    axes[0].legend(frameon=False)
    axes[1].plot(pressure_time_myr, pressure_ratio, color="tab:purple")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("time [Myr]")
    axes[1].set_ylabel(r"$P_{\rm rad}/P_{\rm gas}$")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pressure_figure, dpi=180)
    plt.close(fig)
    pressure_csv = Path(output['savedir']) / f"{figure_stem}_PressureRatio.csv"
    np.savetxt(
        pressure_csv,
        np.column_stack((
            pressure_time_myr,
            radiation_pressure,
            gas_pressure,
            pressure_ratio,
        )),
        delimiter=",",
        header="time_Myr,radiation_pressure_dyn_cgs_cm2,gas_pressure_dyn_cgs_cm2,pressure_ratio",
        comments="",
    )

    eu.write_radial_profile_csv(outputfilenames[-1], Path(output['directory']) / "radial_profile_rhd.csv")
    print("final gas momentum = %.6e g cm/s" % gas[-1])
    print("absorbed photon momentum = %.6e g cm/s" % radiation[-1])
    print("momentum figure = %s" % momentum_figure)
    print("final radiation/gas pressure ratio = %.6e" % pressure_ratio[-1])
    print("pressure ratio figure = %s" % pressure_figure)
    print("pressure ratio data = %s" % pressure_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
