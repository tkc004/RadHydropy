"""Uniform spherical HM12 PIE cooling/heating hydro test."""

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import unyt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, EXAMPLE_ROOT, EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/radhydropy-matplotlib")))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.thermo_networks.pie import MetalPIETable
from radhydropy.units import CodeUnits
import example_utils as eu
from tools import Simwrap


DEFAULT_CONFIG = EXAMPLE_DIR / "pie_uvbg_cooling_uniform_sph1d.yaml"
CASES = {"diffuse": 1.0, "self_shielded": 100.0}


def _snapshot(filename):
    with h5py.File(filename, "r") as handle:
        data = handle["Data"]
        header = handle["Header"]
        noghost = int(header.attrs.get("GhostCells", 0))
        nogrid = int(header.attrs["GridCells"])
        first = noghost
        last = first + nogrid
        boundary = np.asarray(data["Boundary"][()])[first : last + 1]
        return {
            "radius": 0.5 * (boundary[:-1] + boundary[1:]),
            "density": np.asarray(data["Density"][()])[first:last],
            "temperature": np.asarray(data["Temperature"][()])[first:last],
        }


def _run_case(config, label, hydrogen_density_cgs_cm3, table):
    par = config['par']; initial_mapping = config['initial_condition']
    output_dir = EXAMPLE_DIR / "outputs" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    case = {**par, 'simulation': {**par['simulation'], 'initial_condition_filename': str(output_dir / f'InitialCondition_{label}.hdf5')}, 'output': {**par['output'], 'directory': str(output_dir), 'savedir': str(output_dir), 'filename_prefix': f'Output_{label}'}}

    case_config = {'par': case, 'initial_condition': {**initial_mapping, 'hydrogen_mass_fraction': case['thermochemistry']['hydrogen_mass_fraction'], 'proton_mass_g': float(unyt.mp.to_value(unyt.g)), 'vini': 0.0 * unyt.cm / unyt.s}, 'example': config['example']}
    code_units = CodeUnits.from_mapping(case['units']['CodeUnits'])
    ric = Simwrap(case_config, code_units, hydrogen_density_cgs_cm3)
    rio.writehdf5(ric, case['simulation']['initial_condition_filename'])

    runtime_only = {
        'final_time', 'number_of_cells', 'evolution_timestep',
        'chemistry_timestep', 'box_size', 'coordinate_system',
        'current_time', 'grid_cells', 'initial_temperature',
        'mean_molecular_weight',
    }
    sim = Rsim(case)
    sim.RunAll(outputtime=0, mode="hydro")
    snapshots = sorted(output_dir.glob(f"{case['output']['filename_prefix']}_*.hdf5"))
    if len(snapshots) < 2:
        raise RuntimeError(f"expected initial and final snapshots in {output_dir}")

    temperature = float(case_config['initial_condition']["tempini"].to_value(unyt.K))
    heating, cooling = table.rates(
        temperature,
        hydrogen_density_cgs_cm3,
        metallicity=case['thermochemistry']["metallicity"],
        redshift=case['thermochemistry']["metal_pie_redshift"],
    )
    if hydrogen_density_cgs_cm3 > case['thermochemistry']["metal_pie_photoheating_max_density_cgs_cm3"]:
        heating_used = 0.0
    else:
        heating_used = heating
    print(
        f"{label}: nH={hydrogen_density_cgs_cm3:g} cm^-3, "
        f"table heating={heating:.6e}, used heating={heating_used:.6e}, "
        f"cooling={cooling:.6e}, net={heating_used - cooling:.6e} "
        "erg cm^-3 s^-1"
    )
    return snapshots, output_dir


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    config = eu.load_nested_example_config(config_filename)
    par = config['par']
    table_path = (config_filename.parent / par['thermochemistry']["metal_pie_table_filename"]).resolve()
    par['thermochemistry']['metal_pie_table_filename'] = str(table_path)
    table = MetalPIETable(table_path)
    if not table.is_hm12_uv_background:
        raise ValueError("the example requires an HM12 UV-background table")
    if par['radiation'].get("radiative_transfer", False):
        raise ValueError("the example requires radiative_transfer: false")

    results = {}
    for label, density in CASES.items():
        results[label] = _run_case(config, label, density, table)

    figure = EXAMPLE_DIR / "PIEUVBGCoolingUniformSph1D.jpg"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
    for column, (label, (snapshots, _)) in enumerate(results.items()):
        initial = _snapshot(snapshots[0])
        final = _snapshot(snapshots[-1])
        # The physical radius is ~1e15 cm.  Plotting those large absolute
        # values can trigger a misleading sliver/spike artifact in some
        # Matplotlib backends when the temperature is nearly uniform.
        radius_scale = 1.0e15
        radius_initial = initial["radius"] / radius_scale
        radius_final = final["radius"] / radius_scale
        # Remove round-off-level differences before the logarithmic plot;
        # otherwise the renderer can amplify 1e-11 K noise into visible
        # downward spikes.
        temperature_initial = np.round(initial["temperature"], decimals=6)
        temperature_final = np.round(final["temperature"], decimals=6)
        expected_style = {
            "linestyle": "None",
            "marker": "s",
            "markersize": 3.5,
        }
        simulation_style = {"linestyle": "-", "linewidth": 1.5}
        axes[0, column].plot(
            radius_initial,
            temperature_initial,
            label="expected",
            **expected_style,
        )
        axes[0, column].plot(
            radius_final,
            temperature_final,
            label="simulation",
            **simulation_style,
        )
        axes[1, column].plot(
            radius_initial,
            initial["density"],
            label="expected",
            **expected_style,
        )
        axes[1, column].plot(
            radius_final,
            final["density"],
            label="simulation",
            **simulation_style,
        )
        axes[0, column].set_title(
            rf"{label}: $n_H={CASES[label]:g}\ \mathrm{{cm^{{-3}}}}$"
        )
        axes[0, column].set_yscale("log")
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel(r"radius [$10^{15}$ cm]")
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
    axes[0, 0].set_ylabel("temperature [K]")
    axes[1, 0].set_ylabel("density [code units]")
    axes[0, 1].legend(frameon=False)
    fig.suptitle("HM12 PIE UV-background cooling without radiative transfer")
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"figure = {figure}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
