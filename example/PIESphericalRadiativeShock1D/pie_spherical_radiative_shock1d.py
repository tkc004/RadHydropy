"""Gravity-free spherical radiative-shock overstability experiment."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, EXAMPLE_ROOT, EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import radhydropy.io as rio
from radhydropy.rsim import Rsim
from radhydropy.solver import Solver
from radhydropy.units import code_unit_scales
from radhydropy.units import CodeUnits
from radhydropy.thermo_networks.pie import MetalPIETable
import example_utils as eu
from tools import Simwrap, estimate_cooling_length, load_snapshot, shock_history


DEFAULT_CONFIG = EXAMPLE_DIR / 'pie_spherical_radiative_shock1d.yaml'
CASES = (
    ('adiabatic', 'adiabatic control', False, 0.0),
    ('pie_z0p1', 'PIE, Z=0.1', True, 0.1),
    ('pie_z1', 'PIE, Z=1', True, 1.0),
)


class CollidingStreamsSolver(Solver):
    """Maintain an inner outward stream and an outer inward stream."""

    def SetBoundary(self, mesh, fluid, par):
        first = par.mesh.ghost_cells
        last = first + par.mesh.grid_cells - 1
        right_start = first + par.mesh.grid_cells
        left_ghost = slice(0, first)
        right_ghost = slice(right_start, right_start + par.mesh.ghost_cells)
        scales = code_unit_scales(getattr(par, 'CodeUnits', None))

        left_state = {
            'rho': par.boundary.outflow_density,
            'vel': par.boundary.outflow_velocity,
            'pre': fluid.eos.pressure(
                par.boundary.outflow_density,
                par.boundary.outflow_temperature,
                par.boundary.outflow_mu,
            ),
        }
        right_state = {
            'rho': par.boundary.inflow_density,
            'vel': par.boundary.inflow_velocity,
            'pre': fluid.eos.pressure(
                par.boundary.inflow_density,
                par.boundary.inflow_temperature,
                par.boundary.inflow_mu,
            ),
        }
        for state, side in ((left_state, left_ghost), (right_state, right_ghost)):
            if hasattr(fluid, 'xHI'):
                state['xHI'] = getattr(par, 'hydrogen_xHI_inflow', 1.0)
            if hasattr(fluid, 'ngamma'):
                state['ngamma'] = self._to_code_number_density(
                    getattr(par, 'hydrogen_ngamma_inflow', 0.0), scales
                )
            self._copy_boundary_state(fluid, side, state)


def _run_case(base_runparams, icparams, label, title, pie_enabled, metallicity, table):
    case = dict(base_runparams)
    case_dir = Path(base_runparams['outdir']).resolve() / label
    case_dir.mkdir(parents=True, exist_ok=True)
    case.update({
        'simname': f"PIESphericalRadiativeShock1D_{label}",
        'ICfilename': str(case_dir / 'InitialCondition.hdf5'),
        'outdir': str(case_dir),
        'savedir': str(case_dir),
        'outfileprefix': 'Output',
        'metallicity': metallicity,
        'metal_pie_enabled': pie_enabled,
    })
    if not pie_enabled:
        case['thermochemistry_network'] = 'hydrogen'

    eu.clean_previous_outputs(case)
    code_units = CodeUnits.from_mapping(case.get('CodeUnits'))

    initial = Simwrap(
        icparams, code_units, float(case['hydrogen_mass_fraction'])
    )
    rio.writehdf5(initial, case['ICfilename'])

    runtime_only = {
        'final_time', 'number_of_cells', 'evolution_timestep',
        'chemistry_timestep', 'box_size', 'coordinate_system',
        'current_time', 'grid_cells', 'initial_temperature',
        'mean_molecular_weight',
    }
    sim = Rsim({key: value for key, value in case.items()
                if key not in runtime_only})
    sim.par.boundary = SimpleNamespace(
        outflow_density=case['rho_outflow'],
        outflow_velocity=case['vel_outflow'],
        outflow_temperature=case['temp_outflow'],
        outflow_mu=case['mu_outflow'],
        inflow_density=case['rho_inflow'],
        inflow_velocity=case['vel_inflow'],
        inflow_temperature=case['temp_inflow'],
        inflow_mu=case['mu_inflow'],
    )
    sim.solver = CollidingStreamsSolver()
    # Maintain an outward inner stream and inward outer stream so the shock
    # forms near the initial midpoint instead of at a reflecting wall.
    sim.RunAll(outputtime=0, mode='hydro_sources' if pie_enabled else 'hydro')

    output_files = sorted(
        case_dir.glob(f"{case['outfileprefix']}_*.hdf5")
    )
    if len(output_files) < 2:
        raise RuntimeError(f'expected snapshots for {label}')

    # Output headers currently do not retain the evolving hydro time.  The
    # numbered snapshots span the configured run, so use their normalized
    # positions for plot/report labels until that core I/O issue is fixed.
    final_time_myr = float(case['timesim'].to_value('Myr'))
    history = shock_history(output_files)
    history[:, 0] = np.linspace(0.0, final_time_myr, len(output_files))
    final_snapshot = load_snapshot(output_files[-1])
    cooling = None if not pie_enabled else estimate_cooling_length(
        final_snapshot, table, metallicity,
        float(case['hydrogen_mass_fraction']), float(icparams['muini']),
    )
    report = case_dir / 'ShockHistory.txt'
    with report.open('w', encoding='utf-8') as stream:
        stream.write('time_Myr shock_radius_kpc\n')
        for time, radius in history:
            stream.write(f'{time:.8g} {radius:.8g}\n')
        stream.write('\nfinal_cooling_diagnostics\n')
        if cooling is None:
            stream.write('cooling_length_kpc nan\n')
        else:
            stream.write(
                'cooling_time_Myr %.8g\ncooling_length_kpc %.8g\n'
                'cooling_cells %.8g\n' % (
                    cooling['cooling_time_Myr'], cooling['cooling_length_kpc'],
                    cooling['cooling_cells'],
                )
            )
    return {
        'label': label,
        'title': title,
        'history': history,
        'snapshots': output_files,
        'report': report,
        'cooling': cooling,
    }


def main(config_filename=DEFAULT_CONFIG):
    config_filename = Path(config_filename).resolve()
    nested = eu.load_nested_example_config(config_filename)
    runparams = eu.legacy_example_parameters(nested)
    icparams = nested['initial_condition']
    if runparams.get('metal_pie_table_filename'):
        runparams['metal_pie_table_filename'] = str(
            (config_filename.parent / runparams['metal_pie_table_filename']).resolve()
        )
    table = MetalPIETable(runparams['metal_pie_table_filename'])
    Path(runparams['outdir']).mkdir(parents=True, exist_ok=True)
    results = [
        _run_case(runparams, icparams, label, title, pie_enabled, metallicity, table)
        for label, title, pie_enabled, metallicity in CASES
    ]

    figure = Path(runparams['outdir']).resolve() / 'PIESphericalRadiativeShock1D.jpg'
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), squeeze=False)
    for row, result in enumerate(results):
        sample_indices = np.unique(np.linspace(
            0, len(result['snapshots']) - 1, 5, dtype=int
        ))
        sample_times = result['history'][sample_indices, 0]
        for index, time_myr in zip(sample_indices, sample_times):
            snapshot = load_snapshot(result['snapshots'][index])
            radius = (
                0.5 * (snapshot['boundary_cm'][1:] + snapshot['boundary_cm'][:-1])
                / 3.0856775814913673e21
            )
            label = f'{time_myr:.2g} Myr'
            axes[row, 0].plot(
                radius, snapshot['density_g_cm3'], label=label,
            )
            axes[row, 1].plot(
                radius, snapshot['temperature_K'], label=label,
            )
        axes[row, 2].plot(
            result['history'][:, 0], result['history'][:, 1],
            marker='o', ms=3, label=result['title'],
        )
        axes[row, 0].set_ylabel(f"{result['title']}\\n$\\rho$ [g cm$^{{-3}}$]")
        axes[row, 1].set_ylabel('T [K]')
        axes[row, 1].set_yscale('log')
        axes[row, 2].set_ylabel('shock radius [kpc]')
        for column in range(3):
            axes[row, column].grid(alpha=0.25)
            axes[row, column].legend(frameon=False, fontsize=8)
    for column in range(2):
        axes[2, column].set_xlabel('r [kpc]')
        for row in range(3):
            if column == 0:
                axes[row, column].set_yscale('log')
            else:
                axes[row, column].set_yscale('log')
    for row in range(3):
        axes[row, 2].set_xlabel('time [Myr]')
        collision_radius = 0.5 * (
            float(icparams['rmin'].to_value('kpc'))
            + float(icparams['rmax'].to_value('kpc'))
        )
        axes[row, 0].set_xlim(collision_radius - 3.0, collision_radius + 3.0)
        axes[row, 1].set_xlim(collision_radius - 3.0, collision_radius + 3.0)
    axes[0, 0].set_title('Absolute density profiles')
    axes[0, 1].set_title('Absolute temperature profiles')
    axes[0, 2].set_title('Shock-radius history')
    fig.suptitle('Gravity-free spherical shock: adiabatic and HM12 PIE cases')
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    for result in results:
        print(f"{result['label']}: snapshots = {len(result['snapshots'])}")
        print(f"{result['label']}: shock history = {result['report']}")
    print(f'figure = {figure}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
