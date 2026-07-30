"""Spherical long-characteristic radiative-transfer example.

A source at the coordinate origin emits ionizing photons at a constant rate.
Hydrodynamics and hydrogen thermo-chemistry are not advanced; the script only
applies the optional long-characteristic radiative-transfer update and compares
the resulting photon number density with the analytic optically thin spherical
dilution solution.

The example builds the static spherical problem from YAML parameters, applies
the long-characteristic radiative-transfer update once through ``Rsim``, writes
an HDF5 snapshot, reloads that snapshot, and compares the result with the
analytic optically thin spherical dilution solution.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import tempfile

import unyt

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
example_root = Path(__file__).resolve().parents[1]
if str(example_root) not in sys.path:
    sys.path.insert(0, str(example_root))

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

from radhydropy.example_config import load_example_parameters
import radhydropy.radiative_transfer as rrt
from radhydropy.rsim import Rsim
import example_utils as eu
import tools as et
from radhydropy.units import _as_cgs_float


DEFAULT_CONFIG = Path(__file__).resolve().with_name('radiative_transfer_sph1d.yaml')
start_time = time.time()


def RunRadiativeTransferOnly(sim):
    """Apply radiative transfer once and write a single HDF5 snapshot."""
    print("--- Initization finished. Start running ... ---")
    print("--- %s seconds ---" % (time.time() - start_time))
    sim.solver.SetBoundary(sim.mesh, sim.fluid, sim.par)
    sim.solver.SetConserved(sim.mesh, sim.fluid, verbose=getattr(sim.par, 'verbose', 0))
    result = rrt.trace_long_characteristics(
        sim.mesh,
        sim.fluid.rho,
        sim.fluid.xHI,
        hydrogen_mass_fraction=getattr(sim.par, 'hydrogen_mass_fraction', 1.0),
        sigma_gamma=_as_cgs_float(
            getattr(sim.par, 'hydrogen_sigma_gamma', rrt.rh.DEFAULT_SIGMA_GAMMA),
            unyt.cm**2,
        ),
        boundary_flux=_as_cgs_float(
            getattr(sim.par, 'radiative_transfer_boundary_flux', 0.0),
            1.0 / (unyt.cm**2 * unyt.s),
        ),
        source_photon_rate=_as_cgs_float(
            getattr(sim.par, 'radiative_transfer_source_photon_rate', 0.0),
            1.0 / unyt.s,
        ),
        direction=getattr(sim.par, 'radiative_transfer_direction', 1),
        coordsys=getattr(sim.mesh, 'coordsys', 'cartesian'),
    )
    sim.fluid.ngamma[:] = result.cell_photon_density.to(sim.fluid.ngamma.units)
    if not hasattr(sim.fluid, 'time'):
        sim.fluid.SetFluidTime(0.0 * unyt.s)
    sim.fluid.SetTemperature()
    sim._write_numbered_hdf5(0)
    print("--- Simulation finished. ---")
    print("--- %s seconds ---" % (time.time() - start_time))
    return result


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    print('rundir', rundir)
    runparams, ICparams = load_example_parameters(config_filename, rundir)
    eu.clean_previous_outputs(runparams)
    config = {**runparams, **ICparams}

    par, mesh, fluid, solver = et.build_problem(config)
    sim = Rsim.FromComponents(par, mesh, fluid, solver)
    result = RunRadiativeTransferOnly(sim)

    output_filename = Path(runparams['outdir']) / f"{runparams['outfileprefix']}_000.hdf5"
    out_par, out_mesh, out_fluid = et.load_output_state(output_filename, config)
    relative_error = et.save_plot(
        out_mesh,
        out_fluid,
        out_par,
        config['source_photon_rate'],
        str(Path(runparams['savedir']) / 'RadiativeTransferSph1D.jpg'),
    )

    print('outer face photon rate = %s' % result.face_photon_rate[-1])
    print('max relative error = %.3e' % relative_error)
    print('figure = %s' % (Path(runparams['savedir']) / 'RadiativeTransferSph1D.jpg'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the spherical radiative-transfer example.',
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG,
        help='YAML file containing runparams and ICparams.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
