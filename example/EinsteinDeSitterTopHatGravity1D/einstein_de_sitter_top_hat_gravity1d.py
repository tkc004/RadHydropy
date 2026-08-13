"""Analytic cosmological self-gravity diagnostic for a spherical top-hat."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import radhydropy.io as rio
from radhydropy.example_config import load_example_parameters
from radhydropy.gravity import Gravity
from radhydropy.rsim import Rsim
from radhydropy.units import CodeUnits
import example_utils as eu
import tools as et


DEFAULT_CONFIG = Path(__file__).with_name('einstein_de_sitter_top_hat_gravity1d.yaml')


def main(config_filename=DEFAULT_CONFIG):
    rundir = Path.cwd().resolve()
    runparams, icparams = load_example_parameters(config_filename, rundir)
    eu.clean_previous_outputs(runparams)
    units = CodeUnits.from_mapping(runparams['CodeUnits'])
    cosmology = et.EinsteinDeSitter.from_code_units(
        units, t_ref=float(runparams['cosmology_t_ref']),
        a_ref=float(runparams['cosmology_a_ref']),
    )
    initial = et.Simwrap(icparams, units, cosmology)
    rio.writehdf5(initial, runparams['ICfilename'])

    sim = Rsim(runparams)
    sim.Callreadhdf5()
    sim.SetMesh()
    sim.SetFluid()
    sim.SetInitFluid()
    sim.par.gravity = Gravity(
        selfgravity=True, externalgravity=False, cosmological=True,
        cosmology=sim.par.cosmology, code_units=sim.par.CodeUnits,
    )
    numerical = sim.par.gravity.acceleration_on_mesh(sim.mesh, sim.fluid.rho, sim.par)
    physical = slice(sim.par.noghost, sim.par.noghost + sim.par.nogrid)
    radius = np.asarray(sim.mesh.coordinate[physical], dtype=float)
    tau = float(np.asarray(sim.par.time).flat[0])
    a = sim.par.cosmology.scale_factor_from_supercomoving(tau)
    cosmic_time = sim.par.cosmology.cosmic_time_from_supercomoving(tau)
    rho_background = sim.par.cosmology.background_density(cosmic_time)
    analytic = et.top_hat_acceleration(
        radius, float(icparams['top_hat_radius']), float(icparams['overdensity']),
        rho_background * a**3, a, sim.par.cosmology.gravitational_constant,
    )
    comparison = slice(1, None)
    error = np.abs((numerical[physical][comparison] - analytic[comparison]) /
                   np.maximum(np.abs(analytic[comparison]), 1.0e-300))
    max_error = float(np.max(error))
    if not np.isfinite(max_error) or max_error > 5.0e-3:
        raise RuntimeError('top-hat gravity error %.6g exceeds tolerance' % max_error)

    filename = os.path.join(runparams['savedir'], 'EinsteinDeSitterTopHatGravity1D.jpg')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(radius, numerical[physical], label='numerical')
    axes[0].plot(radius, analytic, '--', label='analytic')
    axes[0].set(xlabel='comoving radius [code length]', ylabel='supercomoving acceleration')
    axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(radius[comparison], error)
    axes[1].set(xlabel='comoving radius [code length]', ylabel='relative error')
    axes[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(filename, dpi=200); plt.close(fig)
    print('Einstein-De Sitter top-hat gravity passed')
    print('scale factor = %.8g, maximum relative error = %.6g' % (a, max_error))
    print('figure = %s' % filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
