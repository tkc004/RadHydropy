"""Spherical long-characteristic radiative-transfer example.

A source at the coordinate origin emits ionizing photons at a constant rate.
Hydrodynamics and hydrogen thermo-chemistry are not advanced; the script only
applies the optional long-characteristic radiative-transfer update and compares
the resulting photon number density with the analytic optically thin spherical
dilution solution.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.fluid import Fluid
import radhydropy.hydrogen as rh
from radhydropy.mesh import Mesh
from radhydropy.solver import Solver


rundir = os.path.dirname(os.path.abspath(__file__))
figure_filename = os.path.join(rundir, 'RadiativeTransferSph1D.jpg')

source_photon_rate = 1.0e49 / unyt.s
boxsize = 1.0 * unyt.pc
number_of_cells = 256


def build_static_problem():
    par = SimpleNamespace(
        coordsys='spherical',
        boundcond='OpenSph',
        nogrid=number_of_cells,
        noghost=2,
        area=1.0 * unyt.cm**2,
        hydrogen_chemistry=False,
        hydrogen_mass_fraction=1.0,
        hydrogen_ngamma_initial=0.0 / unyt.cm**3,
        hydrogen_sigma_gamma=0.0 * unyt.cm**2,
        radiative_transfer=True,
        radiative_transfer_method='long_characteristics',
        radiative_transfer_boundary_flux=0.0 / (unyt.cm**2 * unyt.s),
        radiative_transfer_source_photon_rate=source_photon_rate,
        radiative_transfer_direction=1,
    )

    mesh = Mesh()
    mesh.boundary = np.linspace(
        0.0,
        boxsize.to_value(unyt.cm),
        par.nogrid + 1,
    ) * unyt.cm
    mesh.SetUpMesh(par)

    fluid = Fluid()
    fluid.rho = np.ones(par.nogrid) * unyt.mp / unyt.cm**3
    fluid.vel = np.zeros(par.nogrid) * unyt.cm / unyt.s
    fluid.temp = np.ones(par.nogrid) * unyt.K
    fluid.mu = np.ones(par.nogrid)
    fluid.xHI = np.ones(par.nogrid)
    fluid.SetUpFluid(par)

    solver = Solver()
    solver.SetBoundary(mesh, fluid, par)
    result = solver.ApplyRadiativeTransfer(mesh, fluid, par)
    return par, mesh, fluid, result


def interior_slice(par):
    first = par.noghost
    return slice(first, first + par.nogrid)


def analytic_finite_volume_density(mesh, par):
    interior = interior_slice(par)
    boundary = mesh.boundary[interior.start : interior.stop + 1]
    dr = (boundary[1:] - boundary[:-1]).to(unyt.cm)
    volume = mesh.vol[interior].to(unyt.cm**3)
    return (source_photon_rate * dr / volume / rh.SPEED_OF_LIGHT).to(
        1.0 / unyt.cm**3
    )


def analytic_point_density(radius):
    return (
        source_photon_rate
        / (4.0 * np.pi * radius.to(unyt.cm)**2 * rh.SPEED_OF_LIGHT)
    ).to(1.0 / unyt.cm**3)


def save_plot(mesh, fluid, par):
    interior = interior_slice(par)
    radius = mesh.coordinate[interior].to(unyt.pc)
    simulated = fluid.ngamma[interior].to(1.0 / unyt.cm**3)
    analytic_fv = analytic_finite_volume_density(mesh, par)

    r_min = mesh.boundary[interior.start + 1]
    r_max = mesh.boundary[interior.stop]
    radius_line = np.geomspace(
        r_min.to_value(unyt.pc),
        r_max.to_value(unyt.pc),
        512,
    ) * unyt.pc
    analytic_point = analytic_point_density(radius_line)

    relative_error = np.max(
        np.abs((simulated - analytic_fv) / analytic_fv).to_value('')
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(
        radius.to_value(unyt.pc),
        simulated.to_value(1.0 / unyt.cm**3),
        marker='o',
        ms=3.0,
        lw=0.0,
        label='RadHydropy long characteristic',
    )
    ax.plot(
        radius.to_value(unyt.pc),
        analytic_fv.to_value(1.0 / unyt.cm**3),
        color='black',
        lw=2.0,
        label='Analytic finite-volume average',
    )
    ax.plot(
        radius_line.to_value(unyt.pc),
        analytic_point.to_value(1.0 / unyt.cm**3),
        color='tab:orange',
        ls='--',
        lw=1.5,
        label=r'$Q/(4\pi r^2 c)$',
    )
    ax.text(
        0.04,
        0.06,
        'max relative error = %.2e' % relative_error,
        transform=ax.transAxes,
    )
    ax.set_xlabel('Radius [pc]')
    ax.set_ylabel(r'Photon number density [cm$^{-3}$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)
    return relative_error


def main():
    par, mesh, fluid, result = build_static_problem()
    relative_error = save_plot(mesh, fluid, par)
    print('outer face photon rate = %s' % result.face_photon_rate[-1])
    print('max relative error = %.3e' % relative_error)
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    main()
