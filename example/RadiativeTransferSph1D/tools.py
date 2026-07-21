"""Helper utilities for the spherical radiative-transfer example."""

from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.fluid import Fluid
from radhydropy.mesh import Mesh
from radhydropy.solver import Solver
import radiative_transfer_analytic as rta


def build_static_problem(number_of_cells, boxsize, source_photon_rate):
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


def save_plot(mesh, fluid, par, source_photon_rate, figure_filename):
    interior = interior_slice(par)
    radius = mesh.coordinate[interior].to(unyt.pc)
    simulated = fluid.ngamma[interior].to(1.0 / unyt.cm**3)
    analytic_fv = rta.finite_volume_density(
        mesh.boundary[interior.start : interior.stop + 1],
        mesh.vol[interior],
        source_photon_rate,
    )

    r_min = mesh.boundary[interior.start + 1]
    r_max = mesh.boundary[interior.stop]
    radius_line = np.geomspace(
        r_min.to_value(unyt.pc),
        r_max.to_value(unyt.pc),
        512,
    ) * unyt.pc
    analytic_point = rta.point_density(radius_line, source_photon_rate)

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
