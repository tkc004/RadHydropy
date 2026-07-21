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

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

cache_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-cache')
mplconfig_dir = os.path.join(tempfile.gettempdir(), 'radhydropy-matplotlib')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(mplconfig_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)
os.environ.setdefault('MPLCONFIGDIR', mplconfig_dir)

import unyt

import tools as et


rundir = os.path.dirname(os.path.abspath(__file__))
figure_filename = os.path.join(rundir, 'RadiativeTransferSph1D.jpg')

source_photon_rate = 1.0e49 / unyt.s
boxsize = 1.0 * unyt.pc
number_of_cells = 256


def main():
    par, mesh, fluid, result = et.build_static_problem(
        number_of_cells,
        boxsize,
        source_photon_rate,
    )
    relative_error = et.save_plot(
        mesh,
        fluid,
        par,
        source_photon_rate,
        figure_filename,
    )
    print('outer face photon rate = %s' % result.face_photon_rate[-1])
    print('max relative error = %.3e' % relative_error)
    print('figure = %s' % figure_filename)


if __name__ == '__main__':
    main()
