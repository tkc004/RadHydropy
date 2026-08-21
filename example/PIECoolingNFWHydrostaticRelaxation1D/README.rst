PIE Cooling NFW Hydrostatic Relaxation
======================================

This example starts from the existing isothermal hydrostatic atmosphere in a
fixed ``5.497e8 Msun`` NFW potential (with ``Tvir`` approximately 5000 K) and
enables the HM12 PIE heating/cooling network. It deliberately tests whether the nominal hydrostatic atmosphere
remains stable once radiative sources are active; runaway cooling or expansion
is a possible result.

The default mesh extends to ``rmin = 0.01 kpc`` and fills the central region
with gas. Because ``OutflowSph`` prescribes the inner ghost state, its
``rho_outflow``, ``temp_outflow``, and ``mu_outflow`` values are matched to the
central hydrostatic atmosphere rather than left at the generic defaults.

Diagnostics include:

* central density and temperature evolution;
* whether the HM12 temperature floor is reached;
* atmosphere mass inside ``R200`` as a contraction/expansion diagnostic;
* the force-balance residual ``(dP/dr + rho*g)/(rho*g)``.

Run it from this directory::

   python pie_cooling_nfw_hydrostatic_relaxation1d.py

The script writes saved snapshots under ``outputs/``, together with
``PIECoolingNFWHydrostaticRelaxation1D.jpg`` and
``PIECoolingNFWHydrostaticRelaxation1D_Report.txt``.

The default run time is 5 Gyr, with additional snapshots at 0.5, 1, 2, 3,
4, and 5 Gyr.

An equivalent ``Tvir approximately 8000 K`` case is provided by
``pie_cooling_nfw_hydrostatic_relaxation1d_8000K.yaml``. It writes a separate
``PIECoolingNFWHydrostaticRelaxation1D_8000K.jpg`` figure and report.

The HM12 table is distributed with Git LFS through
`tkc004/RadhydropyData <https://github.com/tkc004/RadhydropyData>`_.
