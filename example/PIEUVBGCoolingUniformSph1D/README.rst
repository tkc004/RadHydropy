PIEUVBGCoolingUniformSph1D
==========================

This example exercises the non-radiative-transfer HM12
``PIEUVBGCoolingNetwork`` in hydrodynamics. It runs two uniform spherical
gas cases at ``z=4``:

* ``nH=1 cm^-3`` receives HM12 photoheating and cooling;
* ``nH=100 cm^-3`` is self-shielded, so only the HM12 cooling rate is used.

Run it from this directory with::

   python pie_uvbg_cooling_uniform_sph1d.py

The script writes case snapshots under ``outputs/`` and the comparison figure
``PIEUVBGCoolingUniformSph1D.jpg``.
