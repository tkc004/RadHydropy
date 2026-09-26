Adiabatic NFW Virial Shock Benchmark
====================================

This separate benchmark evolves cold gas falling into a fixed ``1e12 Msun``
NFW halo. The spherical domain extends to approximately ``4 R200`` and the
gas is adiabatic with ``gamma=5/3``. It is intended to verify the basic
accretion-shock structure before radiative cooling is introduced.

Run it from this directory::

   python nfw_virial_shock_adiabatic1d.py

The run evolves to ``2000 Myr`` and writes saved HDF5 snapshots, including a
final ``2000 Myr`` output, plus ``NFWVirialShockAdiabatic1D.jpg`` and
``NFWVirialShockAdiabatic1D_RankineHugoniot.txt``. The report contains the
shock radius in kpc and units of ``R200``, shock speed, Mach number, and the
measured and finite-Mach Rankine--Hugoniot density and temperature ratios.
Running from a clean checkout
----------------------------

From the repository root, install RadHydropy and the example dependencies::

   cd RadHydropy
   python -m pip install -e ".[test,docs]"

Then change into this example directory before running the command shown above::

   cd example/NFWVirialShockAdiabatic1D

If a command above begins with ``python example/``, run that command from
the repository root instead of changing into this directory.
