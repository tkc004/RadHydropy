CMB Compton Heating and Cooling 1D
==================================

This fixed-density benchmark runs two mostly neutral hydrogen parcels with a
one-percent free-electron fraction at redshift ``z=100``: one starts at
``10^6 K`` and cools, while the other starts at ``1 K`` and heats. Radiative
transfer, recombination, collisional ionization, atomic cooling, and
hydrodynamic evolution are disabled so that the optional CMB Compton source
can be compared with its analytic exponential solution.

Run from this directory with::

   python compton_cmb_heating1d.py

The output figure compares both numerical histories with the analytic solution
and plots the relative error. The equilibrium temperature is
``T_CMB = 2.7255 (1 + z) K``. The low density in the setup makes ordinary
free-free cooling negligible compared with the Compton term.
