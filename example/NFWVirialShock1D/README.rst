NFW Virial Shock 1D
===================

This example uses the same ``1e8 Msun`` NFW halo as the hydrostatic-equilibrium
example. Uniform gas at the cosmic-mean baryon density starts with the Hubble
velocity field and a CMB temperature at the configured initial redshift. The
NFW halo is the central dark-matter perturbation; the gas evolves through its
fixed potential and develops an infall shock.

Run from this directory with::

   python nfw_virial_shock1d.py

The outer computational boundary is ``2 R_200``; it is not a gas shell. The
output figure shows the density and temperature profiles at the saved times.
The halo is external and gas self-gravity is not included.

The runner also writes ``NFWVirialShock1D_RankineHugoniot.txt``. It detects the
strongest temperature jump, estimates the shock speed from neighboring
snapshots, measures the upstream Mach number, and compares the measured density
and temperature jumps with the finite-Mach Rankine--Hugoniot relations.
