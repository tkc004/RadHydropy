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

To exercise the coupled implicit energy--``xHI`` source solver explicitly,
run the companion configuration with the exact Compton-only shortcut disabled::

   python compton_cmb_heating1d.py \
      --config compton_cmb_heating1d_coupled_implicit.yaml

This writes the comparison figure to ``outputs_coupled_implicit`` and uses
``hydrogen_source_solver: coupled_implicit`` with an error fallback, so a
non-convergent nonlinear source solve cannot be hidden by explicit subcycling.
The example also compares each timestep against a run with half the timestep;
if the difference exceeds ``hydrogen_implicit_convergence_tolerance``, it
automatically halves the timestep again up to
``hydrogen_implicit_max_refinements``.
