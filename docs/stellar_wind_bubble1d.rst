Stellar Wind Bubble 1D
======================

The ``example/StellarWindBubble1D`` case evolves a spherically symmetric,
energy-driven stellar-wind bubble following the Weaver et al. (1977)
similarity solution. It injects a fast wind into a low-density ambient medium
and compares the resulting bubble growth against analytic radius, velocity,
and pressure tracks.

The plotting script produces four comparison figures:

* density and temperature profiles with the analytic shock location marked at
  each snapshot;
* a cavity-side inner-shell-edge radius comparison against the Weaver
  solution;
* a shock-velocity comparison against the Weaver solution; and
* a bubble-pressure comparison against the Weaver solution.

The profiles are useful for checking the shell structure directly, while the
time-series plots show whether the simulated bubble follows the expected
energy-driven scaling.

.. figure:: ../example/StellarWindBubble1D/StellarWindBubble1D_profiles.jpg
   :width: 100%
   :alt: Stellar-wind bubble density and temperature profiles

   Density and temperature profiles with the analytic shock location marked
   for each snapshot.

.. figure:: ../example/StellarWindBubble1D/StellarWindBubble1D_radius.jpg
   :width: 100%
   :alt: Stellar-wind bubble radius comparison

   Cavity-side inner-shell-edge radius compared against the Weaver et al.
   (1977) radius.

.. figure:: ../example/StellarWindBubble1D/StellarWindBubble1D_velocity.jpg
   :width: 100%
   :alt: Stellar-wind bubble shock velocity comparison

   Shock velocity compared against the Weaver et al. (1977) solution.

.. figure:: ../example/StellarWindBubble1D/StellarWindBubble1D_pressure.jpg
   :width: 100%
   :alt: Stellar-wind bubble pressure comparison

   Bubble pressure compared against the Weaver et al. (1977) solution.
