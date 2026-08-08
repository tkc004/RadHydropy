Stellar Wind Bubble 1D
======================

The ``example/StellarWindBubble1D`` case evolves a spherically symmetric,
energy-driven stellar-wind bubble following the Weaver et al. (1977)
similarity solution. It injects a fast wind into a low-density ambient medium
and compares the resulting bubble growth against analytic radius, velocity,
and pressure tracks.

Two YAML configurations are provided:

* ``stellar_wind_bubble1d.yaml`` uses CHIANTI CIE cooling with solar
  metallicity (``metallicity: 1.0``) and 70% hydrogen by mass.
* ``stellar_wind_bubble1d_no_metal.yaml`` uses the hydrogen network and a
  100% hydrogen composition. Hydrogen source evolution is disabled in this
  baseline configuration, so it provides a no-metal, no-radiative-cooling
  comparison.

The CIE run reads the ion-fraction and cooling tables from the sibling
``CHIANTI_11.0.2_database`` directory by default. CIE cooling is integrated
with explicit adaptive subcycling; the default ``cooling_safety_factor`` is
``0.1``. The source update enforces the configured
``cooling_temperature_floor``.

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

.. figure:: ../example/StellarWindBubble1D/StellarWindBubble1D_no_metal_profiles.jpg
   :width: 100%
   :alt: Stellar-wind bubble density and temperature profiles

   Density and temperature profiles with the analytic shock location marked
   for each snapshot.

.. figure:: ../example/StellarWindBubble1D/StellarWindBubble1D_no_metal_radius.jpg
   :width: 100%
   :alt: Stellar-wind bubble radius comparison

   Cavity-side inner-shell-edge radius compared against the Weaver et al.
   (1977) radius.

.. figure:: ../example/StellarWindBubble1D/StellarWindBubble1D_no_metal_velocity.jpg
   :width: 100%
   :alt: Stellar-wind bubble shock velocity comparison

   Shock velocity compared against the Weaver et al. (1977) solution.

.. figure:: ../example/StellarWindBubble1D/StellarWindBubble1D_no_metal_pressure.jpg
   :width: 100%
   :alt: Stellar-wind bubble pressure comparison

   Bubble pressure compared against the Weaver et al. (1977) solution.

To run either configuration:

.. code-block:: bash

   cd example/StellarWindBubble1D
   python stellar_wind_bubble1d.py --config stellar_wind_bubble1d.yaml
   python stellar_wind_bubble1d.py --config stellar_wind_bubble1d_no_metal.yaml

Both configurations use ``InitialCondition.hdf5`` and ``Output_*.hdf5`` in
the example directory, so run them separately if the snapshots need to be
preserved. The figure prefixes remain distinct: ``with_metal`` and
``no_metal``.
