PIE Spherical Radiative Shock
=============================

This is a deliberately simple spherical radiative-shock experiment designed
to test cooling-induced shock overstability. It has no gravity and no
cosmological expansion. Cold supersonic gas streams outward from the inner
boundary while an equal and opposite stream enters at the outer boundary. The
streams collide near the middle of the spherical shell and form a shock.

The custom mixed boundary condition maintains both streams. HM12
photoionization-equilibrium heating and cooling acts in the post-shock layer.
The shock history and cooling-layer width can be compared against the
estimated PIE cooling length. Strong cooling can suppress or destabilize the
shock when the cooling layer is sufficiently thin, but a stationary shock is
not by itself evidence of stability: the cooling layer must also be resolved
and the run must cover several cooling times.

Run it from this directory::

   python pie_spherical_radiative_shock1d.py

The example runs three cases:

* ``adiabatic``: cooling disabled, used as the stable reference;
* ``pie_z0p1``: HM12 PIE heating/cooling with ``Z=0.1``;
* ``pie_z1``: HM12 PIE heating/cooling with ``Z=1``.

Each case writes snapshots and ``ShockHistory.txt`` in its own subdirectory
under ``outputs/``. A combined comparison figure is written to
``outputs/PIESphericalRadiativeShock1D.jpg``.

The comparison figure contains density and temperature profiles at five
output times for every case, together with the shock-radius history. The
displayed times are reconstructed from the snapshot sequence and configured
run duration because the current HDF5 output header does not preserve the
evolving hydro time.

The left panels show absolute density in ``g cm**-3`` and the middle panels
show absolute temperature in K. The right panels track the strongest density
gradient near the collision region. ``ShockHistory.txt`` contains the sampled
shock radius and, for PIE runs, a final estimate of the post-shock cooling
time, cooling length, and number of grid cells per cooling length.

The metallicity cases are intended to show the transition from a weakly
cooled shock to a strongly cooled shock that may become overstable:

* ``adiabatic`` is the stable reference with cooling disabled;
* ``pie_z0p1`` uses HM12 PIE heating/cooling with ``Z=0.1``;
* ``pie_z1`` uses HM12 PIE heating/cooling with ``Z=1``.

A growing oscillation amplitude in the shock-radius history indicates
overstability. Monotonic inward motion indicates shock collapse instead. In
the current default 20 Myr, 512-cell run, cooling visibly changes the
post-shock temperature structure, especially for ``Z=1``, but the shock-radius
history does not yet show a clear growing oscillation. This is a diagnostic
result, not a claim that the flow is unconditionally stable.

The default run is 20 Myr, which covers several cooling times for the solar

Resolution matters for this test. A cooling length comparable to one or two
cells is under-resolved and can produce numerical suppression or spurious
behavior. For a stronger overstability test, increase ``nogrid`` to 1024 or
2048 and run for 50--100 Myr while retaining the 0.5 Myr output cadence.
