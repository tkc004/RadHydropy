NFW boundary-driven virial shock
================================

This experiment continuously feeds cold gas through the outer boundary of a
fixed ``1e12 Msun`` NFW halo.  It avoids relying on a finite reservoir of the
original Hubble-flow initial condition.  A virial-temperature hydrostatic
atmosphere seeds the downstream state inside ``0.8 R200``; the outer state has
``v=-1.25 V200`` and ``rho=Mdot/(4 pi r^2 |v|)`` for ``Mdot=30 Msun/yr``.
Its initial temperature is the equilibrium temperature interpolated from the
HM12 UV-background PIE table.

The runner first settles the accretion flow adiabatically for 800 Myr, then
restarts the final snapshot with HM12 PIE heating and cooling for another
1.2 Gyr (2 Gyr total).  The inner boundary is a diode (gas may leave but is never injected),
while the cold outer inflow is maintained.  Outputs include density,
temperature and radial-velocity profiles, plus independently measured shock
radius histories for both stages.  A separate PIE stability report and figure
compare simulated and finite-Mach Rankine--Hugoniot post-shock pressure,
temperature, and cooling time, and plot the measured and analytic
``gamma_eff`` against the Birnboim--Dekel ``10/7`` stability threshold.

Run from the repository root with::

  python example/NFWBoundaryDrivenVirialShock1D/nfw_boundary_driven_virial_shock1d.py

The default 512-cell, 2-Gyr calculation is intended as a resolved production
example.  For quick checks, copy the YAML and reduce ``nogrid`` and both final
times while keeping output times within the selected intervals.

Halo-mass sequence
------------------

Two additional configurations keep the same supplied ``30 Msun/yr`` baryon
flux while lowering the fixed NFW mass to ``3e11`` and ``1e11 Msun``.  This
raises the gas density relative to the virial temperature and tests the
cooling-driven loss of shock support. Run and compare all three cases with::

  python example/NFWBoundaryDrivenVirialShock1D/run_mass_sequence.py

If all snapshots already exist, regenerate only the comparison with::

  python example/NFWBoundaryDrivenVirialShock1D/run_mass_sequence.py --skip-runs

The mass-sequence report marks a state ``unstable`` when the measured
``gamma_eff`` is below ``10/7`` and marks snapshots without a simultaneous
compressive density, temperature, and velocity jump as
``no_resolved_virial_shock``.

High-cadence instability check
------------------------------

For a stronger time-domain test of the ``1e11 Msun`` case, run the dedicated
configuration::

  python example/NFWBoundaryDrivenVirialShock1D/nfw_boundary_driven_virial_shock1d.py \
    --config example/NFWBoundaryDrivenVirialShock1D/nfw_boundary_driven_virial_shock_1e11_long.yaml

It evolves to 4 Gyr total and saves PIE snapshots every 50 Myr.  The saved
diagnostics include ``P_post/P_ram``, where
``P_ram=rho_upstream*v_relative**2``.  In the validated run the shock moved
from ``0.860`` to ``0.887 R200`` and remained resolved, while
``P_post/P_ram`` varied from about ``0.79`` to ``1.25`` and
``gamma_eff`` repeatedly fell below ``10/7``.  This demonstrates an
unsupported, time-dependent radiative shock; it does not show complete shock
disappearance within the 4-Gyr interval.

The identical high-cadence experiment for a massive halo is available with::

  python example/NFWBoundaryDrivenVirialShock1D/nfw_boundary_driven_virial_shock1d.py \
    --config example/NFWBoundaryDrivenVirialShock1D/nfw_boundary_driven_virial_shock_1e13_long.yaml

Its outputs are written under ``outputs/mass_1e13_long`` and use the same
profile, shock-history, pressure-ratio, cooling-time, and effective-gamma
plots.
