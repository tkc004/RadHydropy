Uniform EdS Thermo-Chemistry
============================

This example evolves four uniform spherical gas cells with
``Rsim.Run(mode="sources")``.  The four cells are uniform, gravity and fluxes
are disabled, while the EdS cosmology and thermo-chemistry source path remain
active.

The gas starts at the CMB temperature at ``z=100`` and evolves to ``z=10``.
The code-unit reference time is chosen so this interval spans about 1.3 Gyr,
making the temperature evolution visible in the generated figure.

Two runs are made:

* Compton only, compared with the EdS temperature ODE;
* atomic cooling plus Compton coupling, plotted against the Compton-only
  analytic solution.

Atomic cooling has nonlinear temperature and ionization dependence, so it does
not have a closed-form reference in this configuration.  The example checks
that it cools below the Compton-only solution and that all temperatures remain
finite.

Run from this directory with::

   python uniform_eds_thermochemistry1d.py

The figure is written to ``outputs/UniformEdSThermochemistry1D.jpg``.
