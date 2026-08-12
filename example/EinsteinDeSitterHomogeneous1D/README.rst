Einstein--de Sitter Homogeneous Expansion
=========================================

This Phase 1 cosmology diagnostic uses supercomoving variables. It verifies
that a uniform ``gamma=5/3`` gas remains constant in supercomoving density,
velocity, pressure, and energy while the reconstructed physical variables
follow the Einstein--de Sitter scaling

.. math::

   a\propto t^{2/3},\qquad \rho=\varrho/a^3,\qquad
   u=Hax+v/a,\qquad p=\tilde p/a^{5}.

The test deliberately has no perturbation gravity; density-contrast gravity
will be introduced in a later cosmological phase.

Cosmological metadata is written to the HDF5 ``Header``. The shared example
helper ``example_utils.snapshot_physical_fields`` converts these fields back
to physical radius, density, velocity, and temperature using that metadata.

Run with::

   python einstein_de_sitter_homogeneous1d.py
