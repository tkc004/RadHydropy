Uniform-Sphere Self-Gravity Diagnostic
======================================

This diagnostic builds a spherical gas sphere with constant density and
compares the numerical self-gravity field with the analytic result

.. math::

   g(r) = -\frac{4\pi}{3}G\rho_0 r.

The example does not advance the fluid. It uses the normal initial-condition
HDF5 startup flow, evaluates the gravity field from the loaded density, and
saves ``SelfGravityUniformSphere1D.jpg``. The origin-containing cell is
excluded from the relative-error check because spherical source updates impose
zero acceleration there by symmetry.

Run it from this directory with::

   python self_gravity_uniform_sphere1d.py \
       --config self_gravity_uniform_sphere1d.yaml
