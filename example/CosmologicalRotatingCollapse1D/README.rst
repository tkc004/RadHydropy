Cosmological rotating collapse in an Einstein--de Sitter universe
==================================================================

This example compares three one-dimensional spherical gas-collapse runs:

* ``nonrotating``: ``j = 0``;
* ``moderate``: finite shell angular momentum;
* ``high``: larger shell angular momentum.

The initial profile is regular at the origin.  Since the enclosed mass scales
as ``x**3`` near the origin, the initialization

.. math::

   j(x) = f\sqrt{G M(<x) a x}

behaves as ``x**2``.  The rotation factor ``f`` controls the centrifugal
barrier while preserving signed shell angular momentum.

This is a spherical-collapse benchmark, not a literal disk-formation model:
the one-dimensional solver cannot represent disk thickness, vertical settling,
or non-axisymmetric angular-momentum transport.  It tests the ingredients a
future multidimensional disk calculation will require: cosmological variable
conversion, conservative ``J`` transport, centrifugal support, and rotational
energy accounting.

The gas angular-momentum transport uses a local flux-corrected transport
(FCT) construction.  The donor-cell flux, ``F_J = j_donor F_M``, is the
positivity-safe low-order flux.  The MUSCL flux supplies an antidiffusive
correction with a face-by-face coefficient between zero and one.  Only faces
whose adjacent cells would violate local ``J/M`` bounds are reduced toward the
donor flux; the limiter is not applied globally.  Rotational-energy fluxes
use the same limited face value of ``j``.

Run it with::

   python cosmological_rotating_collapse1d.py

Each case writes ``InitialCondition.hdf5`` and ``Output_final.hdf5`` below
``outputs/<case>/``.  The run checks that final compression decreases and
centrifugal support increases from nonrotating to moderate to high rotation.

It also integrates a pressureless physical shell ODE for the same initial
enclosed masses and angular momenta,

.. math::

   \\ddot r = -\\frac{G M(<r)}{r^2} + \\frac{j^2}{r^3},

and writes ``CosmologicalRotatingCollapse1D_shell_ode.jpg``.  The ODE curves
are a ballistic reference: the gas calculation includes pressure and
Eulerian numerical transport, so agreement is expected only in the cold,
weak-pressure limit.  The analytic centrifugal barrier is
``r_cent = j**2 / (G M)``.

For a density comparison, the same pressureless shell ODE is integrated for
the shell boundaries and remapped conservatively onto the fixed Eulerian
cells by spherical-volume overlap.  This avoids pointwise interpolation of
the density and writes ``CosmologicalRotatingCollapse1D_density_comparison.jpg``.
The comparison is qualitative because the simulated gas includes pressure,
while the reference is pressureless.
