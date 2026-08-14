Bertschinger Collisional Gas Reference
======================================

This is the first, standalone stage of the Bertschinger (1985) collisional
gas benchmark.  It does not import or call RadHydropy.

The example solves the dimensionless ``epsilon=1`` similarity equations for
``gamma=5/3`` in two regions:

* the cold, pressureless exterior, using the spherical-collapse parametric
  solution;
* the shocked interior, by integrating the similarity ODEs inward.

It shoots on ``lambda_s`` using the regular central asymptote, applies the
strong-shock Rankine--Hugoniot conditions, and returns the matched profiles.
The resulting shock position is approximately ``lambda_s=0.33897694``.
The shock position remains an argument for controlled comparisons.

The generated ``BertschingerGasReference.jpg`` plots the dimensionless
density ``D``, velocity ``V``, pressure ``P``, and enclosed mass ``M`` as
functions of ``lambda``.  The dotted vertical line marks the shock.

.. figure:: BertschingerGasReference.jpg
   :alt: Bertschinger dimensionless collisional gas profiles
   :width: 95%

   Dimensionless exterior and shock-matched interior profiles.

Run it with::

   cd example/BertschingerGasReference
   python bertschinger_gas.py

RadHydropy comparison
---------------------

Run the finite-volume comparison with::

   cd example/BertschingerGasReference
   python bertschinger_gas_radhydropy_comparison.py

This uses ``Rsim`` with spherical gas self-gravity and Einstein--de Sitter
supercomoving expansion.  The initial state is the cold scale-free
perturbation ``Delta M/M = A/r^3``; it evolves to cosmic time
``t=4452.4``.
The 800 kpc domain keeps the outer boundary outside the turnaround
radius.
The script writes the RadHydropy snapshots, separate initial-condition and
final comparison figures, and an RMS-error report.
The central ideal similarity solution has divergent density, so the numerical
comparison is most meaningful around the shock and in the outer profile.

.. figure:: BertschingerGasReference_RadHydroComparison.jpg
   :alt: RadHydropy comparison with the Bertschinger gas similarity solution
   :width: 95%

   RadHydropy profiles compared with the standalone dimensionless solution.

The initial condition in the same similarity units is shown separately:

.. figure:: BertschingerGasReference_RadHydroInitialCondition.jpg
   :alt: RadHydropy initial condition in Bertschinger similarity units
   :width: 95%

   RadHydropy initial condition compared with the standalone solution.
