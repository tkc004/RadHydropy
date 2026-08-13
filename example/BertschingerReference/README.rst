Bertschinger Collisionless Reference
====================================

This example generates the radial collisionless similarity reference for the
Einstein--de Sitter, ``epsilon=1`` secondary-infall problem. For this case

.. math::

   {\Delta M\over M}\propto M^{-1},\qquad
   r_{ta}\propto t^{8/9}.

The implementation uses cold radial shells, a constant central perturbation
mass, cosmological background subtraction, shell crossing, and similarity
coordinates

.. math::

   \lambda={r\over r_{ta}},\qquad
   V={v\over r_{ta}/t},\qquad
   D={\rho\over\rho_b}.

It writes ``BertschingerReference.hdf5`` and
``BertschingerReference.jpg``. This is the collisionless reference stage;
the adiabatic gas shock solution is a separate follow-up problem.

Run with::

   python bertschinger_reference.py
