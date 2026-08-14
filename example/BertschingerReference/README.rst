Bertschinger Collisionless Reference
====================================

This example generates the radial collisionless similarity reference for the
Einstein--de Sitter, ``epsilon=1`` secondary-infall problem. For this case

.. math::

   {\Delta M\over M}\propto M^{-1},\qquad
   r_{ta}\propto t^{8/9}.

Here ``epsilon`` is the perturbation-slope parameter,
``delta M/M ~ M**(-epsilon)``, not the gas adiabatic index. Thus ``epsilon=1``
means a constant excess mass, ``delta M = constant``. In the usual gas
extension, ``gamma=5/3`` is a separate thermodynamic parameter.

The implementation uses cold radial shells, a constant central perturbation
mass, cosmological background subtraction, shell crossing, and similarity
coordinates

.. math::

   \lambda={r\over r_{ta}},\qquad
   V={v\over r_{ta}/t},\qquad
   D={\rho\over\rho_b}.

It writes ``BertschingerReference.hdf5`` and
``BertschingerReference.jpg``. The Eq. (4.1) ODE also writes
``BertschingerEq41XiLambda.jpg``, plotting ``xi`` on the x-axis against
``lambda`` on the y-axis. This is
the collisionless reference stage;
the adiabatic gas shock solution is a separate follow-up problem.

Eq. 4.1 shell ODE
------------------

The example also integrates Bertschinger (1985), Eq. (4.1), which is the
collisionless dark-matter shell equation, not a gas equation:

.. math::

   {d^2\lambda\over d\xi^2}+{7\over9}{d\lambda\over d\xi}
   -{8\over81}\lambda=-{2\over9}{M(\lambda)\over\lambda^2}.

The initial conditions are exactly ``lambda(0)=1`` and
``lambda'(0)=-8/9``. The solver writes ``BertschingerEq41ODE.hdf5`` with
``xi``, ``lambda``, ``lambda_prime``, and ``mass`` datasets. The current ODE
test uses Bertschinger's normalized first-stream mass closure, sets
``ode_angular_momentum`` to zero, and stops at the first ``lambda=0`` event.
The similarity exponent is configured by ``ode_similarity_exponent`` and the
mass normalization is ``9*pi**2/16`` when the force coefficient is ``2/9``.

The trajectory is one representative shell before its first centre passage;
no post-centre reflection is included in this reference.

Run with::

   python bertschinger_reference.py
