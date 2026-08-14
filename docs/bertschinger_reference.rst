BertschingerReference
=====================

``example/BertschingerReference`` generates a standalone collisionless
reference solution for the radial, self-similar secondary-infall problem in an
Einstein--de Sitter universe. It is intended as an analytic/reference-profile
generator for validating cosmological dark-matter shell dynamics before
coupling the solution to the hydrodynamic solver.

The current page describes the collisionless stage of the benchmark. The
example also integrates Bertschinger (1985) Eq. (4.1), which is the
dimensionless collisionless shell equation. There are no gas variables in this
ODE. The adiabatic gas solution, including its accretion shock, is a separate
extension.

Running the example
-------------------

Run it from its own directory so that the local ``tools.py`` import and output
paths resolve correctly:

.. code-block:: bash

   cd example/BertschingerReference
   python bertschinger_reference.py

The default configuration is
``example/BertschingerReference/bertschinger_reference.yaml``. It writes:

* ``BertschingerReference.hdf5`` — dimensionless similarity profiles and
  reference metadata;
* ``BertschingerReference.jpg`` — density and velocity profiles in similarity
  coordinates.
* ``BertschingerEq41XiLambda.jpg`` — the Eq. (4.1) shell trajectory with
  similarity time ``xi`` on the x-axis and ``lambda`` on the y-axis.

Physical setup
--------------

The benchmark uses the Bertschinger ``epsilon=1`` scale-free perturbation in
an Einstein--de Sitter background. For this case the turnaround radius obeys

.. math::

   r_{\rm ta}(t) \propto t^{8/9}.

Meaning of ``epsilon``
~~~~~~~~~~~~~~~~~~~~~~

The similarity parameter ``epsilon`` describes the initial perturbation, not
the gas equation of state. It is defined by

.. math::

   {\delta M\over M} \propto M^{-\epsilon}.

Because the homogeneous background has ``M`` proportional to ``r^3``, the
``epsilon=1`` case has

.. math::

   {\delta M\over M} \propto M^{-1} \propto r^{-3},
   \qquad \delta M = \mathrm{constant}.

Thus the example represents the perturbation as a constant excess mass added
to the homogeneous shell background. The turnaround-radius exponent is

.. math::

   r_{\rm ta}\propto t^{\xi},
   \qquad
   \xi={2\over3}\left(1+{1\over3\epsilon}\right),

which gives ``xi=8/9`` when ``epsilon=1``. This ``epsilon`` must not be
confused with the gas adiabatic index ``gamma``. In the usual Bertschinger
gas problem, the two parameters are ``epsilon=1`` and ``gamma=5/3``:
``epsilon`` sets the initial perturbation and similarity scaling, while
``gamma`` sets the gas thermodynamics.

The initial perturbation is represented by a constant excess mass,

.. math::

   \Delta M = \mathrm{constant},
   \qquad
   {\Delta M\over M_{\rm bg}(<r)} \propto M_{\rm bg}^{-1}.

This is the scale-free ``epsilon=1`` initial condition. The shell masses sample
the homogeneous background, while the constant perturbation mass is supplied
as a central fixed excess. The initial radial velocity is the growing-mode
velocity associated with the enclosed perturbation:

.. math::

   v_{\rm pec}(r) = -{a^2 H\over3}
   {\Delta M\over M_{\rm bg}(<r)}r.

The default setup uses cold, radial shells with zero angular momentum. Shell
crossing is allowed and shell records are resorted by radius after each drift.

Similarity variables
--------------------

At an output time, the shell state is converted to physical radius and
velocity. The turnaround radius is found by interpolating the first shell
interface where the physical radial velocity changes sign. The dimensionless
variables are

.. math::

   \lambda = {r\over r_{\rm ta}},
   \qquad
   V(\lambda) = {v\over r_{\rm ta}/t},
   \qquad
   D(\lambda) = {\rho\over\rho_b(t)}.

The enclosed-mass profile is normalized as

.. math::

   M(\lambda) = {M(<r)\over
   (4\pi/3)\rho_b(t)r_{\rm ta}^3}.

The code deposits shell mass into logarithmic ``lambda`` bins. Density is
computed from shell mass divided by the corresponding spherical bin volume;
velocity is mass-weighted within each bin. Empty bins have undefined velocity
and are stored as ``NaN``.

Cosmological shell equation
---------------------------

The shells evolve in supercomoving time ``tau``. With comoving radius ``x``,

.. math::

   {d^2x_i\over d\tau^2} =
   -{G a\over (x_i+a_{\rm soft})^2}
   \left[M_{\rm DM}(<x_i)+\Delta M
   -M_{\rm bg}(<x_i)\right].

The background mass is subtracted so a homogeneous Einstein--de Sitter shell
distribution has zero peculiar acceleration. The shell mass is included in
the enclosed mass using the half-shell convention at coincident radii. The
constant perturbation mass is added to the live shell mass for this reference
problem. This distinction is important: the existing fixed-enclosed-mass
orbit examples use fixed mass as a replacement background, while this
scale-free perturbation uses it as an additional mass component.

Eq. (4.1) ODE reference
-----------------------

The example directly solves Bertschinger's collisionless shell equation,

.. math::

   {d^2\lambda\over d\xi^2}+{7\over9}{d\lambda\over d\xi}
   -{8\over81}\lambda=-{2\over9}{\mathcal M(\lambda)\over\lambda^2},

with ``lambda(0)=1`` and ``lambda'(0)=-8/9``. The original equation has
``j=0`` and the example sets ``ode_angular_momentum: 0``. The mass closure is

.. math::

   \mathcal M(\lambda)=\int_0^{\infty}
   \exp(2s\xi'/3)\,\mathcal H[\lambda-\lambda(\xi')]\,d\xi',

where ``s`` is configurable as ``ode_similarity_exponent``. The mass in this
equation is normalized by the Einstein--de Sitter background mass inside
``r_ta``. Therefore, before the first centre passage,

.. math::

   \mathcal M(\lambda(\xi))={9\pi^2\over16}
   \exp[-(2s/3)\xi],

so ``M(1)=9*pi**2/16``. The solver then continues through shell crossings.
Each monotonic phase-space branch is retained and the enclosed mass is
reconstructed with the full alternating crossing sum. Near the singular
centre, integration stops at ``ode_centre_match_lambda`` and starts the
outgoing branch with the finite ``ode_centre_matching_velocity``. This
controlled asymptotic matching avoids reflecting the divergent finite-cutoff
velocity. Both parameters are written to the HDF5 header. The ODE output is
``BertschingerEq41ODE.hdf5``.

Output format
-------------

``BertschingerReference.hdf5`` contains the datasets:

``lambda``
   Logarithmic similarity-radius bin centers.
``density``
   Density divided by the physical Einstein--de Sitter background density.
``velocity``
   Physical radial velocity divided by ``r_ta / t``. Empty bins are ``NaN``.
``mass``
   Enclosed mass divided by ``(4*pi/3) rho_b r_ta^3``.

The ``Header`` group records ``Solution``, ``SimilarityEpsilon``,
``TurnaroundExponent``, ``CosmologyType``, the cosmology reference parameters,
the initial and final cosmic times, the perturbation mass, and the measured
turnaround radius.

Interpretation and limitations
------------------------------

The solution is a numerical shell reference after the similarity reduction,
not a closed-form density formula. The collisionless solution contains
multistream regions and caustics. Consequently, the density profile depends
on shell number, radial domain, bin count, softening, and time-step size near
shell crossings.

For convergence studies, increase ``number_of_shells``, reduce
``supercomoving_timestep``, and compare profiles at multiple output times in
``lambda`` rather than at a fixed physical radius. The outer radius must be
large enough that the turnaround region is not truncated, while the inner
radius must remain positive because the shell representation cannot place a
live shell exactly at the origin.

The example validates the collisionless radial reference and the
self-consistent pre-centre Eq. (4.1) trajectory. It does not solve the
separate adiabatic gas similarity equations or apply a Rankine--Hugoniot
shock.
