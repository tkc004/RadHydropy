Gas Centrifugal Cosmological Orbit Benchmark
=============================================

This benchmark integrates an eccentric orbit in Einstein--de Sitter
supercomoving coordinates.  For comoving radius ``x`` and supercomoving
velocity ``v`` it uses

.. math::

   \frac{dx}{d\tau}=v,qquad
   \frac{dv}{d\tau}=-\frac{G a(\tau)M}{x^2}+\frac{j^2}{x^3}.

The numerical RK4 trajectory is compared with a high-accuracy ``solve_ivp``
solution.  It also verifies that

.. math::

   j=xv_{\phi,\rm sc}=r v_{\phi,\rm phys}

is invariant under the physical/supercomoving conversion.  Because the
background scale factor evolves, physical orbital energy is not required to
be constant; trajectory error and angular-momentum invariance are the primary
checks.

The example also writes an HDF5 initial condition and runs a real
supercomoving spherical ``Rsim`` simulation.  Its saved
``SpecificAngularMomentum`` field is checked against the initialized signed
value; the analytic trajectory remains the moving-shell reference because
Eulerian cell coordinates are fixed.

The saved-simulation figure maps an ensemble of analytic shell solutions back
onto the fixed Eulerian cell centers, allowing the saved ``Rsim`` velocity
profile to be compared directly with the ODE prediction.

The simulation is deliberately configured as a validated cold/pressureless
case: the YAML sets ``temperature: 1.e-8``, enables dual energy, and disables
the positivity flux limiter so the cold rotational state is not artificially
heated by the limiter.

Run with::

   python gas_centrifugal_cosmological_orbit1d.py
