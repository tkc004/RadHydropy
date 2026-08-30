Gas Centrifugal Hydro Expansion Benchmark
=========================================

This benchmark tests centrifugal work together with spherical Eulerian hydro
advection.  The initial gas sphere uses the low-temperature isothermal closure
to remain pressureless numerically, and rotates above circular support,

.. math::

   j(r) = 1.2\sqrt{GMr}.

Consequently centrifugal acceleration exceeds central gravity and the gas
expands.  The example writes an HDF5 IC, runs the normal ``Rsim`` hydro path,
and compares the saved state with pressureless shell ODEs mapped to the fixed
Eulerian grid.

The diagnostic plot compares radial velocity, density, ``J/M``, the local
thermal-to-dynamical pressure scale, and the gas-plus-central-gravity energy
audit.  The source-only benchmark remains the exact local work test; this
example adds hydro fluxing, shell mixing, global mass conservation, and global
energy conservation using a closed reflecting domain.

The run also reports the discrete gravity audit
``W_gravity + Delta(sum(M_i Phi_i))``.  Here ``Phi_i = -GM/r_i`` is evaluated
at the spherical cell centers and ``M_i = rho_i V_i`` includes the cell volume.
This diagnostic is deliberately separate from the conserved gas energy: it
measures the residual introduced by cell-centered Eulerian potential
bookkeeping before any potential-energy flux correction is enabled.

Run with::

   python gas_centrifugal_hydro_expansion1d.py

Run the 32/64/128-cell convergence study with::

   python convergence.py
