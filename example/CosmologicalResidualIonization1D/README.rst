Cosmological residual ionization
================================

This homogeneous source-only example tests the post-recombination hydrogen
chemistry from ``z=100`` to ``z=4.9``.  The electron fraction is initialized
to a residual ``x_e=2e-4`` and evolves with RadHydropy's hydrogen recombination
equation.  Compton heating and atomic cooling are enabled, while the radiation
field is absent: there is no photoionization or reionization prescription.

The density follows the cosmological mean, ``n_H=n_H,0 a^-3``, and the gas
temperature is evolved with adiabatic expansion plus the same thermal source
rate.  The output figure contains ``x_e(z)`` and ``T(z)``; the NPZ file stores
the numerical history for comparison with a CLASS residual-electron-fraction
table or digitization.  The supplied CLASS history should only be compared
after ``z=100`` because this example intentionally omits reionization.

Run with::

   python cosmological_residual_ionization1d.py
Running from a clean checkout
----------------------------

From the repository root, install RadHydropy and the example dependencies::

   cd RadHydropy
   python -m pip install -e ".[test,docs]"

Then change into this example directory before running the command shown above::

   cd example/CosmologicalResidualIonization1D

If a command above begins with ``python example/``, run that command from
the repository root instead of changing into this directory.
