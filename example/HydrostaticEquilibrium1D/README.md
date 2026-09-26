# Cartesian hydrostatic equilibrium

This example tests whether a one-dimensional gas remains hydrostatic under a
prescribed gravitational acceleration and pressure gradient.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/HydrostaticEquilibrium1D`
4. `python hydrostatic_equilibrium1d.py --config hydrostatic_equilibrium1d.yaml`

The HDF5 output and diagnostic figure are written to the configured directory.
