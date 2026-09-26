# Planar inflow

This example tests a one-dimensional inflow boundary and the resulting gas
evolution in a planar domain.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/Inflow1D`
4. `python inflow1d.py --config Inflow1d.yaml`

The runner writes snapshots and diagnostics to the configured output directory.
