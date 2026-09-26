# Spherical inflow

This example tests spherical inflow boundary conditions and compares the
evolution with the supplied analytic spherical solution.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/InflowSph1D`
4. `python inflow_sph1d.py --config InflowSph1d.yaml`

Snapshots and comparison diagnostics use the paths in the YAML file.
