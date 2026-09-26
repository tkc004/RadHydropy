# Spherical Sedov–Taylor blast wave

This example evolves a spherical point explosion and compares its density,
velocity, and pressure profiles with the analytic Sedov–Taylor solution.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/SedovTaylorSph1d`
4. `python sedov_taylor_sph1d.py --config SedovTaylorSph1d.yaml`

Snapshots and the comparison figure are written as configured in the YAML.
