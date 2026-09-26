# Ballistic infall onto a spherical point mass

This example follows pressureless spherical shells falling toward a central
point mass and compares the numerical shell motion with the analytic result.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/BallisticInfallSphericalPointMass1D`
4. `python ballistic_infall_spherical_point_mass1d.py --config ballistic_infall_spherical_point_mass1d.yaml`

The runner creates its initial condition, advances to the configured output
times, and writes snapshots and figures under the configured output directory.
