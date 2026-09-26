# Sod shock tube

This example solves the classic one-dimensional Sod shock tube and compares
the numerical density, pressure, and velocity profiles with the analytic
Riemann solution.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/SodShock1D`
4. `python sodshock1d.py --config sodshock1d.yaml`

To select the HLLC solver explicitly, append `--riemann-solver HLLC`. The
short check is `python sodshock1d.py --config sodshock1d_smoke.yaml`.
