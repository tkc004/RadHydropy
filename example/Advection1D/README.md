# 1D advection

This example advects a one-dimensional top-hat profile with the RadHydropy
hydrodynamics solver and compares saved snapshots with the initial profile.

Run it from a clean checkout as follows:

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/Advection1D`
4. `python advection1d.py --config advection1d.yaml`

The smoke run is `python advection1d.py --config advection1d_smoke.yaml`.
Snapshots and the configured plot are written to the output directory in the
selected YAML file.
