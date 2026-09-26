# 1D Sedov–Taylor blast wave

This example evolves a planar point explosion and compares the numerical blast
profile with the Sedov–Taylor similarity solution.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/SedovTaylor1D`
4. `python sedov_taylor1d.py --config SedovTaylor1D.yaml`

The configured HDF5 snapshots and comparison plot are written to the output
directory in the YAML file.
