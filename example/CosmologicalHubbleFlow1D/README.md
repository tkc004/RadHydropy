# Cosmological Hubble flow

This example initializes a homogeneous gas with the Hubble velocity field and
checks that the expansion follows the cosmological analytic solution.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/CosmologicalHubbleFlow1D`
4. `python cosmological_hubble_flow1d.py`

The runner loads `cosmological_hubble_flow1d.yaml` by default and stores its
diagnostics according to that YAML file.
