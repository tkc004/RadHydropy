# Hydrogen recombination

This source-only test follows time-dependent hydrogen recombination and checks
the numerical ionization fraction against the analytic recombination curve.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/HydrogenRecombination1D`
4. `python hydrogen_recombination1d.py --config hydrogen_recombination1d.yaml`

The script writes the configured comparison data and figure.
