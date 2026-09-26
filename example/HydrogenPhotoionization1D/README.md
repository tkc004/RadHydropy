# Hydrogen photoionization

This example evolves hydrogen ionization under an imposed radiation field and
compares the numerical ionization history with an analytic solution.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/HydrogenPhotoionization1D`
4. `python hydrogen_photoionization1d.py --config hydrogen_photoionization1d.yaml`

The generated diagnostics are stored at the output paths in the YAML file.
