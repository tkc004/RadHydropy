# Hydrogen photoheating

This source-only example tests thermal evolution from hydrogen photoheating
and compares the numerical result with the supplied reference calculation.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/HydrogenPhotoheating1D`
4. `python hydrogen_photoheating1d.py --config hydrogen_photoheating1d.yaml`

The runner writes its diagnostic plot and data according to the YAML settings.
