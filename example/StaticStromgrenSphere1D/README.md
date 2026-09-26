# Static Stromgren sphere

This example computes the static ionization structure around a source in a
spherical hydrogen cloud and compares it with the analytic Stromgren radius.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/StaticStromgrenSphere1D`
4. `python static_stromgren_sphere1d.py --config static_stromgren_sphere1d.yaml`

The configured ionization profile and comparison plot are written to the
configured output path.
