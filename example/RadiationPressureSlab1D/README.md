# Radiation-pressure slab

This example evolves a planar gas slab acted on by radiation pressure and
records the resulting density, velocity, and radiation-force diagnostics.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/RadiationPressureSlab1D`
4. `python radiation_pressure_slab1d.py --config radiation_pressure_slab1d.yaml`

The output files are placed in the directory configured in the YAML file.
