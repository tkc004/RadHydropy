# Cosmological adiabatic temperature evolution

This source-only test evolves gas temperature under cosmological expansion and
adiabatic cooling in an Einstein–de Sitter background.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/CosmologicalAdiabaticTemperature1D`
4. `python cosmological_adiabatic_temperature1d.py`

The script uses `cosmological_adiabatic_temperature1d.yaml` by default and
writes its diagnostic data and plot as configured there.
