# Passive gas angular momentum

This example checks angular-momentum advection for a passive gas component,
with a hydrodynamic control run available for comparison.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/PassiveGasAngularMomentum1D`
4. `python passive_gas_angular_momentum1d.py --config passive_gas_angular_momentum1d.yaml`
5. Optionally run the control with `python gas_hydro_control1d.py --config gas_hydro_control1d.yaml`.

Both runs write results to the output locations in their respective YAML files.
