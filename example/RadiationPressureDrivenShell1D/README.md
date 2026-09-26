# Radiation-pressure-driven shell

This example integrates a thin-shell ordinary differential equation for a shell
accelerated by radiation pressure and compares the trajectory with the model
parameters in the YAML file.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/RadiationPressureDrivenShell1D`
4. `python thin_shell_ode.py`

The script writes its configured shell trajectory and figures in this example's
output location.
