# Spherical advection

This example advects a spherical top-hat profile and checks the numerical
solution against the analytic spherical-advection solution.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/AdvectionSph1D`
4. `python advection_sph1d.py --config advectionSph1d.yaml`

For the short CI-style case, run
`python advection_sph1d.py --config advectionSph1d_smoke.yaml`. Outputs are
written where the selected YAML configuration specifies.
