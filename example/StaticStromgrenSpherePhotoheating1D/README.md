# Static Stromgren sphere with photoheating

This example computes a static Stromgren sphere while evolving photoheating
and hydrogen chemistry in the surrounding gas.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/StaticStromgrenSpherePhotoheating1D`
4. `python static_stromgren_sphere_photoheating1d.py --config static_stromgren_sphere_photoheating1d.yaml`

For the C2-Ray variant use
`--config static_stromgren_sphere_photoheating1d_c2ray.yaml`. Outputs follow
the selected configuration.
