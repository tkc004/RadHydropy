# Dynamic Stromgren sphere with photoheating

This spherical radiation-hydrodynamics example follows the expansion and
photoheating of an H II region around a source, including hydrogen chemistry.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/DynamicStromgrenSpherePhotoheating1D`
4. `python dynamic_stromgren_sphere_photoheating1d.py --config dynamic_stromgren_sphere_photoheating1d.yaml`

To run the C2-Ray temporal variant, use
`--config dynamic_stromgren_sphere_photoheating1d_c2ray.yaml`. Snapshots and
diagnostics go to the configured output directory.
