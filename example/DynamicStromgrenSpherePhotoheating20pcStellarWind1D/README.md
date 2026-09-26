# 20 pc Stromgren sphere with stellar wind

This example follows a photoheated H II region driven by a central stellar
wind, including hydrogen chemistry and the wind shock in a spherical domain.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/DynamicStromgrenSpherePhotoheating20pcStellarWind1D`
4. `python dynamic_stromgren_sphere_photoheating20pc_stellar_wind1d.py --config dynamic_stromgren_sphere_photoheating20pc_stellar_wind1d.yaml`
5. Optionally write radial CSV profiles with `python write_snapshot_radial_profiles.py`.

The short smoke configuration is
`dynamic_stromgren_sphere_photoheating20pc_stellar_wind1d_smoke.yaml`.
