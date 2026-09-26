# 20 pc Stromgren sphere with radiation pressure

This example combines dynamic photoheating and hydrogen chemistry with the
radiation-pressure force from a central source in a 20 pc spherical domain.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/DynamicStromgrenSpherePhotoheating20pcRadiationPressure1D`
4. `python dynamic_stromgren_sphere_photoheating20pc_radiation_pressure1d.py --config dynamic_stromgren_sphere_photoheating20pc_radiation_pressure1d.yaml`
5. Optionally compare the gas-energy budget with `python compare_total_gas_energy.py`.

The smoke configuration is available as
`dynamic_stromgren_sphere_photoheating20pc_radiation_pressure1d_smoke.yaml`.
