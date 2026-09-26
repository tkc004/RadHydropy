# H II region expansion

These early- and late-phase STARBENCH examples model D-type expansion of a
spherical H II region with piecewise-isothermal neutral and ionized gas.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/HIIRegionExpansion1D`
4. `python early_hii_region_expansion1d.py --config early_hii_region_expansion1d.yaml`
5. `python late_hii_region_expansion1d.py --config late_hii_region_expansion1d.yaml`

For the C2-Ray variants substitute the corresponding `*_c2ray.yaml` file.
Each run writes HDF5 snapshots, ionization-front history, and density plots to
its configured output directory.
