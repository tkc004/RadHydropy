# Stellar-wind bubble

This spherical example models the expansion of a wind-blown bubble and
compares the simulated shell with the analytic Weaver solution.

The metal configuration uses the CHIANTI CIE ion-fraction and cooling tables.
Download the external data before running that configuration. From the
RadHydropy project root:

```bash
git lfs install
git clone https://github.com/tkc004/RadhydropyData.git
cd RadhydropyData
git lfs pull
cp -R CHIANTI_11.0.2_database ../CHIANTI_11.0.2_database
cd ../RadHydropy
test -s ../CHIANTI_11.0.2_database/cooling_tables/chianti_cie_ion_fractions.h5
test -s ../CHIANTI_11.0.2_database/cooling_tables/chianti_cooling_table.h5
```

The cloned data repository must be beside the `RadHydropy` checkout so the
default CIE table paths resolve. If it is already cloned, run `git lfs pull`
from `../RadhydropyData` instead of cloning it again.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/StellarWindBubble1D`
4. `python stellar_wind_bubble1d.py --config stellar_wind_bubble1d.yaml`

The no-metallicity variant uses
`stellar_wind_bubble1d_no_metal.yaml`. Snapshots and plots are written to the
configured output directory.
