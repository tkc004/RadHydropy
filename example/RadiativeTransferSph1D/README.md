# Spherical radiative transfer

This example tests spherical radiative transfer through a hydrogen medium,
including the standard and C2-Ray temporal schemes and an analytic reference.

1. `cd RadHydropy`
2. `python -m pip install -e ".[test,docs]"`
3. `cd example/RadiativeTransferSph1D`
4. `python radiative_transfer_sph1d.py --config radiative_transfer_sph1d.yaml`

For the C2-Ray variant use
`--config radiative_transfer_sph1d_c2ray.yaml`. Outputs follow the YAML paths.
