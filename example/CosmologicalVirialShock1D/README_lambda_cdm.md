# LambdaCDM entry points

The original LambdaCDM implementation remains preserved in
`../CosmologicalVirialShock1DLambdaCDM/`. The files in this directory are
additive copies and entry points; they do not modify those originals.

Run them from this directory or from the repository root:

```bash
python example/CosmologicalVirialShock1D/generate_cosmological_correlation_ic_lambda_cdm.py
python example/CosmologicalVirialShock1D/cosmological_dark_matter_correlation_z100_lambda_cdm.py
python example/CosmologicalVirialShock1D/cosmological_gas_correlation_z100_lambda_cdm.py
python example/CosmologicalVirialShock1D/cosmological_virial_shock1d_lambda_cdm.py
```

The preserved LCDM configurations use an initial scale factor of `1/101` and
the matched final cosmic time `1.832399728`, corresponding to the EdS
endpoint at the same final scale factor. Their output directories remain
separate from the EdS outputs.

The additive local YAML variants are:

- `cosmological_dark_matter_correlation_z100_lambda_cdm.yaml`
- `cosmological_gas_correlation_z100_lambda_cdm.yaml`
- `cosmological_gas_correlation_z100_compton_atomic_lambda_cdm.yaml`

The copied initial conditions are stored in:

- `outputs_correlation_lcdm/InitialCondition.hdf5`
- `outputs_correlation_gas_lcdm/InitialCondition.hdf5`
- `outputs_correlation_gas_compton_atomic_lcdm/InitialCondition.hdf5`
