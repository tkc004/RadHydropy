from pathlib import Path
import tempfile

import h5py

from example.HDF5UnitCosmologyRoundtrip1D.hdf5_unit_cosmology_roundtrip1d import run


def test_hdf5_unit_cosmology_roundtrip_example():
    with tempfile.TemporaryDirectory() as output_directory:
        results = run(output_directory=output_directory)

        assert set(results) == {
            "proper_cgs",
            "proper_astrophysical",
            "cosmological_astrophysical",
        }
        for case_name, output_filename in results.items():
            with h5py.File(output_filename, "r") as handle:
                header = handle["Header"]
                data = handle["Data"]
                assert "CodeUnits" in header.attrs
                assert "Gamma" in header.attrs
                assert "storage_unit" in data["rho_comoving_code" if "cosmological" in case_name else "rho_proper_code"].attrs
                if case_name == "cosmological_astrophysical":
                    assert "CosmologyType" in header.attrs
                    assert "ScaleFactor" in header.attrs
                    assert "HubbleParameterKmS_Mpc" in header.attrs
                    assert "rho_comoving_code" in data
                    assert "vel_supercomoving_code" in data
                else:
                    assert "CosmologyType" not in header.attrs
                    assert "rho_proper_code" in data
                    assert "vel_proper_code" in data
