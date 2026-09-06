import numpy as np

from radhydropy.cosmology import EinsteinDeSitter
from radhydropy.cosmology_state import (
    ProperCgsState,
    SupercomovingState,
    proper_to_supercomoving_time,
    supercomoving_to_cosmic_time,
    to_proper_state,
    to_supercomoving_state,
    validate_supercomoving_contract,
)
from radhydropy.units import CodeUnits
from radhydropy.cosmology_state_io import (
    read_supercomoving_state_hdf5,
    write_supercomoving_state_hdf5,
)


def _units():
    return CodeUnits.from_mapping({
        "UnitMass_in_cgs": 1.0e33,
        "UnitLength_in_cgs": 1.0e18,
        "UnitVelocity_in_cgs": 1.0e5,
        "UnitCurrent_in_cgs": 1.0,
        "UnitTemp_in_cgs": 1.0,
    })


def test_supercomoving_proper_round_trip():
    units = _units()
    cosmology = EinsteinDeSitter.from_code_units(units, t_ref=1.0)
    tau = float(cosmology.supercomoving_time(2.0))
    state = SupercomovingState(
        x_comoving_code=np.array([0.5, 2.0]),
        rho_comoving_code=np.array([3.0, 4.0]),
        vel_supercomoving_code=np.array([-0.2, 0.7]),
        pre_supercomoving_code=np.array([1.0e-3, 2.0e-3]),
        temp_supercomoving_code=np.array([10.0, 20.0]),
        tau_supercomoving_code=tau,
    )

    proper = to_proper_state(state, cosmology, units, gamma=5.0 / 3.0)
    restored = to_supercomoving_state(
        proper, cosmology, units, tau_supercomoving_code=tau, gamma=5.0 / 3.0
    )

    np.testing.assert_allclose(restored.x_comoving_code, state.x_comoving_code)
    np.testing.assert_allclose(
        restored.rho_comoving_code, state.rho_comoving_code
    )
    np.testing.assert_allclose(
        restored.vel_supercomoving_code, state.vel_supercomoving_code
    )
    np.testing.assert_allclose(
        restored.pre_supercomoving_code, state.pre_supercomoving_code
    )
    np.testing.assert_allclose(
        restored.temp_supercomoving_code, state.temp_supercomoving_code
    )


def test_cosmic_and_supercomoving_time_round_trip():
    units = _units()
    cosmology = EinsteinDeSitter.from_code_units(units, t_ref=1.0)
    cosmic_time = 2.5
    tau = proper_to_supercomoving_time(cosmology, cosmic_time)
    restored = supercomoving_to_cosmic_time(cosmology, tau)
    np.testing.assert_allclose(restored, cosmic_time)


def test_proper_state_rejects_nonpositive_cosmic_time():
    try:
        ProperCgsState(
            x_proper_cgs_cm=np.ones(1),
            rho_proper_cgs_g_cm3=np.ones(1),
            vel_peculiar_proper_cgs_cm_s=np.zeros(1),
            pre_proper_cgs_erg_cm3=np.ones(1),
            temp_proper_cgs_K=np.ones(1),
            time_cosmic_cgs_s=0.0,
        )
    except ValueError:
        return
    raise AssertionError("nonpositive proper cosmic time was accepted")


def test_supercomoving_contract_rejects_legacy_velocity_name():
    from types import SimpleNamespace

    par = SimpleNamespace(
        coordinate_frame="comoving",
        time_coordinate="supercomoving",
        velocity_representation="supercomoving_peculiar",
        time_cosmic_code=1.0,
        tau_supercomoving_code=0.0,
    )
    mesh = SimpleNamespace(
        x_comoving_code=1.0,
        boundary_comoving_code=1.0,
        width_comoving_code=1.0,
    )
    fluid = SimpleNamespace(
        rho_comoving_code=1.0,
        vel_supercomoving_code=1.0,
        pre_supercomoving_code=1.0,
        temp_supercomoving_code=1.0,
        tau_supercomoving_code=0.0,
        vel_code=1.0,
    )
    try:
        validate_supercomoving_contract(par, mesh, fluid)
    except ValueError as error:
        assert "legacy fluid.vel_code" in str(error)
        return
    raise AssertionError("legacy cosmological velocity name was accepted")


def test_canonical_hdf5_state_round_trip(tmp_path):
    state = SupercomovingState(
        x_comoving_code=np.array([0.5, 1.5]),
        rho_comoving_code=np.array([2.0, 3.0]),
        vel_supercomoving_code=np.array([-1.0, 0.25]),
        pre_supercomoving_code=np.array([0.1, 0.2]),
        temp_supercomoving_code=np.array([4.0e3, 5.0e3]),
        tau_supercomoving_code=0.75,
    )
    filename = tmp_path / "canonical.hdf5"
    write_supercomoving_state_hdf5(
        filename,
        state=state,
        boundary_comoving_code=np.array([0.0, 1.0, 2.0]),
        width_comoving_code=np.array([1.0, 1.0]),
        box_size_comoving_code=np.array([2.0]),
        code_units=_units(),
    )
    restored = read_supercomoving_state_hdf5(filename)
    np.testing.assert_allclose(
        restored.state.temp_supercomoving_code,
        state.temp_supercomoving_code,
    )
    import h5py

    with h5py.File(filename, "r") as handle:
        assert "tau_supercomoving_code" in handle["Header"]
        assert "time_code" not in handle["Header"]
        assert "vel_supercomoving_code" in handle["Data"]
        assert "temp_supercomoving_code" in handle["Data"]
        assert "vel_code" not in handle["Data"]
