from types import SimpleNamespace
from tests.parameter_fixtures import parameter_namespace

import h5py
import numpy as np
import pytest

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS
from radhydropy.thermo_networks.cie.cie_cooling import CIECoolingNetwork, _state
from radhydropy.thermo_networks.cie.cie_tables import CIETable
from radhydropy.units import CodeUnits


def make_cie_tables(tmp_path):
    """Create small analytic CIE tables for deterministic unit tests."""
    temperatures = np.array([4.0, 5.0, 6.0])
    densities = np.array([0.0, 1.0])
    metallicities = np.array([0.0, 1.0])
    atomic_number = np.array([1, 2, 6])

    # Every element has one electron in this synthetic table.  This makes the
    # electron-fraction expectation easy to calculate from the abundance file.
    fractions = np.zeros((3, 2, temperatures.size))
    fractions[:, 1, :] = 1.0
    ion_fraction_file = tmp_path / "ion_fractions.h5"
    with h5py.File(ion_fraction_file, "w") as table:
        table["log10_temperature_K"] = temperatures
        table["atomic_number"] = atomic_number
        table["ion_fraction"] = fractions

    # Lambda = 1e-23 * (1 + Z) * temperature_factor * density_factor.
    temperature_factor = np.array([1.0, 3.0, 9.0])[:, None]
    density_factor = np.array([1.0, 2.0])[None, :]
    cooling = np.empty((2, 3, 2))
    cooling[0] = 1.0e-23 * temperature_factor * density_factor
    cooling[1] = 2.0 * cooling[0]
    cooling_file = tmp_path / "cooling.h5"
    with h5py.File(cooling_file, "w") as table:
        table["log10_temperature_K"] = temperatures
        table["log10_electron_density_cm-3"] = densities
        table["metallicity_Zsun"] = metallicities
        table["cooling_erg_cm3_s"] = cooling

    abundance_file = tmp_path / "abundance.abund"
    abundance_file.write_text("1 12.0 H\n2 11.0 He\n6 8.0 C\n")
    return ion_fraction_file, cooling_file, abundance_file


def test_cie_table_interpolates_temperature_density_and_metallicity(tmp_path):
    paths = make_cie_tables(tmp_path)
    table = CIETable(*paths)

    # T=1e5, ne=10, Z=1: 1e-23 * 3 * 2 * 2.
    result = table.cooling_coefficient(1.0e5, 10.0, 1.0)
    assert np.isclose(result, 1.2e-22)

    # At Z=0.5 the implementation linearly interpolates Lambda in linear space.
    assert np.isclose(table.cooling_coefficient(1.0e5, 10.0, 0.5), 9.0e-23)
    assert table.cooling_coefficient(1.0e3, 1.0, 1.0) == 0.0
    assert table.cooling_coefficient(1.0e7, 1.0, 1.0) == 0.0

    with pytest.raises(ValueError, match="outside the cooling table range"):
        table.cooling_coefficient(1.0e5, 1.0, 2.0)


def test_cie_table_electron_fraction_scales_metals(tmp_path):
    paths = make_cie_tables(tmp_path)
    table = CIETable(*paths)

    # H contributes 1, He contributes 0.1, and C contributes 1e-4 at Z=1.
    assert np.isclose(table.electron_fraction(1.0e5, 1.0), 1.1001)
    assert np.isclose(table.electron_fraction(1.0e5, 0.0), 1.1)
    assert np.all(np.isfinite(table.electron_fraction([1.0e4, 1.0e6], 1.0)))


def make_network_state(paths, temperature=1.0e5, rho_code=1.0e-20, metallicity=1.0):
    par = parameter_namespace(
        cie_ion_fraction_table=str(paths[0]),
        cie_cooling_table=str(paths[1]),
        cie_abundance_file=str(paths[2]),
        metallicity=metallicity,
        hydrogen_mass_fraction=1.0,
        cooling_safety_factor=0.1,
    )
    nH = rho_code / PROTON_MASS_CGS
    return {
        "par": par,
        "metallicity": metallicity,
        "hydrogen_mass_fraction": 1.0,
        "rho_cgs_g_cm3": np.array([rho_code]),
        "temperature_cgs_K": np.array([temperature]),
        "specific_energy_cgs_erg_g": np.array([BOLTZMANN_CONSTANT_CGS * temperature / (2.0 / 3.0 * PROTON_MASS_CGS)]),
        "gamma": 5.0 / 3.0,
        "mu": np.array([1.0]),
        "volume_cgs_cm3": np.array([1.0]),
        "velocity_cgs_cm_s": np.array([0.0]),
        "nH": nH,
    }


def test_cie_thermal_rate_has_expected_sign_and_density_scaling(tmp_path):
    paths = make_cie_tables(tmp_path)
    network = CIECoolingNetwork()
    state = make_network_state(paths)
    rate = network.thermal_rate(state, None)

    assert rate.shape == (1,)
    assert rate[0] < 0.0
    assert np.all(np.isfinite(rate))

    denser_state = make_network_state(paths, rho_code=2.0e-20)
    denser_rate = network.thermal_rate(denser_state, None)
    assert abs(denser_rate[0]) > abs(rate[0])

    zero_metal_state = make_network_state(paths, metallicity=0.0)
    zero_metal_rate = network.thermal_rate(zero_metal_state, None)
    assert abs(rate[0]) > abs(zero_metal_rate[0])


def test_cie_apply_fast_subcycles_and_enforces_temperature_floor(tmp_path):
    paths = make_cie_tables(tmp_path)
    network = CIECoolingNetwork()
    code_units = CodeUnits.from_mapping(
        {
            "InternalUnitSystem": {
                "UnitMass_in_cgs": 1.0,
                "UnitLength_in_cgs": 1.0,
                "UnitVelocity_in_cgs": 1.0,
                "UnitCurrent_in_cgs": 1.0,
                "UnitTemp_in_cgs": 1.0,
            }
        }
    )
    temperature = 1.0e5
    rho = 1.0e-20
    gamma = 5.0 / 3.0
    specific_energy = BOLTZMANN_CONSTANT_CGS * temperature / (
        (gamma - 1.0) * PROTON_MASS_CGS
    )
    fluid = SimpleNamespace(
        rho_code=np.array([rho, rho]),
        vel_code=np.zeros(2),
        temp_code=np.array([temperature, temperature]),
        mu=np.ones(2),
        Energy_code=np.array([specific_energy * rho, specific_energy * rho]),
            pre_code=np.ones(2),
        eos=SimpleNamespace(gamma=gamma),
    )
    mesh = SimpleNamespace(vol=np.ones(2))
    par = parameter_namespace(
        CodeUnits=code_units,
        noghost=1,
        nogrid=0,
        cie_cooling=True,
        cie_ion_fraction_table=str(paths[0]),
        cie_cooling_table=str(paths[1]),
        cie_abundance_file=str(paths[2]),
        metallicity=1.0,
        hydrogen_mass_fraction=1.0,
        cooling_safety_factor=0.1,
        cooling_temperature_floor=2.0e4,
    )
    # Use one interior cell and one ghost cell; the source implementation
    # indexes [noghost:noghost+nogrid].
    par.nogrid = 1
    par.mesh.grid_cells = 1
    initial_energy = fluid.Energy_code[1]
    steps = network.apply_fast(1.0e8, mesh, fluid, par)

    assert steps > 1
    assert fluid.Energy_code[1] < initial_energy
    assert fluid.temp_code[1] >= par.cooling_temperature_floor * (1.0 - 1.0e-12)
    assert np.isfinite(fluid.Energy_code[1])
    assert np.isfinite(fluid.temp_code[1])


def test_cie_state_converts_supercomoving_hydro_fields_to_physical():
    code_units = CodeUnits.from_mapping(
        {
            "InternalUnitSystem": {
                "UnitMass_in_cgs": 1.0,
                "UnitLength_in_cgs": 1.0,
                "UnitVelocity_in_cgs": 1.0,
                "UnitCurrent_in_cgs": 1.0,
                "UnitTemp_in_cgs": 1.0,
            }
        }
    )
    scale_factor = 2.0
    gamma = 5.0 / 3.0
    physical_temperature = 100.0
    specific_internal = (
        BOLTZMANN_CONSTANT_CGS * physical_temperature
        / ((gamma - 1.0) * PROTON_MASS_CGS)
    )
    velocity_supercomoving = 3.0
    fluid = SimpleNamespace(
        rho_code=np.array([8.0]),
        vel_code=np.array([velocity_supercomoving]),
        temp_code=np.array([physical_temperature * scale_factor**2]),
        mu=np.ones(1),
        Mass_code=np.array([8.0]),
        Energy_code=np.array([
            8.0 * (specific_internal * scale_factor**2
                   + 0.5 * velocity_supercomoving**2)
        ]),
        eos=SimpleNamespace(gamma=gamma),
    )
    mesh = SimpleNamespace(vol=np.array([1.0]))
    par = parameter_namespace(
        CodeUnits=code_units,
        noghost=0,
        nogrid=1,
        supercomoving_coordinates=True,
        fluid_time=0.0,
        cosmology=SimpleNamespace(
            scale_factor_from_supercomoving=lambda _: scale_factor,
        ),
    )

    state = _state(mesh, fluid, par)

    np.testing.assert_allclose(state["rho_cgs_g_cm3"], [1.0])
    np.testing.assert_allclose(state["volume_cgs_cm3"], [8.0])
    np.testing.assert_allclose(state["velocity_cgs_cm_s"], [1.5])
    np.testing.assert_allclose(state["temperature_cgs_K"], [physical_temperature])
    np.testing.assert_allclose(state["specific_energy_cgs_erg_g"], [specific_internal])
