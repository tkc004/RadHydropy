"""Initial-condition and snapshot helpers for multifrequency examples."""

from pathlib import Path

import numpy as np
import unyt

import radhydropy.io as rio
from radhydropy.arrays import as_named_array
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.thermo_networks.hydrogen import (
    collisional_equilibrium_neutral_fraction,
)
from radhydropy.units import quantity_to_value


def build_initial_condition(config):
    """Build a validated proper-coordinate multifrequency IC writer."""
    initial = config["initial_condition"]
    par_config = config["par"]
    chemistry = par_config.get("chemistry", {})
    thermochemistry = par_config.get("thermochemistry", {})
    units = config["_code_units"]
    grid_cells = int(par_config["mesh"]["grid_cells"])
    box_size_proper_unyt = initial["box_size_proper"]
    boundary_proper_unyt = (
        np.linspace(0.0, 1.0, grid_cells + 1) * box_size_proper_unyt
    )
    hydrogen_mass_fraction = float(
        chemistry.get("hydrogen_mass_fraction", 1.0)
    )
    rho_proper_unyt = (
        np.ones(grid_cells)
        * initial["hydrogen_number_density"]
        * unyt.mp
        / hydrogen_mass_fraction
    ).to(unyt.g / unyt.cm**3)
    temperature_proper_unyt = np.ones(grid_cells) * initial["temperature_proper"]

    writer = InitialConditionWriter(
        par_config=par_config, code_units=units, ic_config=initial,
    )
    writer.box_size = writer.radquantity(box_size_proper_unyt)
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(
        np.zeros(grid_cells) * units.velocity_unit
    )
    writer.fluid.temp_radarray = writer.radarray(temperature_proper_unyt)

    hydrogen_xHI_initial = float(chemistry.get("hydrogen_xHI_initial", 1.0))
    if thermochemistry.get("hydrogen_initial_collisional_equilibrium", False):
        hydrogen_xHI_initial = collisional_equilibrium_neutral_fraction(
            temperature_proper_unyt.to_value(unyt.K)
        )
    writer.simulation.fluid.xHI = as_named_array(
        np.full(grid_cells, hydrogen_xHI_initial, dtype=float)
    )

    network_name = thermochemistry.get("thermochemistry_network", "hydrogen")
    if network_name == "hydrogen_helium":
        helium_mass_fraction = float(
            chemistry.get("helium_mass_fraction", 0.0)
        )
        writer.simulation.fluid.mu = as_named_array(
            np.full(
                grid_cells,
                1.0 / (hydrogen_mass_fraction + helium_mass_fraction / 4.0),
            )
        )
        writer.simulation.fluid.xHeI = as_named_array(
            np.full(
                grid_cells,
                float(chemistry.get("hydrogen_helium_xHeI_initial", 1.0)),
            )
        )
        writer.simulation.fluid.xHeII = as_named_array(
            np.full(
                grid_cells,
                float(chemistry.get("hydrogen_helium_xHeII_initial", 0.0)),
            )
        )
        writer.simulation.fluid.xHeIII = as_named_array(
            np.full(
                grid_cells,
                float(chemistry.get("hydrogen_helium_xHeIII_initial", 0.0)),
            )
        )
    else:
        writer.simulation.fluid.mu = as_named_array(
            np.ones(grid_cells, dtype=float)
        )

    group_count_value = getattr(
        writer.simulation.par.radiation,
        "number_of_radiation_groups",
        None,
    )
    group_count = 1 if group_count_value is None else int(group_count_value)
    photon_number_density_cgs_cm3_unyt = (
        np.zeros((group_count, grid_cells)) / unyt.cm**3
    )
    writer.fluid.ngamma_radarray = writer.radarray(
        photon_number_density_cgs_cm3_unyt,
    )
    return writer


def load_snapshot(filename, config):
    """Load a multifrequency snapshot through the typed nested-config API."""
    return rio.loadhdf5(config, str(filename))


def active_radarray(values, active_cells, ghost_cells, *, boundary=False):
    """Return an active-cell RadArray from either saved shape."""
    expected_active = active_cells + 1 if boundary else active_cells
    values_shape = np.shape(values)
    values_size = values_shape[-1] if len(values_shape) > 1 else int(values_shape[0])
    if values_size == expected_active:
        return values
    expected_ghosted = expected_active + 2 * ghost_cells
    if values_size == expected_ghosted:
        if len(values_shape) > 1:
            return values[..., ghost_cells:ghost_cells + expected_active]
        return values[ghost_cells:ghost_cells + expected_active]
    raise ValueError(
        f"unexpected {'boundary' if boundary else 'fluid'} RadArray size "
        f"{values_size}; expected {expected_active} or {expected_ghosted}"
    )


def load_log_reference_profile(filename, radius_unit):
    """Load a log10 reference profile using explicit proper-radius naming."""
    if filename is None or not Path(filename).exists():
        return None
    data = np.loadtxt(filename, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        "radius_proper_kpc": data[:, 0] * radius_unit.to_value(unyt.kpc),
        "value": 10.0 ** data[:, 1],
    }
