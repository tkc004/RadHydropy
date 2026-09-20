# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Validation of persisted HDF5 state against a runtime configuration."""

import numpy as np

from radhydropy.io.metadata import _restore_header_attr_value
from radhydropy.units import CodeUnits


class SnapshotConfigurationError(ValueError):
    """Raised when a snapshot is incompatible with the supplied runtime."""


def _code_units_from_parameter(par):
    """Return pre-existing runtime code units, if the runtime declares them."""
    code_units = getattr(par, "CodeUnits", None)
    if code_units is not None:
        return code_units
    return getattr(getattr(par, "units", None), "CodeUnits", None)


def validate_snapshot_configuration(par, header, header_code_units):
    """Validate header compatibility before mutating runtime objects."""
    expected_units = _code_units_from_parameter(par)
    if expected_units is not None:
        if not isinstance(expected_units, CodeUnits):
            expected_units = CodeUnits.from_mapping(expected_units)
        scales = (
            ("mass", expected_units.mass_in_cgs, header_code_units.mass_in_cgs),
            ("length", expected_units.length_in_cgs, header_code_units.length_in_cgs),
            ("velocity", expected_units.velocity_in_cgs, header_code_units.velocity_in_cgs),
            ("current", expected_units.current_in_cgs, header_code_units.current_in_cgs),
            (
                "temperature",
                expected_units.temperature_in_cgs,
                header_code_units.temperature_in_cgs,
            ),
        )
        for name, expected, actual in scales:
            if not np.isclose(expected, actual, rtol=1e-12, atol=0.0):
                raise SnapshotConfigurationError(
                    f"snapshot CodeUnits {name} scale ({actual}) does not "
                    f"match runtime scale ({expected})",
                )

    header_coordsys = _restore_header_attr_value(
        header.attrs.get("CoordinateSystem", None),
    )
    expected_coordsys = getattr(
        getattr(par, "simulation", None),
        "coordinate_system",
        None,
    )
    if (
        header_coordsys is not None
        and expected_coordsys is not None
        and str(header_coordsys) != str(expected_coordsys)
    ):
        raise SnapshotConfigurationError(
            f"snapshot coordinate system {header_coordsys!r} does not match "
            f"runtime coordinate system {expected_coordsys!r}",
        )

    header_grid = header.attrs.get("GridCells", None)
    expected_grid = getattr(getattr(par, "mesh", None), "grid_cells", None)
    if (
        header_grid is not None
        and expected_grid is not None
        and int(_restore_header_attr_value(header_grid)) != int(expected_grid)
    ):
        raise SnapshotConfigurationError(
            f"snapshot grid size {int(_restore_header_attr_value(header_grid))} "
            f"does not match runtime grid size {int(expected_grid)}",
        )

    header_cosmology = _restore_header_attr_value(
        header.attrs.get("CosmologyType", None),
    )
    expected_expansion = getattr(par, "cosmological_expansion", None)
    if header_cosmology is not None and expected_expansion is False:
        raise SnapshotConfigurationError(
            "snapshot is cosmological but runtime has cosmological_expansion=False",
        )
    if header_cosmology is None and expected_expansion is True:
        raise SnapshotConfigurationError(
            "snapshot is non-cosmological but runtime has cosmological_expansion=True",
        )
    expected_cosmology = getattr(par, "cosmology_type", None)
    if expected_cosmology is None:
        expected_model = getattr(par, "cosmology", None)
        expected_cosmology = getattr(expected_model, "type_name", None)
    if header_cosmology is not None and expected_cosmology is not None:
        header_cosmology = str(header_cosmology)
        expected_cosmology = str(expected_cosmology)
        if header_cosmology != expected_cosmology:
            raise SnapshotConfigurationError(
                f"snapshot cosmology {header_cosmology!r} does not match "
                f"runtime cosmology {expected_cosmology!r}",
            )

    for header_key, parameter_key in (
        ("CoordinateFrame", "coordinate_frame"),
        ("TimeCoordinate", "time_coordinate"),
        ("VelocityRepresentation", "velocity_representation"),
        ("DensityRepresentation", "density_representation"),
        ("PressureRepresentation", "pressure_representation"),
        ("TemperatureRepresentation", "temperature_representation"),
    ):
        header_value = _restore_header_attr_value(
            header.attrs.get(header_key, None),
        )
        expected_value = getattr(par, parameter_key, None)
        if (
            header_value is not None
            and expected_value is not None
            and str(header_value) != str(expected_value)
        ):
            raise SnapshotConfigurationError(
                f"snapshot {parameter_key} {header_value!r} does not match "
                f"runtime {parameter_key} {expected_value!r}",
            )
