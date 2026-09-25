# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Export the stellar-wind Stromgren-sphere snapshots for a static web page."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import radhydropy.io as rio
from radhydropy.example_config import load_example_config

DEFAULT_CONFIG = Path(
    "example/DynamicStromgrenSpherePhotoheating20pcStellarWind1D/"
    "dynamic_stromgren_sphere_photoheating20pc_stellar_wind1d.yaml",
)
DEFAULT_OUTPUT = Path("docs/_static/interactive/stellar-wind-data.json")
PROTON_MASS_G = 1.67262192369e-24
NEUTRAL_FRONT_THRESHOLD = 0.5
WIND_EXCLUSION_CELLS = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],  # noqa: S607
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip(),
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _snapshot_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def _snapshots(config: dict[str, Any], config_path: Path) -> list[Path]:
    output = config["par"]["output"]
    directory = Path(output.get("directory", "."))
    if not directory.is_absolute():
        directory = config_path.parent / directory
    prefix = str(output.get("filename_prefix", "Output"))
    result = sorted(directory.glob(f"{prefix}_*.hdf5"), key=_snapshot_index)
    if not result:
        raise FileNotFoundError(f"no {prefix}_*.hdf5 snapshots found in {directory}")
    return result


def _pressure_rows(snapshot_directory: Path) -> np.ndarray:
    filename = snapshot_directory / (
        "DynamicStromgrenSpherePhotoheating20pcStellarWind1D_PressureRatio.csv"
    )
    if not filename.exists():
        return np.empty((0, 5), dtype=float)
    return np.genfromtxt(filename, delimiter=",", names=True).view(float).reshape(-1, 5)


def _interpolate(rows: np.ndarray, time_myr: float, column: int) -> float | None:
    if rows.size == 0:
        return None
    values = np.interp(time_myr, rows[:, 0], rows[:, column])
    return float(values) if np.isfinite(values) else None


def _export(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_example_config(config_path)
    snapshots = _snapshots(config, config_path)
    pressure = _pressure_rows(snapshots[0].parent)
    frames: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    for filename in snapshots:
        snapshot = rio.loadhdf5(config, str(filename))
        first = int(snapshot.par.mesh.ghost_cells)
        count = int(snapshot.par.mesh.grid_cells)
        last = first + count
        boundary = np.asarray(snapshot.mesh.boundary_radarray.to_value("pc"), dtype=float)[
            first : last + 1
        ]
        radius = 0.5 * (boundary[:-1] + boundary[1:])
        density = (
            np.asarray(snapshot.fluid.rho_radarray.to_value("g/cm**3"), dtype=float)[first:last]
            / PROTON_MASS_G
        )
        temperature = np.asarray(snapshot.fluid.temp_radarray.to_value("K"), dtype=float)[
            first:last
        ]
        velocity = np.asarray(snapshot.fluid.vel_radarray.to_value("km/s"), dtype=float)[first:last]
        neutral_fraction = np.asarray(snapshot.fluid.xHI, dtype=float)[first:last]
        time_myr = float(
            np.asarray(snapshot.fluid.time_proper_code, dtype=float)
            * snapshot.par.units.CodeUnits.time_unit.to_value("Myr"),
        )
        neutral = np.flatnonzero(neutral_fraction >= NEUTRAL_FRONT_THRESHOLD)
        front_radius = float(radius[neutral[0]]) if neutral.size else None
        shell_index = (
            int(np.argmax(density[WIND_EXCLUSION_CELLS:]) + WIND_EXCLUSION_CELLS)
            if density.size > WIND_EXCLUSION_CELLS
            else int(np.argmax(density))
        )
        frames.append(
            _json_value(
                {
                    "snapshot": filename.name,
                    "time_myr": time_myr,
                    "radius_pc": radius,
                    "density_cm3": density,
                    "temperature_k": temperature,
                    "velocity_km_s": velocity,
                    "neutral_fraction": neutral_fraction,
                    "ionization_front_pc": front_radius,
                    "wind_shell_pc": radius[shell_index],
                    "wind_pressure_dyn_cm2": _interpolate(pressure, time_myr, 2),
                    "gas_pressure_dyn_cm2": _interpolate(pressure, time_myr, 3),
                    "pressure_ratio": _interpolate(pressure, time_myr, 4),
                },
            ),
        )
        source_files.append({"name": filename.name, "sha256": _sha256(filename)})
    return {
        "name": "stellar_wind_stromgren_sphere",
        "config_filename": str(config_path),
        "config_sha256": _sha256(config_path),
        "snapshot_count": len(snapshots),
        "grid_cells": int(config["par"]["mesh"]["grid_cells"]),
        "ghost_cells": int(config["par"]["mesh"]["ghost_cells"]),
        "source_files": source_files,
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "title": "Photoheated Stromgren sphere with a stellar wind",
        "generated_by": "tools/export_stellar_wind_interactive.py",
        **_git_metadata(),
        "run": _export(args.config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8",
    )
    print(f"wrote {args.output} ({len(payload['run']['frames'])} frames)")  # noqa: T201


if __name__ == "__main__":
    main()
