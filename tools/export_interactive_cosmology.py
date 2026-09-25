# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Export RadHydropy cosmological snapshots for the static web explorer."""

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
    "example/CosmologicalVirialShock1D/cosmological_gas_correlation_tvir1e3_z15.yaml",
)
DEFAULT_OUTPUT = Path("docs/_static/interactive/cosmological-collapse-data.json")


def _json_value(value: Any) -> Any:
    """Convert NumPy values to JSON-safe values, preserving missing data."""
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    """Return best-effort source identity without requiring a Git checkout."""
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


def _snapshot_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"snapshot filename is not numbered: {path}") from exc


def _find_snapshots(config: dict[str, Any], config_path: Path) -> list[Path]:
    output = config["par"]["output"]
    directory = Path(output["directory"])
    if not directory.is_absolute():
        directory = config_path.parent / directory
    prefix = str(output.get("filename_prefix", "Output"))
    snapshots = sorted(
        directory.glob(f"{prefix}_*.hdf5"),
        key=_snapshot_index,
    )
    if not snapshots:
        raise FileNotFoundError(f"no {prefix}_*.hdf5 snapshots found in {directory}")
    return snapshots


def _profile_npz(config: dict[str, Any], snapshots: list[Path]) -> dict[str, np.ndarray]:
    prefix = str(config["example"].get("figure_prefix", "CosmologicalGasCorrelationZ100"))
    profile = snapshots[0].parent / f"{prefix}.npz"
    if not profile.exists():
        return {}
    with np.load(profile, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _dark_matter_profile_npz(
    config: dict[str, Any],
    snapshots: list[Path],
) -> dict[str, np.ndarray]:
    prefix = str(config["example"].get("figure_prefix", "CosmologicalGasCorrelationZ100"))
    profile = snapshots[0].parent / f"{prefix}_DarkMatterDensities.npz"
    if not profile.exists():
        return {}
    with np.load(profile, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _active_fields(
    snapshot: Any,
    temperature_proper_cgs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first = int(snapshot.par.mesh.ghost_cells)
    count = int(snapshot.par.mesh.grid_cells)
    last = first + count
    boundary = np.asarray(
        snapshot.mesh.boundary_radarray.to_value("kpc"),
        dtype=float,
    )[first : last + 1]
    radius = 0.5 * (boundary[:-1] + boundary[1:])
    density = np.asarray(snapshot.fluid.rho_radarray.to_value("g/cm**3"), dtype=float)[first:last]
    if temperature_proper_cgs is not None:
        temperature_array = np.asarray(temperature_proper_cgs, dtype=float)
        temperature = (
            temperature_array if temperature_array.size == count else temperature_array[first:last]
        )
    else:
        temperature = np.asarray(snapshot.fluid.temp_radarray.to_value("K"), dtype=float)[
            first:last
        ]
    velocity = np.asarray(snapshot.fluid.vel_radarray.to_value("km/s"), dtype=float)[first:last]
    return radius, density, temperature, velocity


def _marker(profile: dict[str, np.ndarray], key: str, index: int) -> float | None:
    values = profile.get(key)
    if values is None or index >= values.shape[0]:
        return None
    value = float(values[index])
    return value if np.isfinite(value) else None


def _export_run(name: str, config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_example_config(config_path)
    snapshots = _find_snapshots(config, config_path)
    profile = _profile_npz(config, snapshots)
    dark_matter_profile = _dark_matter_profile_npz(config, snapshots)
    frames: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    for index, filename in enumerate(snapshots):
        snapshot = rio.loadhdf5(config, str(filename))
        temperature_profile = profile.get("temperature_proper_cgs_K")
        temperature_values = (
            temperature_profile[index]
            if temperature_profile is not None and index < temperature_profile.shape[0]
            else None
        )
        radius, density, temperature, velocity = _active_fields(snapshot, temperature_values)
        frame: dict[str, Any] = {
            "snapshot": filename.name,
            "radius_comoving_kpc": radius,
            "density_g_cm3": density,
            "temperature_k": temperature,
            "velocity_km_s": velocity,
            "time_cosmic_gyr": _marker(profile, "time_cosmic_Gyr", index),
            "scale_factor": _marker(profile, "scale_factor", index),
            "redshift": None,
            "rvir_comoving_kpc": _marker(profile, "rvir_kpc", index),
            "rvir_proper_kpc": _marker(profile, "rvir_proper_kpc", index),
            "shock_comoving_kpc": _marker(profile, "rshock_kpc", index),
            "splashback_comoving_kpc": _marker(profile, "rsplashback_kpc", index),
            "shock_mach": _marker(profile, "shock_local_mach", index),
        }
        if frame["scale_factor"] is not None and frame["scale_factor"] > 0.0:
            frame["redshift"] = 1.0 / frame["scale_factor"] - 1.0
        dark_matter = getattr(snapshot, "dark_matter", None)
        if dark_matter is not None:
            dark_matter_radius = dark_matter_profile.get("radius_proper_kpc")
            if dark_matter_radius is not None and index < dark_matter_radius.shape[0]:
                frame["dark_matter_radius_proper_kpc"] = np.asarray(
                    dark_matter_radius[index],
                    dtype=float,
                )
            else:
                frame["dark_matter_radius_proper_kpc"] = np.asarray(
                    dark_matter.radius_radarray.to_value("kpc"),
                    dtype=float,
                )
            frame["dark_matter_mass_msun"] = np.asarray(
                dark_matter.dark_matter_mass_radarray.to_value("Msun"),
                dtype=float,
            )
        frames.append(_json_value(frame))
        source_files.append(
            {
                "name": filename.name,
                "sha256": _sha256(filename),
                "bytes": filename.stat().st_size,
            },
        )
    return {
        "name": name,
        "config_filename": str(config_path),
        "config_sha256": _sha256(config_path),
        "snapshot_count": len(snapshots),
        "grid_cells": int(config["par"]["mesh"]["grid_cells"]),
        "ghost_cells": int(config["par"]["mesh"]["ghost_cells"]),
        "coordinate_system": str(config["par"]["simulation"]["coordinate_system"]),
        "cosmology_type": str(config["par"]["cosmology"]["cosmology_type"]),
        "source_files": source_files,
        "frames": frames,
    }


def _run_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("runs must use NAME=CONFIG.yaml")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", action="append", type=_run_argument, metavar="NAME=CONFIG")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    requested = args.run or [("tvir_1000_z15", args.config)]
    runs = {name: _export_run(name, path) for name, path in requested}
    payload = {
        "schema_version": 1,
        "title": "Cosmological collapse and virial shock",
        "default_run": requested[0][0],
        "generated_by": "tools/export_interactive_cosmology.py",
        **_git_metadata(),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    print(  # noqa: T201
        f"wrote {args.output} "
        f"({len(runs)} runs, {sum(len(run['frames']) for run in runs.values())} frames)",
    )


if __name__ == "__main__":
    main()
