# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Source-level contracts for production runtime representation boundaries."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = REPO_ROOT / "radhydropy"

# These names are allowed only where the code deliberately detects and rejects
# legacy input.  They must not become runtime field aliases again.
LEGACY_REJECTION_FILES = {
    Path("analysis.py"),
    Path("cosmology/state.py"),
    Path("io/cosmology_state.py"),
    Path("io/hdf5.py"),
}
LEGACY_RUNTIME_NAMES = (
    "time_code",
    "box_size_code",
    "rho_code",
    "vel_code",
    "temp_code",
    "pre_code",
)


def _production_sources():
    return sorted(
        path
        for path in SOURCE_ROOT.rglob("*.py")
        if "tests" not in path.parts and "example" not in path.parts
    )


def test_production_source_does_not_reintroduce_legacy_runtime_names():
    failures = []
    for filename in _production_sources():
        relative = filename.relative_to(SOURCE_ROOT)
        if relative in LEGACY_REJECTION_FILES:
            continue
        source = filename.read_text(encoding="utf-8")
        for name in LEGACY_RUNTIME_NAMES:
            if re.search(rf"\b{re.escape(name)}\b", source):
                failures.extend([f"{relative}: contains forbidden runtime name {name!r}"])
    assert not failures, "\n".join(failures)


def test_representation_selection_is_centralized_in_runtime_consumers():
    required_tokens = {
        Path("diagnostics.py"): (
            "select_fluid_primitive_arrays",
            "select_mesh_geometry_arrays",
        ),
        Path("solver/core.py"): (
            "select_fluid_primitive_arrays",
            "select_mesh_geometry_arrays",
        ),
        Path("rsim/stepping.py"): (
            "select_fluid_primitive_arrays",
            "select_mesh_geometry_arrays",
        ),
        Path("output.py"): ("runtime_fields",),
    }
    failures = []
    for relative, tokens in required_tokens.items():
        source = (SOURCE_ROOT / relative).read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in source]
        if missing:
            failures.append(f"{relative}: missing canonical selector(s) {missing}")
    assert not failures, "\n".join(failures)


def test_hdf5_legacy_names_are_confined_to_rejection_guards():
    source = (SOURCE_ROOT / "io/hdf5.py").read_text(encoding="utf-8")
    assert "_GENERIC_HEADER_DATASETS" in source
    assert "_GENERIC_PRIMITIVE_DATASETS" in source
    assert "unsupported generic field name" in source
    assert "canonical representation-specific" in source


def test_runtime_io_diagnostics_use_structured_logging():
    for relative in (Path("output.py"), Path("io/hdf5.py")):
        source = (SOURCE_ROOT / relative).read_text(encoding="utf-8")
        assert "log_diagnostic" in source
        assert "print(" not in source
