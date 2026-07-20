"""Sphinx configuration for RadHydropy."""

from pathlib import Path
import os
import sys

try:
    from importlib.metadata import version as metadata_version
except ImportError:  # pragma: no cover
    from importlib_metadata import version as metadata_version


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "docs" / "_build" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "docs" / "_build" / "cache"))

project = "RadHydropy"
author = "Tsang Keung Chan"
copyright = "2026, Tsang Keung Chan"

try:
    release = metadata_version("radhydropy")
except Exception:
    release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
