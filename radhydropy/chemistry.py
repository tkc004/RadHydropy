"""Composition-level chemistry module selection.

This module provides a clear separation between composition presets:

- ``H`` for hydrogen only
- ``HHe`` for hydrogen + helium
- ``HHeM`` for hydrogen + helium + metals
- ``HHeMol`` for hydrogen + helium + molecules
- ``HHeMMol`` for hydrogen + helium + metals + molecules

The actual species microphysics lives in :mod:`radhydropy.chemistry_species`.
"""

from dataclasses import dataclass
from typing import Tuple

from radhydropy.chemistry_species import helium, hydrogen, metal, molecule


@dataclass(frozen=True)
class ChemistryModule:
    """A named chemistry composition preset."""

    key: str
    species: Tuple[str, ...]


CHEMISTRY_MODULES = {
    "H": ChemistryModule("H", ("hydrogen",)),
    "HHe": ChemistryModule("HHe", ("hydrogen", "helium")),
    "HHeM": ChemistryModule("HHeM", ("hydrogen", "helium", "metal")),
    "HHeMol": ChemistryModule("HHeMol", ("hydrogen", "helium", "molecule")),
    "HHeMMol": ChemistryModule(
        "HHeMMol",
        ("hydrogen", "helium", "metal", "molecule"),
    ),
}


def available_chemistry_modules():
    """Return the supported chemistry composition keys."""
    return tuple(sorted(CHEMISTRY_MODULES))


def get_chemistry_module(par=None, key=None):
    """Return the selected chemistry composition preset."""
    chemistry_key = key if key is not None else getattr(par, "chemistry_key", "H")
    try:
        return CHEMISTRY_MODULES[chemistry_key]
    except KeyError as exc:
        available = ", ".join(available_chemistry_modules())
        raise ValueError(
            f"Unknown chemistry module {chemistry_key!r}; available modules: {available}"
        ) from exc


def get_species_modules(chemistry_key):
    """Return the underlying species helper modules for a composition key."""
    modules = {
        "hydrogen": hydrogen,
        "helium": helium,
        "metal": metal,
        "molecule": molecule,
    }
    chemistry_module = get_chemistry_module(key=chemistry_key)
    return tuple(modules[name] for name in chemistry_module.species)


__all__ = [
    "CHEMISTRY_MODULES",
    "ChemistryModule",
    "available_chemistry_modules",
    "get_chemistry_module",
    "get_species_modules",
]
