"""Repository-wide contract checks for the maintained example workflows.

These checks are intentionally source-oriented.  The ordinary example tests
exercise selected workflows; this module makes the conventions from the
example-maintenance skill fail fast for every example and every YAML file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from example.example_utils import load_nested_example_config


REPO_ROOT = Path(__file__).parents[1]
EXAMPLE_ROOT = REPO_ROOT / "example"

# These names are physical quantities in the example configuration contract.
# Dimensionless thresholds and ratios deliberately do not appear here.
UNIT_REQUIRED_KEYS = {
    "box_size",
    "cmb_temperature_0",
    "cie_temperature_proper",
    "hydrogen_number_density",
    "photon_flux",
    "radius_inner",
    "radius_outer",
    "source_photon_rate",
    "target_halo_mass",
}

FORBIDDEN_IDENTIFIERS = {
    "ICparams",
    "Runparams",
    "SimpleNamespace",
    "WindSph",
    "ic_for_sim",
    "legacy_example_parameters",
    "legacy_initial_condition_parameters",
    "load_nested_example_parameters",
    "outdir",
    "outfileprefix",
    "outputtimefilename",
    "parameter_namespace",
    "runparams",
}

GENERIC_PHYSICAL_NAMES = {
    "density",
    "mass",
    "pressure",
    "radius",
    "rho",
    "temperature",
    "temp",
    "time",
    "vel",
    "velocity",
}

PHYSICAL_SOURCE_MARKERS = (
    "_cgs",
    "_code",
    "_comoving",
    "_proper",
    "_supercomoving",
    "fluid.",
    "mesh.",
    "rho_",
    "temperature_",
    "velocity_",
)

# These are physical output/argument names which look unit-qualified but do
# not identify the coordinate representation.  Configuration keys are kept
# separate: those are checked by ``test_physical_yaml_values_have_explicit_units``.
AMBIGUOUS_PHYSICAL_NAMES = {
    "time_s",
    "time_yr",
    "time_myr",
    "temperature_cgs_K",
    "density_cgs_g_cm3",
    "radius_pc",
    "radius_cgs_cm",
    "velocity_km_s",
    "cosmic_time",
    "time_myr",
}

PHYSICAL_FALLBACK_KEYS = {
    "box_size_comoving",
    "time_cosmic",
    "temperature_proper",
    "radius_inner_proper",
    "radius_outer_proper",
}

INTENTIONAL_KEYWORD_EXCEPTIONS = {
    # This is the established core constructor API; migrating it belongs to
    # the core-source workflow, not to an example-only compatibility layer.
    "DarkMatterShells": {"radius", "velocity", "mass"},
    # Current core Gravity API boundary; changing this requires coordinated
    # core-source and regression-test migration.
    "acceleration_on_mesh": {"rho"},
}


def _yaml_files() -> list[Path]:
    return sorted(EXAMPLE_ROOT.rglob("*.yaml"))


def _python_files() -> list[Path]:
    return sorted(EXAMPLE_ROOT.rglob("*.py"))


def _walk_mapping(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_mapping(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mapping(child)


def _is_unit_mapping(value) -> bool:
    return isinstance(value, dict) and {"value", "unit"}.issubset(value)


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader variant which rejects duplicate YAML mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                f"found duplicate key {key!r}", key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _literal_string(node: ast.AST):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _dict_key_strings(node: ast.Dict):
    return [key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]


def test_every_example_yaml_is_a_complete_loadable_config():
    failures = []
    for filename in _yaml_files():
        try:
            config = load_nested_example_config(filename)
        except Exception as exc:  # noqa: BLE001 - report every bad config at once
            failures.append(f"{filename.relative_to(REPO_ROOT)}: {exc}")
            continue
        if set(config) != {"par", "initial_condition", "example"}:
            failures.append(
                f"{filename.relative_to(REPO_ROOT)}: incomplete top-level config"
            )
    assert not failures, "\n".join(failures)


def test_physical_yaml_values_have_explicit_units():
    failures = []
    for filename in _yaml_files():
        raw = yaml.safe_load(filename.read_text(encoding="utf-8"))
        for key, value in _walk_mapping(raw):
            if key not in UNIT_REQUIRED_KEYS or value is None:
                continue
            if not _is_unit_mapping(value):
                failures.append(
                    f"{filename.relative_to(REPO_ROOT)}: {key!r} must be "
                    "a {value, unit} mapping"
                )
    assert not failures, "\n".join(failures)


def test_example_yaml_has_no_duplicate_mapping_keys():
    failures = []
    for filename in _yaml_files():
        try:
            yaml.load(filename.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        except yaml.YAMLError as exc:
            failures.append(f"{filename.relative_to(REPO_ROOT)}: {exc}")
    assert not failures, "\n".join(failures)


def test_example_python_has_no_duplicate_literal_mapping_keys():
    failures = []
    for filename in _python_files():
        source = filename.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filename))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = _dict_key_strings(node)
            duplicates = sorted({key for key in keys if keys.count(key) > 1})
            if duplicates:
                failures.append(
                    f"{filename.relative_to(REPO_ROOT)}:{node.lineno}: "
                    f"duplicate mapping key(s): {', '.join(duplicates)}"
                )
    assert not failures, "\n".join(failures)


def test_diagnostic_keys_and_physical_parameters_use_representation_names():
    failures = []
    for filename in _python_files():
        source = filename.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filename))

        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in _dict_key_strings(node):
                    if key in AMBIGUOUS_PHYSICAL_NAMES:
                        failures.append(
                            f"{filename.relative_to(REPO_ROOT)}:{node.lineno}: "
                            f"ambiguous diagnostic/mapping key: {key}"
                        )

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = [*node.args.args, *node.args.kwonlyargs]
                for argument in arguments:
                    if argument.arg in GENERIC_PHYSICAL_NAMES | AMBIGUOUS_PHYSICAL_NAMES:
                        failures.append(
                            f"{filename.relative_to(REPO_ROOT)}:{argument.lineno}: "
                            f"ambiguous physical parameter: {argument.arg}"
                        )

            if isinstance(node, ast.Call):
                callee_name = (
                    node.func.id if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute)
                    else ""
                )
                allowed = INTENTIONAL_KEYWORD_EXCEPTIONS.get(callee_name, set())
                for keyword in node.keywords:
                    if keyword.arg in GENERIC_PHYSICAL_NAMES and keyword.arg not in allowed:
                        failures.append(
                            f"{filename.relative_to(REPO_ROOT)}:{keyword.lineno}: "
                            f"ambiguous physical keyword: {keyword.arg}"
                        )

    assert not failures, "\n".join(failures)


def test_physical_get_fallbacks_are_not_unitless():
    failures = []
    for filename in _python_files():
        source = filename.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filename))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "get" or len(node.args) < 2:
                continue
            key = _literal_string(node.args[0])
            fallback = node.args[1]
            if key in PHYSICAL_FALLBACK_KEYS and isinstance(fallback, (ast.Constant, ast.Num)):
                failures.append(
                    f"{filename.relative_to(REPO_ROOT)}:{node.lineno}: "
                    f"unitless fallback for physical key {key!r}"
                )
    assert not failures, "\n".join(failures)


def test_example_python_has_no_forbidden_compatibility_apis():
    failures = []
    for filename in _python_files():
        source = filename.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filename))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_IDENTIFIERS:
                failures.append(f"{filename.relative_to(REPO_ROOT)}:{node.lineno}: {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_IDENTIFIERS:
                failures.append(f"{filename.relative_to(REPO_ROOT)}:{node.lineno}: {node.attr}")

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "safe_load" and filename.name != "example_utils.py":
                failures.append(
                    f"{filename.relative_to(REPO_ROOT)}:{node.lineno}: direct yaml.safe_load"
                )
    assert not failures, "\n".join(failures)


def test_cosmological_startup_restores_all_three_clocks():
    failures = []
    for filename in _python_files():
        source = filename.read_text(encoding="utf-8")
        if "SetInitFluid" not in source:
            continue
        if not any(
            marker in filename.parts
            for marker in ("Cosmological", "BertschingerGasReference", "GasCentrifugalCosmologicalOrbit1D")
        ):
            continue
        required = (
            "initial_tau.copy()",
            "sim.par.tau_supercomoving_code = initial_tau.copy()",
            "sim.par.simulation.tau_supercomoving_code = initial_tau.copy()",
            "sim.fluid.SetFluidTime(initial_tau)",
            "np.allclose",
        )
        missing = [item for item in required if item not in source]
        if missing:
            failures.append(f"{filename.relative_to(REPO_ROOT)}: missing {', '.join(missing)}")
    assert not failures, "\n".join(failures)


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


def test_physical_locals_use_explicit_names():
    failures = []
    for filename in _python_files():
        source = filename.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filename))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            generic_names = _assigned_names(node) & GENERIC_PHYSICAL_NAMES
            if not generic_names:
                continue
            rhs = ast.get_source_segment(source, node.value) or ""
            if any(marker in rhs for marker in PHYSICAL_SOURCE_MARKERS):
                failures.append(
                    f"{filename.relative_to(REPO_ROOT)}:{node.lineno}: "
                    f"generic physical local(s): {', '.join(sorted(generic_names))}"
                )
    assert not failures, "\n".join(failures)


def _is_par_projection(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    index = node.slice
    if isinstance(node.value, ast.Name):
        return node.value.id == "config" and isinstance(index, ast.Constant) and index.value == "par"
    return _is_par_projection(node.value)


def test_complete_config_is_preserved_at_helper_boundaries():
    failures = []
    boundary_names = {
        "build_initial_condition",
        "interior_slice",
        "load_output_state",
        "snapshot_physical_fields",
        "write_initial_condition",
    }
    for filename in _python_files():
        source = filename.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filename))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else ""
            )
            if callee not in boundary_names and not callee.startswith("_to_"):
                continue
            projected = []
            if node.args and _is_par_projection(node.args[0]):
                projected.append(node.args[0])
            projected.extend(
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "config" and _is_par_projection(keyword.value)
            )
            if projected:
                failures.append(
                    f"{filename.relative_to(REPO_ROOT)}:{node.lineno}: "
                    f"{callee} received config['par'] instead of complete config"
                )
    assert not failures, "\n".join(failures)
