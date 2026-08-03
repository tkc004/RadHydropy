"""Reuse the maintained dynamic Stromgren-sphere helper implementation."""

import importlib.util
import sys
from pathlib import Path


TEMPLATE_TOOLS = (
    Path(__file__).resolve().parents[1]
    / 'DynamicStromgrenSpherePhotoheating1D'
    / 'tools.py'
)
spec = importlib.util.spec_from_file_location(
    '_radhydropy_dynamic_stromgren_template_tools',
    TEMPLATE_TOOLS,
)
_template = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = _template
spec.loader.exec_module(_template)

for _name in dir(_template):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_template, _name)
