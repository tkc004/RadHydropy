Validation and Quality Checks
=============================

Use the checks below before submitting changes. Choose the smallest relevant
check while developing, then run the complete set before merging.

Documentation
-------------

Build the Sphinx documentation from the repository root:

.. code-block:: bash

   python -m sphinx -b html docs /tmp/radhydropy-docs

The build should complete without errors. Missing generated example figures or
NPZ files produce warnings; these should be investigated before enabling a
warnings-as-errors documentation job.

Python tests
------------

Run the full suite:

.. code-block:: bash

   python -m pytest

For configuration and example workflow changes, run the focused alignment
audit as well:

.. code-block:: bash

   python -m pytest -q tests/test_example_alignment.py

Code quality
------------

The repository quality configuration uses Ruff and mypy:

.. code-block:: bash

   ruff check .
   ruff format --check .
   mypy

The configured mypy file list is intentionally focused on the typed runtime,
IO, solver, diagnostics, and supporting tools. Do not treat an unconfigured
module as covered by the mypy command.

Example validation
------------------

Run examples from their own directories so local helper imports and relative
output paths resolve correctly. Confirm both the final time and expected
artifacts, not only process exit status. For a standard small smoke test:

.. code-block:: bash

   cd example/SodShock1D
   python sodshock1d.py

For cosmological or generated-input examples, record prerequisites and
non-fatal warnings separately. In particular, verify correlation tables exist
before running the cosmological virial-shock workflow and report C²-Ray
warnings when a configuration explicitly requests ``warn`` behavior.

Change hygiene
--------------

Before committing, check whitespace and review the complete diff:

.. code-block:: bash

   git diff --check
   git status --short
   git diff

Preserve validated output names, diagnostic formats, and representation-aware
field names when changing example workflows. New persistence behavior should
use the public ``radhydropy.io`` façade and existing HDF5 contracts.
