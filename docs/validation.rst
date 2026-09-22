Validation and Quality Checks
=============================

Use the checks below before submitting changes. Choose the smallest relevant
check while developing, then run the complete set before merging.

Authoritative baseline
----------------------

The current validation baseline was recorded on **2026-09-22** at commit
``76c5754``. Update this section whenever a code or test change alters the
expected results; the commit identifies the source tree for the numbers below.

The expected baseline is:

* the full test suite reports ``438 passed, 1 warning``;
* ``ruff check .`` passes;
* ``ruff format --check .`` passes; and
* strict mypy passes for the configured subset of 93 source, tool, and
  example-utility files.

The mypy result is intentionally not a claim that every repository module is
typed. The configured file list is the scope of that check.

Documentation
-------------

Build the Sphinx documentation from the repository root:

.. code-block:: bash

   python -m sphinx -b html -W docs /tmp/radhydropy-docs

The build should complete without errors or warnings. Missing generated example
figures or NPZ files indicate incomplete example artifacts and should be
investigated rather than hidden from the documentation build.

Python tests
------------

Run the full suite:

.. code-block:: bash

   python -m pytest

Expected result at the baseline commit: ``438 passed, 1 warning``. The warning
is currently expected; investigate any additional failures or warnings before
merging.

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

At the baseline commit, both Ruff commands pass. Strict mypy covers only its
configured file list, currently 93 source, tool, and example-utility files.

The configured mypy file list is intentionally focused on the typed runtime,
IO, solver, diagnostics, and supporting tools. Do not treat an unconfigured
module as covered by the mypy command.

Example validation
------------------

Run examples from their own directories so local helper imports and relative
output paths resolve correctly. Confirm both the final time and expected
artifacts, not only process exit status. Record the following for each
validated run:

* the example directory, YAML configuration, command, and commit;
* the reached final time and any non-fatal warnings;
* the generated ``InitialCondition.hdf5`` and ``Output_*.hdf5`` files, including
  the snapshot count or filename pattern; and
* generated figures and other diagnostics such as CSV or NPZ files.

For a standard small smoke test:

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
