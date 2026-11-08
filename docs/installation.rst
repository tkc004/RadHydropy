Installation
============

RadHydropy supports Python 3.10 and newer. A virtual environment is
recommended for both the runtime and the optional development dependencies.

Install RadHydropy in editable mode while developing:

.. code-block:: bash

   python -m pip install -e .

Alternatively, use `uv <https://docs.astral.sh/uv/>`_ to create and manage the
project environment. Clone the repository, change into the project directory,
and install the optional test and documentation dependencies with:

.. code-block:: bash

   git clone https://github.com/tkc004/RadHydropy.git
   cd RadHydropy
   uv sync --group dev

Run commands in the uv-managed environment with ``uv run``:

.. code-block:: bash

   uv run pytest

Troubleshooting HPC Python installations
------------------------------------------

If ``uv run pytest`` fails during test collection with:

.. code-block:: text

   ModuleNotFoundError: No module named '_ctypes'

this is not a RadHydropy or uv dependency error. The Python installation
provided by the HPC system is missing Python's standard ``_ctypes`` extension.
SciPy requires ``ctypes``, so test collection fails before RadHydropy tests
run.

Use a complete Python installation, preferably Python 3.11--3.13:

.. code-block:: bash

   uv python install 3.12
   uv venv --python 3.12
   uv sync --group dev
   uv run pytest

You can verify the Python installation directly with:

.. code-block:: bash

   uv run python -c "import ctypes; print(ctypes.__file__)"

If this command produces the same ``_ctypes`` error, load an HPC-provided
Python module with a complete standard library or install Python with ``libffi``
support.

The package requires ``h5py``, ``numpy``, and ``unyt``. Tests additionally use
``pytest``:

.. code-block:: bash

   python -m pip install -e ".[test]"
   python -m pytest

Building The Documentation
--------------------------

Install the documentation extra and run Sphinx from the project root:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   sphinx-build -b html docs docs/_build/html

You can also build from inside the ``docs`` directory:

.. code-block:: bash

   cd docs
   make html

The generated HTML entry point is ``docs/_build/html/index.html``.
