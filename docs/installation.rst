Installation
============

RadHydropy supports Python 3.10 and newer. A virtual environment is
recommended for both the runtime and the optional development dependencies.

Install RadHydropy in editable mode while developing:

.. code-block:: bash

   python -m pip install -e .

Alternatively, use `uv <https://docs.astral.sh/uv/>`_ to create and manage the
project environment. From the project root, install the optional test and
documentation dependencies with:

.. code-block:: bash

   uv sync --group dev

Run commands in the uv-managed environment with ``uv run``:

.. code-block:: bash

   uv run pytest

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
