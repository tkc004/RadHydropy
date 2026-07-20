Installation
============

Install RadHydropy in editable mode while developing:

.. code-block:: bash

   python -m pip install -e .

The package requires ``h5py``, ``numpy``, and ``unyt``. Tests additionally use
``pytest``:

.. code-block:: bash

   python -m pip install -e ".[test]"
   pytest

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
