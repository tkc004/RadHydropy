RadArray Conversion 1D
======================

This standalone example is an executable specification for ``RadArray``.
It checks proper, comoving, and supercomoving conversions using the complete
nested configuration, including the position-dependent supercomoving velocity
conversion.  It also verifies that proper and comoving density arrays cannot
be added without an explicit conversion.

Run it with::

    python example/RadArrayConversion1D/radarray_conversion1d.py
Running from a clean checkout
----------------------------

From the repository root, install RadHydropy and the example dependencies::

   cd RadHydropy
   python -m pip install -e ".[test,docs]"

Then change into this example directory before running the command shown above::

   cd example/RadArrayConversion1D

If a command above begins with ``python example/``, run that command from
the repository root instead of changing into this directory.
