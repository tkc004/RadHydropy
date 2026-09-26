Pure Dark-Matter Shell Crossing
===============================

This example evolves self-gravitating, infinitely thin spherical dark-matter
shells. Shell masses and specific angular momenta are fixed, shells are allowed
to cross, and the arrays are resorted by radius after every drift. The timestep
is reduced before a predicted neighboring-shell crossing.

This first example is intentionally pure dark matter: there is no gas coupling
or HDF5 restart path yet. It validates the shell dynamics and sorting invariant.

Run with::

   python dark_matter_shell_crossing1d.py \
       --config dark_matter_shell_crossing1d.yaml
Running from a clean checkout
----------------------------

From the repository root, install RadHydropy and the example dependencies::

   cd RadHydropy
   python -m pip install -e ".[test,docs]"

Then change into this example directory before running the command shown above::

   cd example/DarkMatterShellCrossing1D

If a command above begins with ``python example/``, run that command from
the repository root instead of changing into this directory.
