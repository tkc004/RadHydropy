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
