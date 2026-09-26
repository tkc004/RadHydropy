Einstein--de Sitter Dark-Matter Shell Growth
============================================

This collisionless benchmark discretizes a homogeneous Einstein--de Sitter
dark-matter background with spherical shells and adds a small top-hat mass
overdensity. Equal-volume shell boundaries place the top-hat radius exactly at
a Lagrangian shell interface, with 1024 shells on each side. The shells use
supercomoving gravity,

.. math::

   x''(\tau)=-{G a\,[M_{DM}(<x)-M_{bg}(<x)]\over x^2},

with the growing-mode initial velocity. The measured overdensity is compared
with the linear prediction

.. math::

   \delta_{DM}(a)=\delta_i a/a_i.

The example also verifies that the unperturbed shell background has zero
peculiar acceleration. Run it with::

   python einstein_de_sitter_dark_matter_shell_growth1d.py
Running from a clean checkout
----------------------------

From the repository root, install RadHydropy and the example dependencies::

   cd RadHydropy
   python -m pip install -e ".[test,docs]"

Then change into this example directory before running the command shown above::

   cd example/EinsteinDeSitterDarkMatterShellGrowth1D

If a command above begins with ``python example/``, run that command from
the repository root instead of changing into this directory.
