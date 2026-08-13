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
