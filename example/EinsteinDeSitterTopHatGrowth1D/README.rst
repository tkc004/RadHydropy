Einstein--de Sitter Top-Hat Linear Growth
=========================================

This example evolves a cold, small-amplitude spherical top-hat perturbation
with supercomoving cosmological gravity.  The initial peculiar velocity is the
Einstein--de Sitter growing-mode velocity.  The mean overdensity is measured
inside the moving shell containing the initial top-hat mass and compared with

.. math::

   \delta(a) = \delta_i {a\over a_i}.

Run it with::

   python einstein_de_sitter_top_hat_growth1d.py

The example writes ``EinsteinDeSitterTopHatGrowth1D.jpg`` and checks the final
growth against linear theory.
