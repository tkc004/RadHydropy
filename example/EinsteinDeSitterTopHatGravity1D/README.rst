Einstein--de Sitter Top-Hat Gravity
===================================

This fixed-time diagnostic initializes a spherical gas top-hat in an
Einstein--de Sitter background using supercomoving variables.  The homogeneous
background is subtracted by cosmological gravity, leaving the exact excess
field

.. math::

   g(x) = -{4\pi\over3} G a \bar\rho_{\rm com}\delta
          \begin{cases}x, & x < R,\\ R^3/x^2, & x \ge R.\end{cases}

Run it with::

   python einstein_de_sitter_top_hat_gravity1d.py

The script writes ``EinsteinDeSitterTopHatGravity1D.jpg`` and fails if the
numerical field differs from the analytic result by more than 0.5 percent
(the origin cell is excluded by the spherical symmetry convention).
