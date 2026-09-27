Interactive Cosmological Collapse
=================================

This page is a static replay of RadHydropy's validated :math:`10^{13}\,M_\odot`,
z = 100 to z = 0 cosmological gas--dark-matter calculation. The browser does not
run a simulation: the page reads a compact data export generated from the
authoritative HDF5 snapshots.

The default replay uses
``cosmological_gas_correlation_z100_adiabatic_256_m1e13_z0_inner06.yaml``
and the scheduled snapshots in
``outputs_correlation_gas_adiabatic_256_m1e13_z0_inner06_schedule40``.
It contains 41 frames from z = 100 through z = 0.

The interaction plots use logarithmic radius. Gas density and gas temperature
also use logarithmic vertical scales so the wide dynamic range remains visible
throughout the collapse.

The 3D slice height is
:math:`\log_{10}|\Phi_{\rm corner}/\Phi(r)|`, where both potentials are from
the same redshift frame. The corner is :math:`(x,y)=(150,150)` kpc and is
fixed at :math:`z=0`; the central value becomes more negative as the halo
deepens. The white corner marker identifies this normalization point. Velocity
arrows are drawn on that curved surface.
The evolving virial radius, :math:`r_{200}`, is shown
in every profile.
The 3D surfaces use a cutaway in the :math:`x>0, y<0` quadrant to expose the
interior of the density and temperature wells. Profile radii are comoving;
the saved proper-radius virial and shock diagnostics are converted to
comoving radii for the markers. The profiles expand with the current
:math:`r_{200}`, while the 3D potential surface uses the fixed comoving range
:math:`[-150,150]` kpc for a consistent evolution scale.

.. raw:: html

   <link rel="stylesheet" href="_static/interactive/cosmological-collapse.css">
   <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
   <script src="_static/interactive/cosmological-collapse.js" defer></script>
   <div class="radhydropy-explorer" data-radhydropy-explorer data-data-url="_static/interactive/cosmological-collapse-data.json">
     <section class="hero">
       <p>RADHYDROpy · STATIC SIMULATION REPLAY</p>
       <h2>From a small overdensity to a virial shock</h2>
       <p>Move through the saved snapshots and follow gas density, temperature, velocity, and live dark-matter shells as the perturbation collapses.</p>
     </section>
     <section class="controls" aria-label="Interactive controls">
       <div class="control-row">
         <label>Run <select data-run></select></label>
         <label>Compare <select data-compare></select></label>
         <label><input type="checkbox" data-show-compare> show comparison</label>
         <button type="button" data-play>Play snapshots</button>
       </div>
       <div class="control-row">
         <label for="frame">Time</label>
         <input id="frame" type="range" min="0" value="0" data-frame>
         <output data-frame-label>loading...</output>
       </div>
       <p data-run-label>loading...</p>
     </section>
     <div class="plot-grid">
       <article class="plot-card"><h3>Gas density</h3><div data-plot="density"></div></article>
       <article class="plot-card"><h3>Gas temperature</h3><div data-plot="temperature"></div></article>
       <article class="plot-card"><h3>Radial velocity</h3><div data-plot="velocity"></div></article>
       <article class="plot-card"><h3>Dark-matter shell positions</h3><div data-plot="dark-matter"></div></article>
     </div>
     <div class="plot-grid plot-grid-3d">
       <article class="plot-card"><h3>3D density slice and velocity quiver</h3><div data-plot3d="density"></div></article>
       <article class="plot-card"><h3>3D temperature slice and velocity quiver</h3><div data-plot3d="temperature"></div></article>
     </div>
     <section class="story" aria-label="Narrative walkthrough">
       <div class="story-copy"><h2>Read the collapse</h2><p>Scroll through the milestones; the plots follow the selected snapshot.</p><p data-dm-count></p></div>
       <div>
         <article class="story-step is-active" data-story-frame="0"><h3>Initial perturbation</h3><p>The gas follows the imposed growing-mode profile while the dark matter is represented by Lagrangian shells.</p></article>
         <article class="story-step" data-story-frame="15"><h3>Infall</h3><p>The radial velocity becomes increasingly negative in the collapsing region and the density contrast grows.</p></article>
         <article class="story-step" data-story-frame="30"><h3>Shock formation</h3><p>The shock marker tracks the entropy-producing transition identified by the saved diagnostic profile.</p></article>
         <article class="story-step" data-story-frame="40"><h3>Virialized region</h3><p>The evolving r_200 marker gives a scale for comparing the hot gas and dark-matter structure at z = 0.</p></article>
       </div>
     </section>
     <section class="provenance"><h2>Reproducibility</h2><pre data-provenance>loading...</pre></section>
   </div>

The exporter is tools/export_interactive_cosmology.py. It loads snapshots
through radhydropy.io.loadhdf5, uses typed RadArray views, excludes ghost
cells, and records configuration and snapshot hashes.

To include compatible comparison runs in the static payload, repeat the
run option, for example ``--run m1e13_z0_inner06=... --run comparison=...``.
