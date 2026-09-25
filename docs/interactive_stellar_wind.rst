Interactive Stellar-Wind Stromgren Sphere
=========================================

This page replays the Dynamic Stromgren Sphere Photoheating 20 pc example
with a central stellar wind. The saved snapshots show how the wind cavity,
photoheated gas, dense shell, and ionization front evolve together.

.. raw:: html

   <link rel="stylesheet" href="_static/interactive/cosmological-collapse.css">
   <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
   <script src="_static/interactive/stellar-wind.js" defer></script>
   <div class="radhydropy-explorer" data-stellar-wind-explorer data-data-url="_static/interactive/stellar-wind-data.json">
     <section class="hero">
       <p>RADHYDROpy · STATIC SIMULATION REPLAY</p>
       <h2>A stellar wind inside a photoionized sphere</h2>
       <p>Use the time slider to follow the wind shell, ionization front, temperature structure, and radial flow in the 20 pc spherical calculation.</p>
     </section>
     <section class="controls" aria-label="Interactive controls">
       <div class="control-row">
         <button type="button" data-play>Play snapshots</button>
         <label for="wind-frame">Time</label>
         <input id="wind-frame" type="range" min="0" value="0" data-frame>
         <output data-frame-label>loading...</output>
       </div>
       <div class="control-row">
         <strong>Ionization front: <span data-front>—</span></strong>
         <strong>Wind shell: <span data-shell>—</span></strong>
         <strong>Wind/gas pressure: <span data-pressure-ratio>—</span></strong>
       </div>
     </section>
     <div class="plot-grid">
       <article class="plot-card"><h3>Hydrogen number density</h3><div data-plot="density"></div></article>
       <article class="plot-card"><h3>Temperature</h3><div data-plot="temperature"></div></article>
       <article class="plot-card"><h3>Neutral fraction</h3><div data-plot="neutral"></div></article>
       <article class="plot-card"><h3>Radial velocity</h3><div data-plot="velocity"></div></article>
     </div>
     <div class="plot-grid plot-grid-3d">
       <article class="plot-card"><h3>3D density slice and velocity quiver</h3><p>Central planar slice reconstructed from the spherical profile. Surface height follows the normalized enclosed-gas gravitational potential; color is log₁₀ density.</p><div data-plot3d="density"></div></article>
       <article class="plot-card"><h3>3D temperature slice and velocity quiver</h3><p>The same slice colored by log₁₀ temperature, with the radial velocity field shown as arrows.</p><div data-plot3d="temperature"></div></article>
     </div>
     <section class="story" aria-label="Narrative walkthrough">
       <div class="story-copy"><h2>Read the interaction</h2><p>Scroll through the milestones; the radial profiles follow the selected snapshot.</p></div>
       <div>
         <article class="story-step is-active" data-story-frame="0"><h3>Wind injection</h3><p>The inner boundary supplies a 1000 km/s wind with the configured mass-loss rate.</p></article>
         <article class="story-step" data-story-frame="1"><h3>Photoionization</h3><p>The central source creates an ionized region whose neutral-fraction transition defines the ionization front.</p></article>
         <article class="story-step" data-story-frame="3"><h3>Shell formation</h3><p>Wind ram pressure compresses gas into a dense shell while the photoheated region remains near 10⁴ K or hotter.</p></article>
         <article class="story-step" data-story-frame="5"><h3>Pressure competition</h3><p>The pressure ratio records whether wind ram pressure or photoheated-gas pressure dominates at the shell.</p></article>
       </div>
     </section>
     <section class="provenance"><h2>Reproducibility</h2><pre data-provenance>loading...</pre></section>
   </div>

The data exporter is tools/export_stellar_wind_interactive.py. It loads the
HDF5 snapshots through radhydropy.io.loadhdf5 and uses physical active-cell
fields, excluding ghost cells.
