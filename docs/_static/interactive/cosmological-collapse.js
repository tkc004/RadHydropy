/* Copyright (C) 2026 Tsang Keung Chan */
/* SPDX-License-Identifier: AGPL-3.0 */
(function () {
  "use strict";
  const root = document.querySelector("[data-radhydropy-explorer]");
  if (!root) return;
  const dataUrl = root.dataset.dataUrl || "_static/interactive/cosmological-collapse-data.json";
  const colors = ["#d64545", "#147d92", "#805ad5", "#2f855a"];
  const state = { run: null, frame: 0, compare: false };
  const fmt = (value, digits) => value == null || !Number.isFinite(value) ? "—" : Number(value).toPrecision(digits || 3);
  const finite = value => value != null && Number.isFinite(value) && value > 0;

  function svgPlot(target, series, options) {
    const width = 620, height = 300, left = 62, right = 16, top = 16, bottom = 42;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const values = series.flatMap(item => item.y).filter(Number.isFinite);
    const xValues = series.flatMap(item => item.x).filter(Number.isFinite);
    if (!values.length || !xValues.length) { target.textContent = "No finite data"; return; }
    const xMin = Math.min(...xValues), xMax = Math.max(...xValues);
    let yMin = options.logY ? Math.min(...values.filter(finite)) : Math.min(...values);
    let yMax = Math.max(...values);
    if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) { target.textContent = "No finite data"; return; }
    if (options.logY) yMin = Math.max(yMin, yMax * 1e-8);
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const sx = x => left + (x - xMin) / (xMax - xMin || 1) * plotWidth;
    const sy = y => options.logY
      ? top + (Math.log(yMax) - Math.log(Math.max(y, yMin))) / (Math.log(yMax) - Math.log(yMin) || 1) * plotHeight
      : top + (yMax - y) / (yMax - yMin) * plotHeight;
    const line = item => item.x.map((x, i) => Number.isFinite(item.y[i]) && (!options.logY || finite(item.y[i])) ? (i ? "L" : "M") + sx(x).toFixed(1) + "," + sy(item.y[i]).toFixed(1) : "").filter(Boolean).join(" ");
    const ticks = [0, .25, .5, .75, 1];
    target.innerHTML = "<svg viewBox=\"0 0 " + width + " " + height + "\" role=\"img\" aria-label=\"" + options.label + "\">"
      + "<rect x=\"0\" y=\"0\" width=\"" + width + "\" height=\"" + height + "\" fill=\"white\"/>"
      + ticks.map(t => "<line x1=\"" + left + "\" x2=\"" + (left + plotWidth) + "\" y1=\"" + (top + t * plotHeight) + "\" y2=\"" + (top + t * plotHeight) + "\" stroke=\"#e6eef5\"/>").join("")
      + "<line x1=\"" + left + "\" x2=\"" + left + "\" y1=\"" + top + "\" y2=\"" + (top + plotHeight) + "\" stroke=\"#829ab1\"/>"
      + "<line x1=\"" + left + "\" x2=\"" + (left + plotWidth) + "\" y1=\"" + (top + plotHeight) + "\" y2=\"" + (top + plotHeight) + "\" stroke=\"#829ab1\"/>"
      + series.map(item => "<path d=\"" + line(item) + "\" fill=\"none\" stroke=\"" + item.color + "\" stroke-width=\"2\"/>").join("")
      + series.flatMap(item => (item.markers || []).filter(marker => Number.isFinite(marker.x)).map(marker => "<line x1=\"" + sx(marker.x) + "\" x2=\"" + sx(marker.x) + "\" y1=\"" + top + "\" y2=\"" + (top + plotHeight) + "\" stroke=\"" + (marker.color || item.color) + "\" stroke-dasharray=\"5 4\"/><text x=\"" + (sx(marker.x) + 3) + "\" y=\"" + (top + 14) + "\" fill=\"" + (marker.color || item.color) + "\" font-size=\"11\">" + marker.label + "</text>")).join("")
      + "<text x=\"" + (left + plotWidth / 2) + "\" y=\"" + (height - 8) + "\" text-anchor=\"middle\" fill=\"#486581\" font-size=\"12\">" + options.xLabel + "</text>"
      + "<text x=\"14\" y=\"" + (top + plotHeight / 2) + "\" transform=\"rotate(-90 14 " + (top + plotHeight / 2) + ")\" text-anchor=\"middle\" fill=\"#486581\" font-size=\"12\">" + options.yLabel + "</text>"
      + "<text x=\"" + left + "\" y=\"" + (height - 8) + "\" fill=\"#486581\" font-size=\"10\">" + fmt(xMin, 2) + "</text>"
      + "<text x=\"" + (left + plotWidth) + "\" y=\"" + (height - 8) + "\" text-anchor=\"end\" fill=\"#486581\" font-size=\"10\">" + fmt(xMax, 2) + "</text></svg>";
  }

  function init(payload) {
    const runs = payload.runs || {}, runNames = Object.keys(runs);
    if (!runNames.length) throw new Error("interactive data contains no runs");
    state.run = payload.default_run && runs[payload.default_run] ? payload.default_run : runNames[0];
    const runSelect = root.querySelector("[data-run]");
    const compareSelect = root.querySelector("[data-compare]");
    runNames.forEach(name => {
      runSelect.add(new Option(name, name));
      compareSelect.add(new Option(name, name));
    });
    runSelect.value = state.run;
    compareSelect.value = runNames.length > 1 ? runNames[1] : state.run;
    const slider = root.querySelector("[data-frame]");
    slider.max = String(runs[state.run].frames.length - 1);
    runSelect.addEventListener("change", () => { state.run = runSelect.value; state.frame = 0; slider.max = String(runs[state.run].frames.length - 1); slider.value = "0"; update(); });
    slider.addEventListener("input", event => { state.frame = Number(event.target.value); update(); });
    root.querySelector("[data-show-compare]").addEventListener("change", event => { state.compare = event.target.checked; update(); });
    root.querySelector("[data-play]").addEventListener("click", () => {
      const timer = setInterval(() => { if (Number(slider.value) >= Number(slider.max)) return clearInterval(timer); slider.value = String(Number(slider.value) + 1); state.frame = Number(slider.value); update(); }, 650);
    });
    root.querySelectorAll("[data-story-frame]").forEach(step => {
      new IntersectionObserver(entries => entries.forEach(entry => { if (entry.isIntersecting) { state.frame = Number(step.dataset.storyFrame); slider.value = String(state.frame); update(); } }), { threshold: .65 }).observe(step);
    });
    compareSelect.addEventListener("change", update);
    update();

    function update() {
      const run = runs[state.run], frame = run.frames[Math.min(state.frame, run.frames.length - 1)];
      root.querySelector("[data-frame-label]").textContent = "snapshot " + frame.snapshot + " · t = " + fmt(frame.time_cosmic_gyr) + " Gyr · z = " + fmt(frame.redshift);
      root.querySelector("[data-run-label]").textContent = state.run + " · " + run.cosmology_type + " · " + run.grid_cells + " active cells";
      const selected = [{ run: run, frame: frame, color: colors[0] }];
      if (state.compare && compareSelect.value !== state.run && runs[compareSelect.value]) {
        const comparison = runs[compareSelect.value];
        selected.push({ run: comparison, frame: comparison.frames[Math.min(state.frame, comparison.frames.length - 1)], color: colors[1] });
      }
      const panels = [
        ["density", "Density", "g/cm³", true, "density_g_cm3"],
        ["temperature", "Temperature", "K", true, "temperature_k"],
        ["velocity", "Radial velocity", "km/s", false, "velocity_km_s"],
      ];
      panels.forEach(([name, label, unit, logY, yKey]) => {
        svgPlot(root.querySelector("[data-plot=\"" + name + "\"]"), selected.map(item => ({
          x: item.frame.radius_comoving_kpc, y: item.frame[yKey], color: item.color,
          markers: name === "density" ? [{ x: item.frame.shock_comoving_kpc, label: "shock" }, { x: item.frame.rvir_comoving_kpc, label: "r₂₀₀", color: "#805ad5" }] : [],
        })), { label: label, logY: logY, xLabel: "comoving radius (kpc)", yLabel: label + " (" + unit + ")" });
      });
      svgPlot(root.querySelector("[data-plot=\"dark-matter\"]"), selected.map(item => {
        const shells = item.frame.dark_matter_radius_proper_kpc || [];
        return { x: shells, y: shells.map((_, index) => index / Math.max(shells.length - 1, 1)), color: item.color };
      }), { label: "Dark-matter shell positions", logY: false, xLabel: "proper radius (kpc)", yLabel: "shell order" });
      root.querySelector("[data-dm-count]").textContent = (frame.dark_matter_radius_proper_kpc || []).length.toLocaleString() + " dark-matter shells";
      root.querySelector("[data-provenance]").textContent = JSON.stringify({ source: run.config_filename, config_sha256: run.config_sha256, snapshots: run.snapshot_count, git_commit: payload.git_commit, git_dirty: payload.git_dirty }, null, 2);
      root.querySelectorAll("[data-story-frame]").forEach(step => step.classList.toggle("is-active", Number(step.dataset.storyFrame) === state.frame));
    }
  }
  fetch(dataUrl).then(response => { if (!response.ok) throw new Error("data request failed (" + response.status + ")"); return response.json(); }).then(init).catch(error => { root.innerHTML = "<p class=\"warning\">The interactive data could not be loaded: " + error.message + "</p>"; });
})();
