/* Copyright (C) 2026 Tsang Keung Chan */
/* SPDX-License-Identifier: AGPL-3.0 */
(function () {
  "use strict";
  const root = document.querySelector("[data-stellar-wind-explorer]");
  if (!root) return;
  const state = { frame: 0 };
  const fmt = (value, digits) => value == null || !Number.isFinite(value) ? "—" : Number(value).toPrecision(digits || 3);

  function plot(target, frame, key, label, unit, options) {
    const width = 620, height = 300, left = 62, right = 16, top = 16, bottom = 42;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const x = frame.radius_pc, y = frame[key];
    const valid = x.map((value, index) => ({ x: value, y: y[index] })).filter(item => Number.isFinite(item.x) && Number.isFinite(item.y) && (!options.logY || item.y > 0));
    const xMin = options.logX ? Math.min(...valid.map(item => item.x)) : Math.min(...valid.map(item => item.x));
    const xMax = Math.max(...valid.map(item => item.x));
    const yMin = options.logY ? Math.min(...valid.map(item => item.y)) : Math.min(...valid.map(item => item.y));
    const yMax = Math.max(...valid.map(item => item.y));
    const sx = value => options.logX
      ? left + (Math.log(value) - Math.log(xMin)) / (Math.log(xMax) - Math.log(xMin) || 1) * plotWidth
      : left + (value - xMin) / (xMax - xMin || 1) * plotWidth;
    const sy = value => options.logY
      ? top + (Math.log(yMax) - Math.log(value)) / (Math.log(yMax) - Math.log(yMin) || 1) * plotHeight
      : top + (yMax - value) / (yMax - yMin || 1) * plotHeight;
    const path = valid.map((item, index) => (index ? "L" : "M") + sx(item.x).toFixed(1) + "," + sy(item.y).toFixed(1)).join(" ");
    const markers = [
      [frame.ionization_front_pc, "ionization front", "#d64545"],
      [frame.wind_shell_pc, "wind shell", "#805ad5"],
    ].filter(item => Number.isFinite(item[0])).map(item => "<line x1=\"" + sx(item[0]) + "\" x2=\"" + sx(item[0]) + "\" y1=\"" + top + "\" y2=\"" + (top + plotHeight) + "\" stroke=\"" + item[2] + "\" stroke-dasharray=\"5 4\"/><text x=\"" + (sx(item[0]) + 3) + "\" y=\"" + (top + 14) + "\" fill=\"" + item[2] + "\" font-size=\"11\">" + item[1] + "</text>").join("");
    target.innerHTML = "<svg viewBox=\"0 0 " + width + " " + height + "\" role=\"img\" aria-label=\"" + label + "\">"
      + "<rect width=\"" + width + "\" height=\"" + height + "\" fill=\"white\"/>"
      + "<line x1=\"" + left + "\" x2=\"" + left + "\" y1=\"" + top + "\" y2=\"" + (top + plotHeight) + "\" stroke=\"#829ab1\"/>"
      + "<line x1=\"" + left + "\" x2=\"" + (left + plotWidth) + "\" y1=\"" + (top + plotHeight) + "\" y2=\"" + (top + plotHeight) + "\" stroke=\"#829ab1\"/>"
      + "<path d=\"" + path + "\" fill=\"none\" stroke=\"#147d92\" stroke-width=\"2\"/>" + markers
      + "<text x=\"" + (left + plotWidth / 2) + "\" y=\"" + (height - 8) + "\" text-anchor=\"middle\" fill=\"#486581\" font-size=\"12\">radius (pc)</text>"
      + "<text x=\"14\" y=\"" + (top + plotHeight / 2) + "\" transform=\"rotate(-90 14 " + (top + plotHeight / 2) + ")\" text-anchor=\"middle\" fill=\"#486581\" font-size=\"12\">" + label + " (" + unit + ")</text></svg>";
  }

  fetch(root.dataset.dataUrl).then(response => response.json()).then(payload => {
    const run = payload.run, slider = root.querySelector("[data-frame]");
    slider.max = String(run.frames.length - 1);
    root.querySelector("[data-play]").addEventListener("click", () => {
      const timer = setInterval(() => { if (Number(slider.value) >= Number(slider.max)) return clearInterval(timer); slider.value = String(Number(slider.value) + 1); state.frame = Number(slider.value); update(); }, 650);
    });
    slider.addEventListener("input", event => { state.frame = Number(event.target.value); update(); });
    root.querySelectorAll("[data-story-frame]").forEach(step => new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) { state.frame = Number(step.dataset.storyFrame); slider.value = String(state.frame); update(); }
    }), { threshold: .65 }).observe(step));
    update();

    function update() {
      const frame = run.frames[state.frame];
      root.querySelector("[data-frame-label]").textContent = "snapshot " + frame.snapshot + " · t = " + fmt(frame.time_myr) + " Myr";
      plot(root.querySelector("[data-plot=\"density\"]"), frame, "density_cm3", "hydrogen number density", "cm⁻³", { logX: true, logY: true });
      plot(root.querySelector("[data-plot=\"temperature\"]"), frame, "temperature_k", "temperature", "K", { logX: true, logY: true });
      plot(root.querySelector("[data-plot=\"neutral\"]"), frame, "neutral_fraction", "neutral fraction", "xHI", { logX: true, logY: false });
      plot(root.querySelector("[data-plot=\"velocity\"]"), frame, "velocity_km_s", "radial velocity", "km/s", { logX: true, logY: false });
      root.querySelector("[data-front]").textContent = fmt(frame.ionization_front_pc) + " pc";
      root.querySelector("[data-shell]").textContent = fmt(frame.wind_shell_pc) + " pc";
      root.querySelector("[data-pressure-ratio]").textContent = fmt(frame.pressure_ratio);
      root.querySelector("[data-provenance]").textContent = JSON.stringify({ source: run.config_filename, config_sha256: run.config_sha256, snapshots: run.snapshot_count, git_commit: payload.git_commit, git_dirty: payload.git_dirty }, null, 2);
      root.querySelectorAll("[data-story-frame]").forEach(step => step.classList.toggle("is-active", Number(step.dataset.storyFrame) === state.frame));
    }
  }).catch(error => { root.innerHTML = "<p class=\"warning\">The interactive data could not be loaded: " + error.message + "</p>"; });
})();
