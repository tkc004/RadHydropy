/* Copyright (C) 2026 Tsang Keung Chan */
/* SPDX-License-Identifier: AGPL-3.0 */
(function () {
  "use strict";
  const root = document.querySelector("[data-stellar-wind-explorer]");
  if (!root) return;
  const state = { frame: 0 };
  const fmt = (value, digits) => value == null || !Number.isFinite(value) ? "—" : Number(value).toPrecision(digits || 3);

  function interpolate(radius, values, point) {
    if (point <= radius[0]) return values[0];
    if (point >= radius[radius.length - 1]) return values[values.length - 1];
    let upper = 1;
    while (upper < radius.length && radius[upper] < point) upper += 1;
    const lower = upper - 1;
    const weight = (Math.log(point) - Math.log(radius[lower])) / (Math.log(radius[upper]) - Math.log(radius[lower]));
    const first = values[lower], second = values[upper];
    if (first > 0 && second > 0) return Math.exp(Math.log(first) * (1 - weight) + Math.log(second) * weight);
    return first * (1 - weight) + second * weight;
  }

  function gravityModel(frame) {
    const radius = frame.radius_pc;
    const density = frame.density_cm3;
    const protonMass = 1.67262192369e-24;
    const parsec = 3.085677581e18;
    const shellMass = [];
    for (let index = 0; index < radius.length; index += 1) {
      const inner = index === 0 ? 0 : (radius[index - 1] + radius[index]) / 2;
      const outer = index === radius.length - 1 ? radius[index] : (radius[index] + radius[index + 1]) / 2;
      shellMass.push(density[index] * protonMass * (4 * Math.PI / 3) * (Math.pow(outer * parsec, 3) - Math.pow(inner * parsec, 3)));
    }
    const outerRadius = radius[radius.length - 1];
    const potentialMagnitude = point => shellMass.reduce(
      (sum, mass, index) => sum + mass / Math.max(point, radius[index]),
      0,
    );
    const centralPotential = Math.max(potentialMagnitude(0), 1e-30);
    return point => {
      const wellDepth = potentialMagnitude(Math.max(point, 0)) / centralPotential;
      return -0.42 * outerRadius * wellDepth;
    };
  }

  function renderSlice(target, frame, field, colorscale) {
    if (!window.Plotly) {
      target.textContent = "The 3D slice requires Plotly.js to be loaded.";
      return;
    }
    const radius = frame.radius_pc;
    const values = frame[field];
    const velocities = frame.velocity_km_s;
    const count = 39;
    const extent = Math.min(20, radius[radius.length - 1]);
    const axis = Array.from({ length: count }, (_, index) => -extent + 2 * extent * index / (count - 1));
    const gravityHeight = gravityModel(frame);
    const surface = [], colors = [], quiverX = [], quiverY = [], quiverZ = [];
    const logVelocity = value => Math.sign(value) * Math.log1p(Math.abs(value));
    const maximumLogVelocity = Math.max(...velocities.map(logVelocity).map(value => Math.abs(value)), 1e-12);
    const arrowScale = 0.55 * extent / maximumLogVelocity;
    const quiverLift = 0.025 * extent;
    for (let row = 0; row < count; row += 1) {
      const surfaceRow = [], colorRow = [];
      for (let column = 0; column < count; column += 1) {
        const x = axis[column], y = axis[row], distance = Math.hypot(x, y);
        const safeDistance = Math.max(distance, radius[0]);
        const value = interpolate(radius, values, safeDistance);
        const velocity = interpolate(radius, velocities, safeDistance);
        const height = gravityHeight(distance);
        surfaceRow.push(height);
        colorRow.push(value > 0 ? Math.log10(value) : null);
        if (row % 4 === 0 && column % 4 === 0 && distance > radius[0]) {
          const visualVelocity = logVelocity(velocity);
          const endX = x + visualVelocity * x / distance * arrowScale;
          const endY = y + visualVelocity * y / distance * arrowScale;
          const deltaX = endX - x, deltaY = endY - y;
          const length = Math.hypot(deltaX, deltaY);
          const head = 0.16 * length;
          const perpendicularX = -deltaY / Math.max(length, 1e-12) * head * 0.6;
          const perpendicularY = deltaX / Math.max(length, 1e-12) * head * 0.6;
          quiverX.push(x, endX, null);
          quiverY.push(y, endY, null);
          quiverZ.push(height + quiverLift, gravityHeight(Math.hypot(endX, endY)) + quiverLift, null);
          quiverX.push(endX, endX - deltaX / Math.max(length, 1e-12) * head + perpendicularX, null);
          quiverY.push(endY, endY - deltaY / Math.max(length, 1e-12) * head + perpendicularY, null);
          quiverZ.push(gravityHeight(Math.hypot(endX, endY)) + quiverLift, gravityHeight(Math.hypot(endX - deltaX / Math.max(length, 1e-12) * head + perpendicularX, endY - deltaY / Math.max(length, 1e-12) * head + perpendicularY)) + quiverLift, null);
          quiverX.push(endX, endX - deltaX / Math.max(length, 1e-12) * head - perpendicularX, null);
          quiverY.push(endY, endY - deltaY / Math.max(length, 1e-12) * head - perpendicularY, null);
          quiverZ.push(gravityHeight(Math.hypot(endX, endY)) + quiverLift, gravityHeight(Math.hypot(endX - deltaX / Math.max(length, 1e-12) * head - perpendicularX, endY - deltaY / Math.max(length, 1e-12) * head - perpendicularY)) + quiverLift, null);
        }
      }
      surface.push(surfaceRow);
      colors.push(colorRow);
    }
    window.Plotly.react(target, [{
      type: "surface", x: axis, y: axis, z: surface, surfacecolor: colors,
      colorscale: colorscale, colorbar: { title: field === "density_cm3" ? "log10(cm⁻³)" : "log10(K)" },
      hovertemplate: "x=%{x:.2f} pc<br>y=%{y:.2f} pc<br>log value=%{surfacecolor:.3g}<extra></extra>",
    }, {
      type: "scatter3d", mode: "lines", x: quiverX, y: quiverY, z: quiverZ,
      line: { color: "#102a43", width: 6 }, name: "velocity quiver", hoverinfo: "skip",
    }], {
      margin: { l: 0, r: 0, t: 10, b: 0 },
      scene: {
        aspectmode: "cube", xaxis: { title: "x (pc)" }, yaxis: { title: "y (pc)" },
        zaxis: { title: "normalized gravitational potential" },
        camera: { eye: { x: 1.45, y: 1.45, z: 1.15 } },
      }, showlegend: false,
    }, { responsive: true, displaylogo: false });
  }

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
      renderSlice(root.querySelector("[data-plot3d=\"density\"]"), frame, "density_cm3", "Viridis");
      renderSlice(root.querySelector("[data-plot3d=\"temperature\"]"), frame, "temperature_k", "Inferno");
      root.querySelector("[data-front]").textContent = fmt(frame.ionization_front_pc) + " pc";
      root.querySelector("[data-shell]").textContent = fmt(frame.wind_shell_pc) + " pc";
      root.querySelector("[data-pressure-ratio]").textContent = fmt(frame.pressure_ratio);
      root.querySelector("[data-provenance]").textContent = JSON.stringify({ source: run.config_filename, config_sha256: run.config_sha256, snapshots: run.snapshot_count, git_commit: payload.git_commit, git_dirty: payload.git_dirty }, null, 2);
      root.querySelectorAll("[data-story-frame]").forEach(step => step.classList.toggle("is-active", Number(step.dataset.storyFrame) === state.frame));
    }
  }).catch(error => { root.innerHTML = "<p class=\"warning\">The interactive data could not be loaded: " + error.message + "</p>"; });
})();
