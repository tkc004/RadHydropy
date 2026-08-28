"""Plot snapshot-resolved gas-cell and dark-matter-shell energies."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs_correlation_gas_compton_atomic"
PREFIX = "CosmologicalGasCorrelationZ100_ComptonAtomic"


def _signed_norm(values):
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    scale = max(float(np.max(np.abs(finite), initial=0.0)), 1.0e-30)
    return SymLogNorm(linthresh=scale * 1.0e-5, vmin=-scale, vmax=scale)


def _plot(fields, time, radius, filename, title, ylabel):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    radius = np.asarray(radius, dtype=float)
    time_grid = np.broadcast_to(np.asarray(time, dtype=float)[:, None], radius.shape)
    radius_grid = radius
    for axis, (key, label) in zip(axes.flat, fields):
        values = np.asarray(key, dtype=float)
        image = axis.scatter(
            time_grid.ravel(), radius_grid.ravel(), c=values.ravel(),
            s=18, marker="s", linewidths=0, cmap="coolwarm",
            norm=_signed_norm(values),
        )
        axis.set_title(label)
        axis.grid(alpha=0.2)
        fig.colorbar(image, ax=axis, label="energy [code units]")
    for axis in axes[-1]:
        axis.set_xlabel("cosmic time [Gyr]")
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def main():
    data = np.load(OUTPUT / (PREFIX + "_EnergyByCellAndShell.npz"))
    profiles = np.load(OUTPUT / (PREFIX + ".npz"))
    scale = np.asarray(profiles["scale_factor"], dtype=float)

    gas_time = np.asarray(data["gas_time_Gyr"], dtype=float)
    gas_radius = np.asarray(profiles["radius_comoving_kpc"], dtype=float)[None, :] * scale[:, None]
    _plot(
        [
            (data["gas_total_energy"], "total energy"),
            (data["gas_kinetic_energy"], "kinetic energy"),
            (data["gas_thermal_energy"], "thermal energy"),
            (data["gas_delta_thermal_energy"], "thermal change from initial"),
        ],
        gas_time, gas_radius,
        OUTPUT / (PREFIX + "_GasCellEnergy_TimeRadius.jpg"),
        "Gas-cell energy versus time and proper radius", "proper radius [kpc]",
    )

    dm_time = np.asarray(data["dm_time_Gyr"], dtype=float)
    dm_radius = np.asarray(data["dm_radius"], dtype=float) * scale[:, None]
    _plot(
        [
            (data["dm_kinetic_energy"], "kinetic energy"),
            (data["dm_potential_energy"], "potential energy"),
            (data["dm_total_energy"], "total energy"),
            (data["dm_delta_total_energy"], "total change from initial"),
        ],
        dm_time, dm_radius,
        OUTPUT / (PREFIX + "_DarkMatterShellEnergy_TimeRadius.jpg"),
        "Dark-matter-shell energy versus time and proper radius",
        "proper radius [kpc]",
    )


if __name__ == "__main__":
    main()
