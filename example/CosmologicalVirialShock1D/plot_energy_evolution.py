"""Plot gas energy audits and saved dark-matter energy proxies.

The adiabatic gas run saves an exact cell-integrated energy audit.  Its
dark-matter profile file does not save shell velocities or force work, so the
dark-matter curves below are reconstructed from finite differences of the
saved proper shell radii and are intentionally labelled as proxies.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs_correlation_gas"
PREFIX = "CosmologicalGasCorrelationZ100"

# Code units are kpc, 1e10 Msun, and km/s for this example.
G_CODE = 4.30091e-6 * 1.0e10
KPC_PER_GYR_PER_cgs_KM_S = 1.022712165045695
SOFTENING_cgs_KPC = 0.2


def _shell_edges(radius_comoving_code):
    radius_comoving_code = np.asarray(radius_comoving_code, dtype=float)
    edges = np.empty(radius_comoving_code.size + 1)
    if radius_comoving_code.size == 1:
        return np.array([0.5 * radius_comoving_code[0], 1.5 * radius_comoving_code[0]])
    edges[1:-1] = np.sqrt(radius_comoving_code[:-1] * radius_comoving_code[1:])
    edges[0] = radius_comoving_code[0] ** 2 / edges[1]
    edges[-1] = radius_comoving_code[-1] ** 2 / edges[-2]
    return edges


def _gas_mass_profile(radius_comoving_code, rho_comoving_code):
    edges = _shell_edges(radius_comoving_code)
    mass_comoving_code = np.asarray(rho_comoving_code, dtype=float) * (4.0 * np.pi / 3.0) * np.diff(edges**3)
    return edges, np.maximum(mass_comoving_code, 0.0)


def _dark_matter_proxy(gas, dark_matter):
    times = np.asarray(dark_matter["time_cosmic_Gyr"], dtype=float)
    scale = np.asarray(dark_matter["scale_factor"], dtype=float)
    radius_shell_proper_kpc = np.asarray(dark_matter["radius_proper_kpc"], dtype=float)
    shell_mass_code = np.asarray(dark_matter["mass_code"], dtype=float)
    gas_radius_comoving = np.asarray(gas["radius_comoving_kpc"], dtype=float)
    gas_density = np.asarray(gas["rho_proper_code"], dtype=float)

    kinetic = np.full(times.size, np.nan)
    potential = np.full(times.size, np.nan)
    for i in range(times.size):
        valid = (
            np.isfinite(radius_shell_proper_kpc[i]) & np.isfinite(shell_mass_code[i])
            & (radius_shell_proper_kpc[i] > 0.0) & (shell_mass_code[i] > 0.0)
        )
        radius_comoving_code = radius_shell_proper_kpc[i, valid]
        mass_code = shell_mass_code[i, valid]
        order = np.argsort(radius_comoving_code)
        radius_comoving_code = radius_comoving_code[order]
        mass_code = mass_code[order]
        gas_radius = gas_radius_comoving * scale[i]
        gas_edges, gas_cell_mass = _gas_mass_profile(gas_radius, gas_density[i])
        gas_cumulative = np.concatenate(([0.0], np.cumsum(gas_cell_mass)))
        gas_inside = np.interp(radius_comoving_code, gas_edges, gas_cumulative,
                               left=0.0, right=gas_cumulative[-1])
        dm_inside = np.cumsum(mass_code)
        core_mass = float(np.asarray(dark_matter["central_core_mass"])[i])
        enclosed = gas_inside + dm_inside + core_mass
        potential[i] = -G_CODE * np.sum(mass_code * enclosed / (radius_comoving_code + SOFTENING_cgs_KPC))

        # Shell velocities are absent from the saved file.  A rank-matched
        # finite difference gives a useful global kinetic-energy proxy, but
        # shell crossings and cosmological expansion make it non-conservative.
        if 0 < i < times.size - 1:
            previous = radius_shell_proper_kpc[i - 1]
            following = radius_shell_proper_kpc[i + 1]
            previous = np.sort(previous[np.isfinite(previous) & (previous > 0.0)])
            following = np.sort(following[np.isfinite(following) & (following > 0.0)])
            count = min(radius_comoving_code.size, previous.size, following.size)
            if count:
                dt = times[i + 1] - times[i - 1]
                vel_peculiar_proper_km_s = (following[:count] - previous[:count]) / dt * (
                    1.0 / KPC_PER_GYR_PER_cgs_KM_S
                )
                kinetic[i] = 0.5 * np.sum(mass_code[:count] * vel_peculiar_proper_km_s**2)
        elif times.size > 1:
            neighbor = radius_shell_proper_kpc[i + (1 if i == 0 else -1)]
            neighbor = np.sort(neighbor[np.isfinite(neighbor) & (neighbor > 0.0)])
            count = min(radius_comoving_code.size, neighbor.size)
            dt = abs(times[i + (1 if i == 0 else -1)] - times[i])
            if count and dt > 0.0:
                vel_peculiar_proper_km_s = (radius_comoving_code[:count] - neighbor[:count]) / dt
                vel_peculiar_proper_km_s /= KPC_PER_GYR_PER_cgs_KM_S
                kinetic[i] = 0.5 * np.sum(mass_code[:count] * vel_peculiar_proper_km_s**2)
    return kinetic, potential


def main():
    gas = np.load(OUTPUT / (PREFIX + ".npz"))
    audit = np.load(OUTPUT / (PREFIX + "_EnergyAudit.npz"))
    dark_matter = np.load(OUTPUT / (PREFIX + "_DarkMatterDensities.npz"))

    audit_time = np.asarray(audit["time_cosmic_Gyr"], dtype=float)
    gas_total = np.asarray(audit["total_gas_energy_code"], dtype=float)
    gas_kinetic = np.asarray(audit["kinetic_energy_code"], dtype=float)
    gas_thermal = np.asarray(audit["thermal_energy_code"], dtype=float)
    gas_gravity = np.cumsum(np.asarray(audit["gravitational_work"], dtype=float))
    gas_boundary = np.cumsum(
        np.asarray(audit["hydro_boundary_energy_flux"], dtype=float)
        + np.asarray(audit["background_reservoir_energy_change"], dtype=float)
    )
    gas_budget = gas_total[0] + gas_gravity + gas_boundary
    gas_residual = gas_total - gas_budget

    dm_kinetic, dm_potential = _dark_matter_proxy(gas, dark_matter)
    dm_total = dm_kinetic + dm_potential
    dm_relative_change = (dm_total - dm_total[0]) / max(abs(dm_total[0]), 1.0e-30)

    diagnostics = OUTPUT / (PREFIX + "_EnergyDiagnostics.npz")
    np.savez(
        diagnostics,
        gas_time_cosmic_Gyr=audit_time,
        gas_total=gas_total,
        gas_kinetic=gas_kinetic,
        gas_thermal=gas_thermal,
        gas_cumulative_gravity_work=gas_gravity,
        gas_cumulative_boundary_and_reservoir=gas_boundary,
        gas_budget=gas_budget,
        gas_budget_residual=gas_residual,
        dm_time_cosmic_Gyr=np.asarray(dark_matter["time_cosmic_Gyr"], dtype=float),
        dm_kinetic_proxy=dm_kinetic,
        dm_potential_proxy=dm_potential,
        dm_total_proxy=dm_total,
        dm_relative_total_proxy_change=dm_relative_change,
    )

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 11.0), sharex="col")
    axes[0, 0].plot(audit_time, gas_kinetic, label="kinetic")
    axes[0, 0].plot(audit_time, gas_thermal, label="thermal")
    axes[0, 0].plot(audit_time, gas_total, label="total", lw=2.0, color="black")
    axes[0, 0].set_title("Gas: exact conserved-energy components")
    axes[0, 0].set_ylabel("energy [code units]")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].plot(audit_time, gas_kinetic, label="kinematic")
    axes[1, 0].plot(audit_time, gas_thermal, label="thermal")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("energy [code units]")
    axes[1, 0].legend(frameon=False)
    axes[2, 0].plot(audit_time, gas_gravity, label="cumulative gravity work")
    axes[2, 0].plot(audit_time, gas_boundary, label="boundary + reservoir")
    axes[2, 0].plot(audit_time, gas_residual, label="budget residual", color="black")
    axes[2, 0].set_ylabel("energy [code units]")
    axes[2, 0].set_xlabel("cosmic time [Gyr]")
    axes[2, 0].legend(frameon=False, fontsize=8)

    dm_time_cosmic_Gyr = np.asarray(dark_matter["time_cosmic_Gyr"], dtype=float)
    axes[0, 1].plot(dm_time_cosmic_Gyr, dm_kinetic, label="kinetic proxy")
    axes[0, 1].plot(dm_time_cosmic_Gyr, dm_potential, label="gravity proxy")
    axes[0, 1].plot(dm_time_cosmic_Gyr, dm_total, label="total proxy", color="black", lw=2.0)
    axes[0, 1].set_title("Dark matter: reconstructed proxies")
    axes[0, 1].set_ylabel("energy [code units]")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(dm_time_cosmic_Gyr, dm_kinetic, label="kinematic proxy")
    axes[1, 1].plot(dm_time_cosmic_Gyr, np.zeros_like(dm_time_cosmic_Gyr), label="thermal = 0")
    axes[1, 1].set_ylabel("energy [code units]")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[2, 1].plot(dm_time_cosmic_Gyr, dm_potential, label="potential-energy proxy")
    axes[2, 1].plot(dm_time_cosmic_Gyr, dm_total, label="total proxy", color="black")
    axes[2, 1].set_ylabel("energy [code units]")
    axes[2, 1].set_xlabel("cosmic time [Gyr]")
    axes[2, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle("Cosmological correlation gas + dark-matter energy diagnostics")
    fig.tight_layout()
    figure = OUTPUT / (PREFIX + "_EnergyEvolution.jpg")
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    conservation, axis = plt.subplots(figsize=(8.0, 5.5))
    axis.plot(audit_time, gas_total - gas_total[0], label="measured gas ΔE", lw=2.0)
    axis.plot(audit_time, gas_budget - gas_budget[0], "--", label="budget prediction")
    axis.plot(audit_time, gas_residual, label="gas budget residual")
    axis.set_xlabel("cosmic time [Gyr]")
    axis.set_ylabel("energy [code units]")
    axis.set_title("Gas energy conservation audit")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    conservation.tight_layout()
    conservation_figure = OUTPUT / (PREFIX + "_EnergyConservation.jpg")
    conservation.savefig(conservation_figure, dpi=220)
    plt.close(conservation)

    print("energy figure = %s" % figure)
    print("conservation figure = %s" % conservation_figure)
    print("diagnostics = %s" % diagnostics)
    print("gas final absolute residual = %.8g" % gas_residual[-1])
    print("gas max absolute residual = %.8g" % np.nanmax(np.abs(gas_residual)))
    print("DM proxy relative change = %.8g" % dm_relative_change[-1])
    print("DM energy is a reconstructed proxy: shell velocities and force work were not saved.")


if __name__ == "__main__":
    main()
