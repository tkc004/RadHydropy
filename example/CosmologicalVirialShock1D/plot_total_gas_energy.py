"""Plot gas energy components summed over all physical cells."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs_correlation_gas_compton_atomic"
PREFIX = "CosmologicalGasCorrelationZ100_ComptonAtomic"


def main():
    filename = OUTPUT / (PREFIX + "_EnergyByCellAndShell.npz")
    data = np.load(filename)
    time = np.asarray(data["gas_time_Gyr"], dtype=float)

    total = np.nansum(np.asarray(data["gas_total_energy"], dtype=float), axis=1)
    thermal = np.nansum(np.asarray(data["gas_thermal_energy"], dtype=float), axis=1)
    kinetic = np.nansum(np.asarray(data["gas_kinetic_energy"], dtype=float), axis=1)

    # Gravity is recorded by the solver as signed work per hydro step.  Map
    # its cumulative contribution onto the lower-cadence cell-energy times.
    audit = np.load(OUTPUT / (PREFIX + "_EnergyAudit.npz"))
    audit_time = np.asarray(audit["time_Gyr"], dtype=float)
    cumulative_gravity_work = np.cumsum(
        np.asarray(audit["gravitational_work"], dtype=float)
    )
    gravity_work = np.interp(time, audit_time, cumulative_gravity_work)
    total_with_gravity_work = total + gravity_work

    figure = OUTPUT / (PREFIX + "_TotalGasEnergyEvolution.jpg")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(time, total, "o-", label="total gas energy", linewidth=2.0)
    axis.plot(time, thermal, "o-", label="thermal energy")
    axis.plot(time, kinetic, "o-", label="kinetic energy")
    axis.plot(time, gravity_work, "o-", label="cumulative gravitational work")
    axis.plot(
        time, total_with_gravity_work, "o--",
        label="hydrodynamic total + gravitational work",
        linewidth=2.0,
    )
    axis.set_xlabel("cosmic time [Gyr]")
    axis.set_ylabel("energy [code units]")
    axis.set_title("Gas energy summed over all physical cells")
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    print("total gas energy figure = %s" % figure)
    print("final total, thermal, kinetic = %.8g, %.8g, %.8g" % (
        total[-1], thermal[-1], kinetic[-1]
    ))
    print("final cumulative gravitational work = %.8g" % gravity_work[-1])
    print("final total including gravity work = %.8g" % total_with_gravity_work[-1])


if __name__ == "__main__":
    main()
