"""Post-recombination residual-ionization chemistry test.

This is a homogeneous, source-only benchmark.  It integrates the same
hydrogen chemistry rate used by RadHydropy after z=100, with Compton heating
and atomic cooling enabled.  There is deliberately no radiation field or
photoionization/reionization source.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from radhydropy.constants import BOLTZMANN_CONSTANT_CGS, PROTON_MASS_CGS
from radhydropy.thermo_networks.hydrogen import (
    ionization_fraction_rate,
    thermal_rate,
)


CONFIG = EXAMPLE_ROOT / "cosmological_residual_ionization1d.yaml"
SECONDS_PER_GYR = 1.0e9 * 365.25 * 86400.0


def evolve(runparams, icparams):
    """Integrate xHI and temperature with the RadHydropy source equations."""
    gamma = float(runparams["hydrodynamics"]["gamma"])
    hydrogen_fraction = float(runparams["chemistry"]["hydrogen_mass_fraction"])
    nH0 = float(icparams["present_hydrogen_density_cgs_cm3"])
    t_ref_s = float(runparams["gravity"]["cosmology_t_ref"].to_value("s"))
    cmb0 = float(runparams["thermochemistry"]["cmb_temperature_0"].to_value("K"))
    z_initial = float(icparams["initial_redshift"])
    z_final = float(icparams["final_redshift"])
    t_initial = t_ref_s * (1.0 / (1.0 + z_initial)) ** 1.5
    t_final = t_ref_s * (1.0 / (1.0 + z_final)) ** 1.5
    initial_xe = float(icparams["initial_xe"])
    initial_temperature = float(icparams["initial_temperature"].to_value("K"))

    def rates(time_s, values):
        xHI = float(np.clip(values[0], 1.0e-12, 1.0 - 1.0e-12))
        temperature = max(float(values[1]), 1.0e-6)
        scale_factor = (time_s / t_ref_s) ** (2.0 / 3.0)
        redshift = 1.0 / scale_factor - 1.0
        nH = nH0 * scale_factor ** -3
        rho = nH * PROTON_MASS_CGS / hydrogen_fraction
        state = {
            "rho_cgs_g_cm3": np.asarray([rho]),
            "temperature_cgs_K": np.asarray([temperature]),
            "xHI": np.asarray([xHI]),
            "hydrogen_mass_fraction": hydrogen_fraction,
            "recombination": True,
            "collisional_ionization": False,
            "atomic_cooling": True,
            "sigma_gamma_cgs_cm2": 0.0,
            "epsilon_gamma_cgs_erg": 0.0,
            "compton_cmb_enabled": True,
            "compton_cmb_redshift": redshift,
            "cmb_temperature_0_cgs_K": cmb0,
            "alpha_B_cgs_cm3_s": None,
            "beta_cgs_cm3_s": None,
        }
        dxhi_dt = float(ionization_fraction_rate(state, None)[0])
        q = float(thermal_rate(state, None)[0])
        mu = 1.0 / (hydrogen_fraction * (2.0 - xHI))
        hubble = 2.0 / (3.0 * time_s)
        adiabatic = -2.0 * hubble * temperature
        source = (gamma - 1.0) * mu * PROTON_MASS_CGS * q / (rho * BOLTZMANN_CONSTANT_CGS)
        return [dxhi_dt, adiabatic + source]

    solution = solve_ivp(
        rates,
        (t_initial, t_final),
        [1.0 - initial_xe, initial_temperature],
        t_eval=np.linspace(t_initial, t_final, int(icparams["output_points"])),
        rtol=2.0e-9,
        atol=[1.0e-12, 1.0e-5],
        method="BDF",
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    time_s = solution.t
    scale_factor = (time_s / t_ref_s) ** (2.0 / 3.0)
    redshift = 1.0 / scale_factor - 1.0
    xe = 1.0 - np.clip(solution.y[0], 0.0, 1.0)
    temperature = np.maximum(solution.y[1], 0.0)
    return redshift, xe, temperature


def main():
    from example import example_utils as eu
    config = eu.load_nested_example_config(CONFIG)
    runparams, icparams = config['par'], config['initial_condition']
    redshift, xe, temperature = evolve(runparams, icparams)
    # The integration proceeds from high to low redshift; retain that order
    # so the horizontal axis also reads forward in cosmic time.
    order = np.argsort(-redshift)
    redshift, xe, temperature = redshift[order], xe[order], temperature[order]
    output = Path(runparams["output"]["savedir"])
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "CosmologicalResidualIonization1D_History.npz",
             redshift=redshift, xe=xe, temperature_cgs_K=temperature)

    figure = output / "CosmologicalResidualIonization1D_xe_temperature.jpg"
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(redshift, xe, color="tab:blue")
    axes[0].set_xlabel("redshift z")
    axes[0].set_ylabel(r"electron fraction $x_e=1-x_{HI}$")
    axes[0].set_yscale("log")
    axes[0].set_xlim(redshift[0], redshift[-1])
    axes[0].grid(alpha=0.25)
    axes[1].plot(redshift, temperature, color="tab:red")
    axes[1].set_xlabel("redshift z")
    axes[1].set_ylabel("gas temperature [K]")
    axes[1].set_yscale("log")
    axes[1].set_xlim(redshift[0], redshift[-1])
    axes[1].grid(alpha=0.25)
    fig.suptitle("Post-recombination chemistry: Compton + atomic cooling, no reionization")
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"z={redshift[0]:.3g} -> {redshift[-1]:.3g}")
    print(f"xe={xe[0]:.6e} -> {xe[-1]:.6e}")
    print(f"figure = {figure}")


if __name__ == "__main__":
    main()
