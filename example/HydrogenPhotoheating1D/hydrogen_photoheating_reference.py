"""Reference quantities for the hydrogen photoheating example."""

import numpy as np
import unyt

from radhydropy.constants import SPEED_OF_LIGHT_CGS
import radhydropy.thermo_networks.hydrogen as rth


def photon_number_density_from_flux(photon_flux_cgs_cm2_s):
    """Return optically thin photon density from photon flux."""

    speed_of_light = SPEED_OF_LIGHT_CGS * unyt.cm / unyt.s
    return (photon_flux_cgs_cm2_s / speed_of_light).to(1.0 / unyt.cm**3)


def photoionization_equilibrium_temperature(excess_photoionization_energy_cgs_erg):
    """Return ``epsilon_gamma / (3 k_B)``."""

    return (excess_photoionization_energy_cgs_erg / (3.0 * unyt.kb)).to(unyt.K)


def thermal_equilibrium_temperature(temperature_photoionization_cgs_K_unyt):
    """Return the approximate thermal-equilibrium reference temperature."""

    return 2.0 * temperature_photoionization_cgs_K_unyt


def recombination_timescale_at_temperature(
    hydrogen_number_density_cgs_cm3_unyt,
    temperature_proper_unyt,
):
    """Return ``1 / (nH alpha_B(T))``."""

    rate = hydrogen_number_density_cgs_cm3_unyt.to_value(1.0 / unyt.cm**3) * rth._cgs_alpha_B(
        temperature_proper_unyt.to_value(unyt.K)
    )
    return unyt.unyt_quantity(1.0 / rate, unyt.s).to(unyt.yr)


def photoionization_timescale(sigma_gamma_cgs_cm2, photon_number_density_cgs_cm3_unyt):
    """Return ``1 / (c sigma_gamma n_gamma)``."""

    sigma_gamma_cgs_cm2 = sigma_gamma_cgs_cm2.to(unyt.cm**2)
    speed_of_light = SPEED_OF_LIGHT_CGS * unyt.cm / unyt.s
    return (
        1.0
        / (
            speed_of_light
            * sigma_gamma_cgs_cm2
            * photon_number_density_cgs_cm3_unyt
        )
    ).to(unyt.yr)


def neutral_fraction_reference(
    hydrogen_number_density_cgs_cm3_unyt,
    sigma_gamma_cgs_cm2,
    photon_number_density_cgs_cm3_unyt,
    temperature_photoionization_cgs_K_unyt,
):
    """Return the equilibrium neutral-fraction reference ``tau_i / tau_r``."""

    tau_i = photoionization_timescale(
        sigma_gamma_cgs_cm2, photon_number_density_cgs_cm3_unyt
    )
    tau_r = recombination_timescale_at_temperature(
        hydrogen_number_density_cgs_cm3_unyt,
        temperature_photoionization_cgs_K_unyt,
    )
    return {
        'temperature_photoionization_cgs_K_unyt': temperature_photoionization_cgs_K_unyt,
        'time_ionization_proper_unyt': tau_i,
        'time_recombination_proper_unyt': tau_r,
        'xHI': (tau_i / tau_r).to_value(''),
    }


def timescale_label(symbol, time_scale_proper_unyt):
    """Return a compact log10 timescale label."""

    exponent = np.log10(time_scale_proper_unyt.to_value(unyt.yr))
    return r'$\tau_%s=10^{%.2f}\ {\rm yr}$' % (symbol, exponent)
