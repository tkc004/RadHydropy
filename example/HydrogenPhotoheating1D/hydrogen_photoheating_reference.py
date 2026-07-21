"""Reference quantities for the hydrogen photoheating example."""

import numpy as np
import unyt

import radhydropy.hydrogen as rh


def photon_number_density_from_flux(photon_flux):
    """Return optically thin photon density from photon flux."""

    return (photon_flux / rh.SPEED_OF_LIGHT).to(1.0 / unyt.cm**3)


def photoionization_equilibrium_temperature(excess_photoionization_energy):
    """Return ``epsilon_gamma / (3 k_B)``."""

    return (excess_photoionization_energy / (3.0 * unyt.kb)).to(unyt.K)


def thermal_equilibrium_temperature(photoionization_temperature):
    """Return the approximate thermal-equilibrium reference temperature."""

    return 2.0 * photoionization_temperature


def recombination_timescale_at_temperature(hydrogen_number_density, temperature):
    """Return ``1 / (nH alpha_B(T))``."""

    rate = hydrogen_number_density * rh.alpha_B(temperature)
    return (1.0 / rate).to(unyt.yr)


def photoionization_timescale(sigma_gamma, photon_number_density):
    """Return ``1 / (c sigma_gamma n_gamma)``."""

    return (
        1.0
        / (
            rh.SPEED_OF_LIGHT
            * rh.photon_cross_section(sigma_gamma)
            * rh.photon_number_density(photon_number_density)
        )
    ).to(unyt.yr)


def neutral_fraction_reference(
    hydrogen_number_density,
    sigma_gamma,
    photon_number_density,
    photoionization_temperature,
):
    """Return the equilibrium neutral-fraction reference ``tau_i / tau_r``."""

    tau_i = photoionization_timescale(sigma_gamma, photon_number_density)
    tau_r = recombination_timescale_at_temperature(
        hydrogen_number_density,
        photoionization_temperature,
    )
    return {
        'temperature': photoionization_temperature,
        'ionization_timescale': tau_i,
        'recombination_timescale': tau_r,
        'xHI': (tau_i / tau_r).to_value(''),
    }


def timescale_label(symbol, time_scale):
    """Return a compact log10 timescale label."""

    exponent = np.log10(time_scale.to_value(unyt.yr))
    return r'$\tau_%s=10^{%.2f}\ {\rm yr}$' % (symbol, exponent)
