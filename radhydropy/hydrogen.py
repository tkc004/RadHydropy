"""Hydrogen thermo-chemistry rates and source terms."""

import numpy as np
import unyt


DEFAULT_SIGMA_GAMMA = 1.62e-18 * unyt.cm**2
DEFAULT_EPSILON_GAMMA = 0.0 * unyt.erg
SPEED_OF_LIGHT = unyt.c.to(unyt.cm / unyt.s)


def _temperature_value_kelvin(temperature):
    if hasattr(temperature, "to_value"):
        return np.asarray(temperature.to_value(unyt.K), dtype=float)
    return np.asarray(temperature, dtype=float)


def _rate_from_temperature(temperature, evaluator, units):
    temp = _temperature_value_kelvin(temperature)
    values = np.zeros_like(temp, dtype=float)
    valid = temp > 0.0
    if np.any(valid):
        values[valid] = evaluator(temp[valid])
    return values * units


def hydrogen_number_density(rho, hydrogen_mass_fraction=1.0):
    """Return hydrogen nuclei number density from mass density."""
    return (hydrogen_mass_fraction * rho / unyt.mp).to(1.0 / unyt.cm**3)


def photon_number_density(ngamma):
    """Return photon number density in ``cm**-3`` units."""
    if hasattr(ngamma, "to"):
        return ngamma.to(1.0 / unyt.cm**3)
    return np.asarray(ngamma, dtype=float) * (1.0 / unyt.cm**3)


def photon_cross_section(sigma_gamma=DEFAULT_SIGMA_GAMMA):
    """Return the photon absorption cross-section in ``cm**2`` units."""
    if hasattr(sigma_gamma, "to"):
        return sigma_gamma.to(unyt.cm**2)
    return sigma_gamma * unyt.cm**2


def photon_excess_energy(epsilon_gamma=DEFAULT_EPSILON_GAMMA):
    """Return photoheating energy per ionization in ``erg`` units."""
    if hasattr(epsilon_gamma, "to"):
        return epsilon_gamma.to(unyt.erg)
    return epsilon_gamma * unyt.erg


def clip_neutral_fraction(xHI):
    """Return neutral hydrogen fraction limited to the physical range."""
    return np.clip(np.asarray(xHI, dtype=float), 0.0, 1.0)


def alpha_B(temperature):
    """Case-B hydrogen recombination coefficient from Hui & Gnedin (1997)."""

    def evaluator(temp):
        lam = 315614.0 / temp
        return (
            2.753e-14
            * lam**1.5
            * (1.0 + (lam / 2.740) ** 0.407) ** -2.242
        )

    return _rate_from_temperature(temperature, evaluator, unyt.cm**3 / unyt.s)


def alpha_A(temperature):
    """Case-A hydrogen recombination coefficient from Hui & Gnedin (1997)."""

    def evaluator(temp):
        lam = 315614.0 / temp
        return (
            1.269e-13
            * lam**1.503
            * (1.0 + (lam / 0.522) ** 0.470) ** -1.923
        )

    return _rate_from_temperature(temperature, evaluator, unyt.cm**3 / unyt.s)


def beta(temperature):
    """Collisional ionization rate coefficient for neutral hydrogen."""

    def evaluator(temp):
        temp5 = temp / 1.0e5
        return (
            1.17e-10
            * temp**0.5
            * np.exp(-157809.1 / temp)
            / (1.0 + temp5**0.5)
        )

    return _rate_from_temperature(temperature, evaluator, unyt.cm**3 / unyt.s)


def gamma_ion_eHI(temperature):
    """Collisional ionization cooling coefficient for neutral hydrogen."""

    def evaluator(temp):
        temp5 = temp / 1.0e5
        return (
            2.54e-21
            * temp**0.5
            * np.exp(-157809.1 / temp)
            / (1.0 + temp5**0.5)
        )

    return _rate_from_temperature(
        temperature,
        evaluator,
        unyt.erg * unyt.cm**3 / unyt.s,
    )


def gamma_line_eHI(temperature):
    """Collisional excitation cooling coefficient for neutral hydrogen."""

    def evaluator(temp):
        temp5 = temp / 1.0e5
        return (
            7.5e-19
            * np.exp(-118348.0 / temp)
            / (1.0 + temp5**0.5)
        )

    return _rate_from_temperature(
        temperature,
        evaluator,
        unyt.erg * unyt.cm**3 / unyt.s,
    )


def gamma_A_eHII(temperature):
    """Case-A recombination cooling coefficient for ionized hydrogen."""

    def evaluator(temp):
        lam = 315614.0 / temp
        return (
            1.778e-29
            * temp
            * lam**1.965
            * (1.0 + (lam / 0.541) ** 0.502) ** -2.697
        )

    return _rate_from_temperature(
        temperature,
        evaluator,
        unyt.erg * unyt.cm**3 / unyt.s,
    )


def gamma_B_eHII(temperature):
    """Case-B recombination cooling coefficient for ionized hydrogen."""

    def evaluator(temp):
        lam = 315614.0 / temp
        return (
            3.435e-30
            * temp
            * lam**1.970
            * (1.0 + (lam / 2.250) ** 0.376) ** -3.720
        )

    return _rate_from_temperature(
        temperature,
        evaluator,
        unyt.erg * unyt.cm**3 / unyt.s,
    )


def gamma_ff_eHII(temperature):
    """Bremsstrahlung cooling coefficient for ionized hydrogen."""

    def evaluator(temp):
        return (
            1.42e-27
            * temp**0.5
            * (1.1 + 0.34 * np.exp(-(5.5 - np.log10(temp)) ** 2 / 3.0))
        )

    return _rate_from_temperature(
        temperature,
        evaluator,
        unyt.erg * unyt.cm**3 / unyt.s,
    )


def photoionization_frequency(ngamma, sigma_gamma=DEFAULT_SIGMA_GAMMA):
    """Return ``c sigma_gamma n_gamma`` for neutral hydrogen."""
    return (
        SPEED_OF_LIGHT
        * photon_cross_section(sigma_gamma)
        * photon_number_density(ngamma)
    ).to(1.0 / unyt.s)


def hydrogen_photon_absorption_rate(
    rho,
    xHI,
    ngamma,
    hydrogen_mass_fraction=1.0,
    sigma_gamma=DEFAULT_SIGMA_GAMMA,
):
    """Return absorbed photon rate per volume."""
    xHI = clip_neutral_fraction(xHI)
    nH = hydrogen_number_density(rho, hydrogen_mass_fraction)
    return (xHI * nH * photoionization_frequency(ngamma, sigma_gamma)).to(
        1.0 / unyt.cm**3 / unyt.s
    )


def hydrogen_radiation_rate(
    rho,
    xHI,
    ngamma,
    hydrogen_mass_fraction=1.0,
    sigma_gamma=DEFAULT_SIGMA_GAMMA,
):
    """Return ``dn_gamma/dt`` from local H I absorption."""
    return -hydrogen_photon_absorption_rate(
        rho,
        xHI,
        ngamma,
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        sigma_gamma=sigma_gamma,
    )


def hydrogen_radiation_analytic_update(
    rho,
    xHI,
    ngamma,
    dt,
    hydrogen_mass_fraction=1.0,
    sigma_gamma=DEFAULT_SIGMA_GAMMA,
):
    """Return analytic photon attenuation holding ``xHI`` and ``nH`` constant."""
    xHI = clip_neutral_fraction(xHI)
    nH = hydrogen_number_density(rho, hydrogen_mass_fraction)
    absorption_frequency = (
        xHI * nH * SPEED_OF_LIGHT * photon_cross_section(sigma_gamma)
    ).to(1.0 / unyt.s)
    dt_value = dt.to_value(unyt.s) if hasattr(dt, "to_value") else float(dt)
    exponent = -dt_value * absorption_frequency.to_value(1.0 / unyt.s)
    return (photon_number_density(ngamma) * np.exp(exponent)).to(1.0 / unyt.cm**3)


def hydrogen_photoheating_rate(
    rho,
    xHI,
    ngamma,
    hydrogen_mass_fraction=1.0,
    sigma_gamma=DEFAULT_SIGMA_GAMMA,
    epsilon_gamma=DEFAULT_EPSILON_GAMMA,
):
    """Return radiation photoheating rate per volume."""
    return (
        photon_excess_energy(epsilon_gamma)
        * hydrogen_photon_absorption_rate(
            rho,
            xHI,
            ngamma,
            hydrogen_mass_fraction=hydrogen_mass_fraction,
            sigma_gamma=sigma_gamma,
        )
    ).to(unyt.erg / unyt.cm**3 / unyt.s)


def hydrogen_neutral_fraction_rate(
    rho,
    temperature,
    xHI,
    hydrogen_mass_fraction=1.0,
    recombination=True,
    collisional_ionization=True,
    ngamma=None,
    sigma_gamma=DEFAULT_SIGMA_GAMMA,
):
    """Return ``dxHI/dt`` from recombination, collisional and photo-ionization."""
    xHI = clip_neutral_fraction(xHI)
    ionized = 1.0 - xHI
    nH = hydrogen_number_density(rho, hydrogen_mass_fraction)
    recombination_coefficient = alpha_B(temperature)
    if not recombination:
        recombination_coefficient = (
            np.zeros_like(recombination_coefficient.value)
            * recombination_coefficient.units
        )
    ionization_coefficient = beta(temperature)
    if not collisional_ionization:
        ionization_coefficient = np.zeros_like(ionization_coefficient.value) * ionization_coefficient.units
    if ngamma is None:
        photoionization_rate = np.zeros_like(xHI, dtype=float) * (1.0 / unyt.s)
    else:
        photoionization_rate = photoionization_frequency(ngamma, sigma_gamma)
    rate = (
        ionized**2 * nH * recombination_coefficient
        - xHI * ionized * nH * ionization_coefficient
        - xHI * photoionization_rate
    )
    return rate.to(1.0 / unyt.s)


def hydrogen_neutral_fraction_implicit_update(
    rho,
    temperature,
    xHI,
    dt,
    hydrogen_mass_fraction=1.0,
    recombination=True,
    collisional_ionization=True,
    ngamma=None,
    sigma_gamma=DEFAULT_SIGMA_GAMMA,
):
    """Return backward-Euler update of the hydrogen neutral fraction."""
    xHI = clip_neutral_fraction(xHI)
    nH = hydrogen_number_density(rho, hydrogen_mass_fraction)
    if recombination:
        recombination_rate = (nH * alpha_B(temperature)).to_value(1.0 / unyt.s)
    else:
        recombination_rate = np.zeros_like(np.asarray(xHI, dtype=float))
    if collisional_ionization:
        ionization_rate = (nH * beta(temperature)).to_value(1.0 / unyt.s)
    else:
        ionization_rate = np.zeros_like(recombination_rate)
    if ngamma is None:
        photoionization_rate = np.zeros_like(recombination_rate)
    else:
        photoionization_rate = photoionization_frequency(
            ngamma,
            sigma_gamma,
        ).to_value(1.0 / unyt.s)
    dt_value = dt.to_value(unyt.s) if hasattr(dt, "to_value") else float(dt)

    a = dt_value * (recombination_rate + ionization_rate)
    b = -(
        1.0
        + dt_value
        * (photoionization_rate + 2.0 * recombination_rate + ionization_rate)
    )
    c = xHI + dt_value * recombination_rate
    discriminant = np.maximum(b**2 - 4.0 * a * c, 0.0)
    denominator = -b + np.sqrt(discriminant)
    updated = np.array(xHI, copy=True, dtype=float)
    updated = np.divide(
        2.0 * c,
        denominator,
        out=updated,
        where=denominator != 0.0,
    )
    return clip_neutral_fraction(updated)


def hydrogen_thermal_rate(
    rho,
    temperature,
    xHI,
    hydrogen_mass_fraction=1.0,
    recombination=True,
    collisional_ionization=True,
    ngamma=None,
    sigma_gamma=DEFAULT_SIGMA_GAMMA,
    epsilon_gamma=DEFAULT_EPSILON_GAMMA,
):
    """Return hydrogen thermal rate per volume, ``rho du/dt``."""
    xHI = clip_neutral_fraction(xHI)
    ionized = 1.0 - xHI
    nH = hydrogen_number_density(rho, hydrogen_mass_fraction)
    eHI_cooling = gamma_line_eHI(temperature)
    if collisional_ionization:
        eHI_cooling += gamma_ion_eHI(temperature)
    eHII_cooling = gamma_ff_eHII(temperature)
    if recombination:
        eHII_cooling += gamma_B_eHII(temperature)
    cooling = nH**2 * (xHI * ionized * eHI_cooling + ionized**2 * eHII_cooling)
    if ngamma is None:
        heating = np.zeros_like(cooling.value) * unyt.erg / unyt.cm**3 / unyt.s
    else:
        heating = hydrogen_photoheating_rate(
            rho,
            xHI,
            ngamma,
            hydrogen_mass_fraction=hydrogen_mass_fraction,
            sigma_gamma=sigma_gamma,
            epsilon_gamma=epsilon_gamma,
        )
    return (heating - cooling).to(unyt.erg / unyt.cm**3 / unyt.s)


def hydrogen_source_terms(
    rho,
    temperature,
    xHI,
    hydrogen_mass_fraction=1.0,
    recombination=True,
    collisional_ionization=True,
    ngamma=None,
    sigma_gamma=DEFAULT_SIGMA_GAMMA,
    epsilon_gamma=DEFAULT_EPSILON_GAMMA,
):
    """Return thermal and neutral-fraction source terms for hydrogen."""
    thermal_rate = hydrogen_thermal_rate(
        rho,
        temperature,
        xHI,
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        recombination=recombination,
        collisional_ionization=collisional_ionization,
        ngamma=ngamma,
        sigma_gamma=sigma_gamma,
        epsilon_gamma=epsilon_gamma,
    )
    neutral_fraction_rate = hydrogen_neutral_fraction_rate(
        rho,
        temperature,
        xHI,
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        recombination=recombination,
        collisional_ionization=collisional_ionization,
        ngamma=ngamma,
        sigma_gamma=sigma_gamma,
    )
    return thermal_rate, neutral_fraction_rate


def pure_hydrogen_mu(xHI, hydrogen_mass_fraction=1.0):
    """Return mean molecular weight for a pure H mixture with neutral fraction ``xHI``."""
    xHI = clip_neutral_fraction(xHI)
    return 1.0 / (hydrogen_mass_fraction * (2.0 - xHI))
