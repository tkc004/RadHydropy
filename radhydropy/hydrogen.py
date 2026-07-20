"""Hydrogen thermo-chemistry rates and source terms."""

import numpy as np
import unyt


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


def hydrogen_neutral_fraction_rate(
    rho,
    temperature,
    xHI,
    hydrogen_mass_fraction=1.0,
    collisional_ionization=True,
):
    """Return ``dxHI/dt`` from recombination and collisional ionization."""
    xHI = clip_neutral_fraction(xHI)
    ionized = 1.0 - xHI
    nH = hydrogen_number_density(rho, hydrogen_mass_fraction)
    ionization_coefficient = beta(temperature)
    if not collisional_ionization:
        ionization_coefficient = np.zeros_like(ionization_coefficient.value) * ionization_coefficient.units
    rate = ionized**2 * nH * alpha_B(temperature) - xHI * ionized * nH * ionization_coefficient
    return rate.to(1.0 / unyt.s)


def hydrogen_neutral_fraction_implicit_update(
    rho,
    temperature,
    xHI,
    dt,
    hydrogen_mass_fraction=1.0,
    collisional_ionization=True,
):
    """Return backward-Euler update of the hydrogen neutral fraction."""
    xHI = clip_neutral_fraction(xHI)
    nH = hydrogen_number_density(rho, hydrogen_mass_fraction)
    recombination_rate = (nH * alpha_B(temperature)).to_value(1.0 / unyt.s)
    if collisional_ionization:
        ionization_rate = (nH * beta(temperature)).to_value(1.0 / unyt.s)
    else:
        ionization_rate = np.zeros_like(recombination_rate)
    dt_value = dt.to_value(unyt.s) if hasattr(dt, "to_value") else float(dt)

    a = dt_value * (recombination_rate + ionization_rate)
    b = -(1.0 + dt_value * (2.0 * recombination_rate + ionization_rate))
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
    collisional_ionization=True,
):
    """Return hydrogen cooling rate per volume, ``rho du/dt``."""
    xHI = clip_neutral_fraction(xHI)
    ionized = 1.0 - xHI
    nH = hydrogen_number_density(rho, hydrogen_mass_fraction)
    eHI_cooling = gamma_line_eHI(temperature)
    if collisional_ionization:
        eHI_cooling += gamma_ion_eHI(temperature)
    eHII_cooling = gamma_ff_eHII(temperature) + gamma_B_eHII(temperature)
    cooling = nH**2 * (xHI * ionized * eHI_cooling + ionized**2 * eHII_cooling)
    return (-cooling).to(unyt.erg / unyt.cm**3 / unyt.s)


def hydrogen_source_terms(
    rho,
    temperature,
    xHI,
    hydrogen_mass_fraction=1.0,
    collisional_ionization=True,
):
    """Return thermal and neutral-fraction source terms for hydrogen."""
    thermal_rate = hydrogen_thermal_rate(
        rho,
        temperature,
        xHI,
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        collisional_ionization=collisional_ionization,
    )
    neutral_fraction_rate = hydrogen_neutral_fraction_rate(
        rho,
        temperature,
        xHI,
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        collisional_ionization=collisional_ionization,
    )
    return thermal_rate, neutral_fraction_rate


def pure_hydrogen_mu(xHI, hydrogen_mass_fraction=1.0):
    """Return mean molecular weight for a pure H mixture with neutral fraction ``xHI``."""
    xHI = clip_neutral_fraction(xHI)
    return 1.0 / (hydrogen_mass_fraction * (2.0 - xHI))
