"""Generate a blackbody radiation-spectrum HDF5 file independently."""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.units as units
import h5py
import numpy as np
from astropy.modeling.models import BlackBody


RADIATION_GROUP = "RadiationSpectrum"
EV_TO_ERG = 1.602176634e-12
DEFAULT_EDGES_EV = (13.6, 24.6, 54.4, 10_000.0)
DEFAULT_TEMPERATURE_K = 1.0e5
DEFAULT_INJECTED_PHOTONS_PER_SECOND = 5.0e48
DEFAULT_OUTPUT_NAME = "radiation_spectrum_BB100000K_3groups_HI.h5"


def read_verner96(filename: Path, atomic_number: int, ion: int) -> np.ndarray:
    """Read one Verner & Yakovlev (1996) fit from the local data file."""
    with filename.open() as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                values = line.split()
                if int(values[0]) == atomic_number and int(values[1]) == ion:
                    return np.asarray(values[2:11], dtype=float)
    raise ValueError(f"fit Z={atomic_number}, ion={ion} was not found in {filename}")


def verner96_sigma(energy_ev: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Evaluate the Verner '96 cross-section fit in cm^2."""
    e0, sigma0, ya, power, yw, y0, y1 = parameters[2:9]
    x = energy_ev / e0 - y0
    y = np.sqrt(x * x + y1 * y1)
    return sigma0 * 1.0e-18 * ((x - 1.0) ** 2 + yw**2) * y ** (
        0.5 * power - 5.5
    ) * (1.0 + np.sqrt(y / ya)) ** (-power)


def calculate_groups(edges_ev, temperature_k, parameters, samples_per_group):
    if len(edges_ev) < 2 or np.any(np.diff(edges_ev) <= 0.0):
        raise ValueError("group edges must be strictly increasing")
    if temperature_k <= 0.0 or samples_per_group < 2:
        raise ValueError("temperature must be positive and samples_per_group >= 2")

    blackbody = BlackBody(temperature=temperature_k * units.K)
    threshold_ev = parameters[0]
    norms, norm_energies, sigmas, epsilons = [], [], [], []

    for lower_ev, upper_ev in zip(edges_ev[:-1], edges_ev[1:]):
        lower_ev = max(lower_ev, threshold_ev)
        if lower_ev >= upper_ev:
            norms.append(0.0)
            norm_energies.append(0.0)
            sigmas.append(0.0)
            epsilons.append(0.0)
            continue
        energy = np.geomspace(lower_ev, upper_ev, samples_per_group) * units.eV
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            intensity = blackbody(energy).value * 2.0
        photon_weight = intensity / energy.value
        cross_section = verner96_sigma(energy.value, parameters)
        norm = np.trapz(photon_weight, energy.value)
        norm_energy = np.trapz(intensity, energy.value)
        sigma_integral = np.trapz(photon_weight * cross_section, energy.value)
        epsilon_integral = np.trapz(
            photon_weight * cross_section * (energy.value - threshold_ev),
            energy.value,
        )
        norms.append(norm)
        norm_energies.append(norm_energy)
        sigmas.append(sigma_integral / norm)
        epsilons.append(epsilon_integral / sigma_integral)

    norms = np.asarray(norms)
    ionizing_energy = np.divide(
        np.asarray(norm_energies), norms, out=np.zeros_like(norms), where=norms > 0.0
    )
    return {
        "ionizing_photon_energy_erg": ionizing_energy * EV_TO_ERG,
        "group_sigma_gamma_cm2": np.asarray(sigmas),
        "group_epsilon_gamma_erg": np.asarray(epsilons) * EV_TO_ERG,
        "norm": norms,
    }


def write_spectrum(output, edges_ev, temperature_k, injected_photons,
                   parameters_by_species, samples, include_helium=False):
    values = {
        species: calculate_groups(edges_ev, temperature_k, parameters, samples)
        for species, parameters in parameters_by_species.items()
    }
    hydrogen = values["HI"]
    unit_energy_per_time = (
        1.98841586e33 * units.g * (1.0e5 * units.cm / units.s) ** 3
        / (3.08567758e21 * units.cm)
    )
    unit_erg_per_second = (unit_energy_per_time / (units.erg / units.s)).value
    rates = (
        injected_photons * hydrogen["ionizing_photon_energy_erg"]
        * hydrogen["norm"] / hydrogen["norm"].sum() / unit_erg_per_second
    )
    star_emission_rates = np.concatenate(([1.0e-32], rates))

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        group = handle.create_group(RADIATION_GROUP)
        group.create_dataset("group_edges_eV", data=edges_ev).attrs["units"] = "eV"
        group.create_dataset("ionizing_photon_energy_erg", data=hydrogen[
            "ionizing_photon_energy_erg"]
        ).attrs["units"] = "erg"
        group.create_dataset("star_emission_rates", data=star_emission_rates).attrs[
            "units"
        ] = "internal_energy/time"
        group.create_dataset("group_sigma_gamma_cm2", data=hydrogen[
            "group_sigma_gamma_cm2"]
        ).attrs["units"] = "cm**2"
        group.create_dataset("group_epsilon_gamma_erg", data=hydrogen[
            "group_epsilon_gamma_erg"]
        ).attrs["units"] = "erg"
        if include_helium:
            for species in ("HeI", "HeII"):
                group.create_dataset(
                    f"group_sigma_gamma_{species}_cm2",
                    data=values[species]["group_sigma_gamma_cm2"],
                ).attrs["units"] = "cm**2"
                group.create_dataset(
                    f"group_epsilon_gamma_{species}_erg",
                    data=values[species]["group_epsilon_gamma_erg"],
                ).attrs["units"] = "erg"
        group.attrs["number_of_radiation_groups"] = len(edges_ev) - 1
        group.attrs["number_of_group_edges"] = len(edges_ev)
        group.attrs["stellar_spectrum_type"] = 1
        group.attrs["stellar_spectrum_type_name"] = "blackbody"
        group.attrs["stellar_spectrum_blackbody_temperature_K"] = temperature_k
        group.attrs["absorber"] = "HHe" if include_helium else "HI"
        group.attrs["species"] = "HI,HeI,HeII" if include_helium else "HI"
        group.attrs["description"] = (
            "H/He blackbody spectrum with Verner '96 group averages"
            if include_helium
            else "Pure-hydrogen BB spectrum for the SPHM1RT comparison"
        )


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=directory / DEFAULT_OUTPUT_NAME,
        help=(
            "output HDF5 file (default: "
            "radiation_spectrum_BB100000K_3groups_HI.h5)"
        ),
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--edges", type=float, nargs="+", default=DEFAULT_EDGES_EV)
    parser.add_argument("--injected-photons-per-second", type=float,
                        default=DEFAULT_INJECTED_PHOTONS_PER_SECOND)
    parser.add_argument("--samples-per-group", type=int, default=4000)
    parser.add_argument(
        "--include-helium",
        action="store_true",
        help="include He I and He II cross-section/heating datasets",
    )
    parser.add_argument("--verner-file", type=Path, default=directory / "data" /
                        "cross_section_fits_verner96.dat")
    args = parser.parse_args()
    edges = np.asarray(args.edges, dtype=float)
    parameters = {
        "HI": read_verner96(args.verner_file, atomic_number=1, ion=1),
        "HeI": read_verner96(args.verner_file, atomic_number=2, ion=2),
        "HeII": read_verner96(args.verner_file, atomic_number=2, ion=1),
    }
    write_spectrum(args.output, edges, args.temperature,
                   args.injected_photons_per_second, parameters,
                   args.samples_per_group, include_helium=args.include_helium)
    print(f"Wrote generated radiation spectrum to {args.output}")


if __name__ == "__main__":
    main()
