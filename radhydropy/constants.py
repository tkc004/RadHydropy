"""Shared physical constants used across the codebase."""

import unyt


BOLTZMANN_CONSTANT_CGS = float(unyt.kb.to_value(unyt.erg / unyt.K))
PROTON_MASS_CGS = float(unyt.mp.to_value(unyt.g))
SPEED_OF_LIGHT_CGS = float(unyt.c.to_value(unyt.cm / unyt.s))

DEFAULT_SIGMA_GAMMA_CGS_CM2 = float((1.62e-18 * unyt.cm**2).to_value(unyt.cm**2))
DEFAULT_EPSILON_GAMMA_CGS_ERG = float((0.0 * unyt.erg).to_value(unyt.erg))

GRAVITATIONAL_CONSTANT_CGS = float(
    unyt.physical_constants.gravitational_constant.to_value(
        unyt.cm**3 / (unyt.g * unyt.s**2)
    )
)
