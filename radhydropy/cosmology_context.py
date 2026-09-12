"""Immutable cosmological state used by representation-aware fields."""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class CosmologyContext:
    """Snapshot cosmology state required for field conversions.

    ``hubble_parameter_km_s_Mpc`` deliberately remains in observational
    units.  Its code-unit value is derived from ``CodeUnits`` at the point
    where a conversion needs it.
    """

    gamma: float
    cosmology: str = "lambda_cdm"
    scale_factor: float = 1.0
    hubble_parameter_km_s_Mpc: float = 0.0
    isothermal: bool = False

    def __post_init__(self):
        if not isinstance(self.cosmology, str) or not self.cosmology:
            raise ValueError("cosmology must be a non-empty string")

        gamma = self._finite_real(self.gamma, "gamma")
        if gamma < 1.0 or (gamma == 1.0 and not self.isothermal):
            raise ValueError(
                "gamma must be greater than one unless the EOS is isothermal"
            )
        object.__setattr__(self, "gamma", gamma)

        scale_factor = self._finite_real(self.scale_factor, "scale_factor")
        if scale_factor <= 0.0:
            raise ValueError("scale_factor must be finite and positive")
        object.__setattr__(self, "scale_factor", scale_factor)

        hubble = self._finite_real(
            self.hubble_parameter_km_s_Mpc,
            "hubble_parameter_km_s_Mpc",
        )
        if hubble < 0.0:
            raise ValueError(
                "hubble_parameter_km_s_Mpc must be finite and non-negative"
            )
        object.__setattr__(self, "hubble_parameter_km_s_Mpc", hubble)

    @staticmethod
    def _finite_real(value, name):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
