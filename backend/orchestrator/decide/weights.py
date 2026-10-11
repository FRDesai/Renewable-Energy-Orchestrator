"""Weight adapter: operator base weights x regime multipliers, renormalised.

PRD Step 2: "The operator sets the base weights and risk appetite; the agent only
shifts them."
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config_loader import Config
from .classifier import ActiveRegime


@dataclass(frozen=True)
class AdaptedCriteria:
    base_weights: dict[str, float]
    final_weights: dict[str, float]
    multiplier_applied: dict[str, float]
    risk_lambda: float
    base_risk_lambda: float
    regimes: tuple[ActiveRegime, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "base_weights": {k: round(v, 4) for k, v in self.base_weights.items()},
            "final_weights": {k: round(v, 4) for k, v in self.final_weights.items()},
            "multiplier_applied": {k: round(v, 4)
                                   for k, v in self.multiplier_applied.items()},
            "risk_lambda": round(self.risk_lambda, 4),
            "base_risk_lambda": round(self.base_risk_lambda, 4),
            "regimes": [r.as_dict() for r in self.regimes],
        }

    def shifted(self) -> dict[str, tuple[float, float]]:
        """Objectives whose weight the agent actually moved, base -> final."""
        return {
            k: (self.base_weights[k], self.final_weights[k])
            for k in self.base_weights
            if abs(self.final_weights[k] - self.base_weights[k]) > 1e-9
        }


def adapt(config: Config, regimes: tuple[ActiveRegime, ...]) -> AdaptedCriteria:
    base = dict(config.objectives.base_weights)
    combined = {k: 1.0 for k in base}
    for regime in regimes:
        for objective, factor in regime.multipliers.items():
            combined[objective] *= factor

    raw = {k: base[k] * combined[k] for k in base}
    total = sum(raw.values())
    if total <= 0:
        final = dict(base)
    else:
        final = {k: v / total for k, v in raw.items()}

    risk = config.objectives.risk
    lam = risk["lambda_base"] + sum(r.risk_aversion for r in regimes)
    lam = float(min(max(lam, risk["lambda_base"]), risk["lambda_max"]))

    return AdaptedCriteria(
        base_weights=base,
        final_weights=final,
        multiplier_applied=combined,
        risk_lambda=lam,
        base_risk_lambda=float(risk["lambda_base"]),
        regimes=regimes,
    )
