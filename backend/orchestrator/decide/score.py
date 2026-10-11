"""Step 3 scoring, Step 4 hysteresis, and the Pareto front.

PRD: "Each objective is min-max normalized across the surviving clusters (with a floor
on the range so trivial differences are not amplified), multiplied by its weight and
summed. The lowest score wins."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config_loader import Config
from ..domain.actions import ActionCluster
from .dispatch import OBJECTIVES, DispatchOutcome
from .weights import AdaptedCriteria


def cvar(values: np.ndarray, quantile: float, axis: int = -1) -> np.ndarray:
    """Mean of the worst (1 - quantile) tail, where "worst" means largest.

    Every objective here is minimised, so the upper tail is the bad one.
    """
    n = values.shape[axis]
    k = max(1, int(np.ceil((1.0 - quantile) * n)))
    ordered = np.sort(values, axis=axis)
    return np.take(ordered, range(n - k, n), axis=axis).mean(axis=axis)


@dataclass(frozen=True)
class ClusterScore:
    index: int
    cluster: ActionCluster
    score: float
    objectives: dict[str, float]          # risk-adjusted, in native units
    normalized: dict[str, float]
    contributions: dict[str, float]       # weight x normalized
    cost_p10: float
    cost_p50: float
    cost_p90: float
    unserved_mwh: float
    survived_guardrails: bool
    on_pareto_front: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "cluster": self.cluster.as_dict(),
            "score": round(self.score, 6),
            "objectives": {k: round(v, 3) for k, v in self.objectives.items()},
            "contributions": {k: round(v, 5) for k, v in self.contributions.items()},
            "cost_p10": round(self.cost_p10, 1),
            "cost_p50": round(self.cost_p50, 1),
            "cost_p90": round(self.cost_p90, 1),
            "unserved_mwh": round(self.unserved_mwh, 4),
            "survived_guardrails": self.survived_guardrails,
            "on_pareto_front": self.on_pareto_front,
        }


def risk_adjusted(
    config: Config, outcome: DispatchOutcome, criteria: AdaptedCriteria
) -> dict[str, np.ndarray]:
    """Collapse each objective over scenarios, penalising bad tails not bad averages."""
    o = config.objectives
    q = o.risk["cvar_quantile"]
    lam = criteria.risk_lambda
    out: dict[str, np.ndarray] = {}

    for key in OBJECTIVES:
        arr = outcome.objectives[key]
        mean = arr.mean(axis=1)
        if key == "net_cost":
            # Cost score = mean + lambda x (CVaR90 - mean).
            out[key] = mean + lam * (cvar(arr, q) - mean)
        elif key == "reliability_risk":
            # Reliability adds the CVaR90 of unserved energy, so a plan that is fine
            # on average but sheds in the worst 10% of scenarios loses.
            u = outcome.unserved_mwh
            tail = cvar(u, o.reliability["cvar_quantile"]) - u.mean(axis=1)
            out[key] = mean + o.reliability["unserved_weight"] * np.maximum(tail, 0.0)
        else:
            out[key] = mean
    return out


def pareto_front(objectives: dict[str, np.ndarray], candidates: list[int]) -> tuple[int, ...]:
    """Non-dominated set over all six objectives (all minimised)."""
    keys = list(objectives)
    mat = np.column_stack([objectives[k] for k in keys])
    front: list[int] = []
    for i in candidates:
        dominated = False
        for j in candidates:
            if i == j:
                continue
            if np.all(mat[j] <= mat[i] + 1e-12) and np.any(mat[j] < mat[i] - 1e-12):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return tuple(front)


def score_clusters(
    config: Config,
    outcome: DispatchOutcome,
    criteria: AdaptedCriteria,
    surviving: tuple[int, ...],
    *,
    previous_cluster_index: int | None = None,
) -> tuple[list[ClusterScore], int, bool]:
    """Returns (all scores sorted best-first, chosen index, held_by_hysteresis)."""
    o = config.objectives
    adj = risk_adjusted(config, outcome, criteria)
    alive = list(surviving) or list(range(outcome.objectives["net_cost"].shape[0]))

    # Min-max normalise across the SURVIVING clusters only, with a floor on the range
    # so a trivial spread is not amplified into a decisive difference.
    normalized: dict[str, np.ndarray] = {}
    for key, arr in adj.items():
        lo = float(arr[alive].min())
        hi = float(arr[alive].max())
        span = max(hi - lo, float(o.range_floor[key]))
        normalized[key] = (arr - lo) / span

    weights = criteria.final_weights
    total = sum(weights[k] * normalized[k] for k in OBJECTIVES)

    front = set(pareto_front(adj, alive))
    cost = outcome.objectives["net_cost"]
    p10 = np.percentile(cost, 10, axis=1)
    p50 = np.percentile(cost, 50, axis=1)
    p90 = np.percentile(cost, 90, axis=1)
    unserved = outcome.unserved_mwh.mean(axis=1)

    scores = [
        ClusterScore(
            index=i,
            cluster=outcome.clusters[i],
            score=float(total[i]),
            objectives={k: float(adj[k][i]) for k in OBJECTIVES},
            normalized={k: float(normalized[k][i]) for k in OBJECTIVES},
            contributions={k: float(weights[k] * normalized[k][i]) for k in OBJECTIVES},
            cost_p10=float(p10[i]), cost_p50=float(p50[i]), cost_p90=float(p90[i]),
            unserved_mwh=float(unserved[i]),
            survived_guardrails=i in set(alive),
            on_pareto_front=i in front,
        )
        for i in range(len(outcome.clusters))
    ]

    alive_sorted = sorted((s for s in scores if s.survived_guardrails),
                          key=lambda s: (s.score, s.index))
    chosen = alive_sorted[0].index
    held = False

    # Step 4 -- stability tie-break: keep the previous cluster when it is within the
    # margin, so set-points do not flip-flop between adjacent intervals.
    if previous_cluster_index is not None:
        prev = next((s for s in alive_sorted if s.index == previous_cluster_index), None)
        if prev is not None and prev.index != chosen:
            if prev.score <= alive_sorted[0].score + o.hysteresis_margin:
                chosen = prev.index
                held = True

    ordered = sorted(scores, key=lambda s: (not s.survived_guardrails, s.score, s.index))
    return ordered, chosen, held
