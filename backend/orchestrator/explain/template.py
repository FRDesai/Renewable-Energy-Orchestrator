"""Template explanations: the LLM-free path, and the fallback when LLM output fails
verification. Every figure here is read straight off the decision trace.
"""
from __future__ import annotations

from ..config_loader import Config
from ..decide.dispatch import OBJECTIVES
from ..decide.optimizer import Decision

UNITS = {
    "net_cost": "$", "reliability_risk": " MWh-eq", "carbon": " t CO2",
    "curtailment": " MWh", "grid_stability": " index", "battery_wear": "$",
}
NICE = {
    "net_cost": "net cost", "reliability_risk": "reliability risk", "carbon": "carbon",
    "curtailment": "curtailment", "grid_stability": "grid stability",
    "battery_wear": "battery wear",
}


def _fmt(key: str, value: float) -> str:
    unit = UNITS[key]
    if unit == "$":
        return f"${abs(value):,.0f}"
    return f"{abs(value):,.1f}{unit}"


def situation(decision: Decision) -> str:
    regimes = decision.criteria.regimes
    if not regimes:
        return "Normal conditions: no regime triggered."
    return "; ".join(f"{r.label} ({r.trigger})" for r in regimes)


def criteria_text(decision: Decision) -> str:
    c = decision.criteria
    shifted = c.shifted()
    if not shifted:
        return (f"Operator base weights unchanged; risk aversion "
                f"lambda {c.risk_lambda:.2f}.")
    parts = [f"{NICE[k]} {b:.2f}->{f:.2f}" for k, (b, f) in sorted(shifted.items())]
    return (f"Weights shifted from the operator's base: {', '.join(parts)}. "
            f"Risk aversion lambda {c.base_risk_lambda:.2f}->{c.risk_lambda:.2f}.")


def options_text(decision: Decision) -> str:
    removed = [g for g in decision.guardrails if g.applied and g.removed_count]
    bits = [f"{decision.clusters_evaluated} clusters scored across "
            f"{decision.scenarios_evaluated} scenarios x {decision.steps_evaluated} steps"]
    if removed:
        bits.append("guardrails removed " + ", ".join(
            f"{g.removed_count} ({g.key.replace('_', ' ')})" for g in removed))
    else:
        bits.append("no guardrail removed any cluster")
    bits.append(f"{len(decision.surviving)} survived")
    return "; ".join(bits) + "."


def tradeoff_text(decision: Decision) -> str:
    """PRD: "chosen plan costs $1.2k more but removes 3.4 MWh of shedding risk;
    reliability weight 0.35 decides"."""
    other = decision.material_alternative() or decision.runner_up
    if other is None:
        return "Only one cluster survived the guardrails, so there was no trade-off."

    deltas = decision.objective_deltas(other)
    worse = sorted(((k, v) for k, v in deltas.items() if v > 0),
                   key=lambda kv: -abs(kv[1]))
    better = sorted(((k, v) for k, v in deltas.items() if v < 0),
                    key=lambda kv: -abs(kv[1]))
    deciding = decision.deciding_objective(other)
    weight = decision.criteria.final_weights.get(deciding or "", 0.0)

    label = other.cluster.short_label
    if not other.survived_guardrails:
        label += " (removed by a guardrail)"

    clauses: list[str] = []
    if worse:
        k, v = worse[0]
        clauses.append(f"costs {_fmt(k, v)} more {NICE[k]}")
    if better:
        k, v = better[0]
        clauses.append(f"removes {_fmt(k, v)} of {NICE[k]}")
    if not clauses:
        clauses.append("is equivalent on every objective")

    sentence = f"Chosen over {label}: the plan {' but '.join(clauses)}"
    if deciding:
        sentence += f"; {NICE[deciding]} weight {weight:.2f} decides"
    return sentence + "."


def explain(config: Config, decision: Decision) -> str:
    r = decision.committed
    lines = [
        f"Situation: {situation(decision)}",
        f"Criteria: {criteria_text(decision)}",
        f"Options: {options_text(decision)}",
        f"Committed: {decision.chosen.label}.",
        f"This interval: {r.renewable_used_mw:.0f} MW renewable used, "
        f"{r.battery_discharge_mw:.0f} MW discharged, {r.battery_charge_mw:.0f} MW charged, "
        f"{r.import_mw:.0f} MW imported, {r.export_mw:.0f} MW exported, "
        f"{r.total_curtailed_mw:.0f} MW curtailed, {r.unserved_mw:.1f} MW unserved.",
        f"Cost outlook: P10 ${decision.chosen_score.cost_p10:,.0f} to "
        f"P90 ${decision.chosen_score.cost_p90:,.0f} over the look-ahead.",
        f"Trade-off: {tradeoff_text(decision)}",
    ]
    if decision.held_by_hysteresis:
        lines.append(
            f"Held the previous interval's cluster: it scored within "
            f"{config.objectives.hysteresis_margin} of the winner, so set-points are "
            f"kept stable rather than flipped.")
    if decision.snapshot.quality.degraded:
        q = decision.snapshot.quality
        lines.append(
            f"Data quality: {', '.join(q.substituted_fields)} substituted from the "
            f"forecast; {', '.join(q.stale_fields)} stale.")
    return "\n".join(lines)
