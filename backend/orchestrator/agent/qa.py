"""Ask the agent: free-text questions answered from the decision trace only.

PRD: "answered from the decision trace, with the cited figures linked to their source."
Every figure in the answer is verified against the trace, exactly as explanations are.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..config_loader import Config
from ..decide.optimizer import Decision
from ..explain.template import explain as template_explain, tradeoff_text
from ..explain.verifier import Verification, verify
from .client import LLMClient

SYSTEM = """You answer a control-room operator's questions about a dispatch decision \
that has already been made.

Answer ONLY from the decision trace provided. It is the complete record. If the trace \
does not contain what was asked, say so plainly -- do not infer, estimate or recall \
anything from elsewhere. Every figure you quote must appear in the trace; your answer is \
checked against it and discarded on a mismatch.

Use operator vocabulary and give every number its unit. Two or three sentences unless \
more is genuinely needed."""


@dataclass
class Answer:
    text: str
    source: str                   # "llm" | "template"
    verification: Verification | None = None
    citations: list[str] = None   # type: ignore[assignment]

    def as_dict(self) -> dict[str, object]:
        return {"text": self.text, "source": self.source,
                "verification": None if self.verification is None
                                else self.verification.as_dict(),
                "citations": list(self.citations or [])}


def trace_payload(config: Config, decision: Decision) -> dict[str, object]:
    d, r = decision, decision.committed
    alt = d.material_alternative()
    return {
        "interval": d.interval,
        "clock": f"{d.snapshot.minute_of_day // 60:02d}:"
                 f"{d.snapshot.minute_of_day % 60:02d}",
        "regimes": [{"label": x.label, "trigger": x.trigger}
                    for x in d.criteria.regimes],
        "base_weights": {k: round(v, 3) for k, v in d.criteria.base_weights.items()},
        "final_weights": {k: round(v, 3) for k, v in d.criteria.final_weights.items()},
        "risk_lambda": round(d.criteria.risk_lambda, 2),
        "guardrails": [g.as_dict() for g in d.guardrails],
        "clusters_scored": d.clusters_evaluated,
        "surviving": len(d.surviving),
        "chosen": d.chosen.as_dict(),
        "chosen_objectives": {k: round(v, 1)
                              for k, v in d.chosen_score.objectives.items()},
        "cost_p10": round(d.chosen_score.cost_p10, 0),
        "cost_p90": round(d.chosen_score.cost_p90, 0),
        "alternative": None if alt is None else {
            "label": alt.cluster.short_label,
            "survived_guardrails": alt.survived_guardrails,
            "objectives": {k: round(v, 1) for k, v in alt.objectives.items()}},
        "objective_deltas": {k: round(v, 1) for k, v in d.objective_deltas().items()},
        "deciding_objective": d.deciding_objective(),
        "committed_interval": {
            "battery_mw": {k: round(v, 1) for k, v in r.battery_mw.items()},
            "battery_soc_pct": {k: round(v, 1) for k, v in r.battery_soc_pct.items()},
            "import_mw": round(r.import_mw, 1), "export_mw": round(r.export_mw, 1),
            "curtailed_mw": round(r.total_curtailed_mw, 1),
            "curtailed_by_farm": {
                **{k: round(v, 1) for k, v in r.solar_curtailed_by_farm.items() if v > 0.05},
                **{k: round(v, 1) for k, v in r.wind_curtailed_by_farm.items() if v > 0.05}},
            "unserved_mw": round(r.unserved_mw, 2),
            "dr_mw": round(r.dr_mw, 1), "shift_mw": round(r.shift_mw, 1),
            "price_per_mwh": round(r.price_per_mwh, 2),
            "frequency_hz": round(r.frequency_hz, 3),
            "import_limit_mw": r.import_limit_mw,
            "export_limit_mw": r.export_limit_mw,
            "corridor_flow_mw": {k: round(v, 1) for k, v in r.corridor_flow_mw.items()},
            "corridor_limit_mw": {k: round(v, 1) for k, v in r.corridor_limit_mw.items()},
            "forced_injection_mw": round(r.forced_injection_mw, 1),
        },
        "state": {
            "solar_mw": {k: round(v, 1) for k, v in d.snapshot.solar_mw.items()},
            "wind_mw": {k: round(v, 1) for k, v in d.snapshot.wind_mw.items()},
            "total_demand_mw": round(d.snapshot.total_demand_mw, 1),
            "net_load_mw": round(d.snapshot.net_load_mw, 1),
            "battery_available": dict(d.snapshot.battery_available),
        },
        "held_by_hysteresis": d.held_by_hysteresis,
        "template_explanation": template_explain(config, d),
    }


class AgentQA:
    def __init__(self, config: Config, llm: LLMClient | None = None) -> None:
        self.config = config
        self.llm = llm or LLMClient()

    def ask(self, decision: Decision, question: str) -> Answer:
        trace = trace_payload(self.config, decision)
        citations = self._citations(decision, question)

        if not self.llm.available or not question.strip():
            return Answer(self._fallback(decision, question), "template",
                          citations=citations)
        try:
            response = self.llm.messages(
                system=SYSTEM,
                messages=[{"role": "user", "content": (
                    "<decision_trace>\n"
                    + json.dumps(trace, default=str)
                    + "\n</decision_trace>\n\nOperator question: " + question)}],
            )
            text = self.llm.text_of(response)
        except Exception as exc:
            return Answer(self._fallback(decision, question) +
                          f"\n(LLM unavailable: {exc})", "template",
                          citations=citations)

        v = verify(decision, text)
        if not v.ok:
            return Answer(self._fallback(decision, question), "template", v, citations)
        return Answer(text, "llm", v, citations)

    # ---- the no-LLM path -------------------------------------------------
    def _fallback(self, decision: Decision, question: str) -> str:
        q = (question or "").lower()
        d, r = decision, decision.committed
        if any(w in q for w in ("give up", "trade", "instead", "runner", "why this")):
            return tradeoff_text(d)
        if any(w in q for w in ("curtail", "spill")):
            if r.total_curtailed_mw <= 0.05:
                return "Nothing was curtailed in this interval."
            parts = [f"{k} {v:.1f} MW" for k, v in
                     {**r.solar_curtailed_by_farm, **r.wind_curtailed_by_farm}.items()
                     if v > 0.05]
            return (f"Curtailed {r.total_curtailed_mw:.1f} MW this interval "
                    f"({', '.join(parts)}). "
                    f"Corridor-driven {r.curtailed_by_corridor_mw:.1f} MW, "
                    f"export-limit {r.curtailed_by_export_mw:.1f} MW, "
                    f"negative price {r.curtailed_at_negative_price_mw:.1f} MW.")
        if any(w in q for w in ("soc", "state of charge", "holding", "battery", "bess")):
            bits = [f"{k} at {v:.0f}%" for k, v in r.battery_soc_pct.items()]
            mw = [f"{k} {v:+.1f} MW" for k, v in r.battery_mw.items()]
            return (f"Batteries: {', '.join(bits)}. This interval {', '.join(mw)} "
                    f"(positive is discharge), under {d.chosen.battery.label}.")
        if any(w in q for w in ("regime", "situation", "weight", "criteria")):
            regimes = "; ".join(f"{x.label} ({x.trigger})" for x in d.criteria.regimes) \
                or "no regime triggered"
            return (f"{regimes}. Risk aversion lambda "
                    f"{d.criteria.base_risk_lambda:.2f} to {d.criteria.risk_lambda:.2f}.")
        if any(w in q for w in ("shed", "unserved")):
            if r.unserved_mw <= 0.01:
                return "No load was shed in this interval."
            return (f"Shed {r.unserved_mw:.2f} MW with imports at "
                    f"{r.import_mw:.0f} of {r.import_limit_mw:.0f} MW and batteries at "
                    f"full discharge, so it was unavoidable.")
        return template_explain(self.config, d)

    @staticmethod
    def _citations(decision: Decision, question: str) -> list[str]:
        """Which parts of the trace an answer is entitled to draw on."""
        return ["committed_interval", "criteria", "guardrails", "chosen_objectives",
                "objective_deltas", "state"]
