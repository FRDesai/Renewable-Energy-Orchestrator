"""Verify every number in LLM prose against the decision trace.

PRD Responsible AI: "Every number in an explanation is checked against the decision
trace before display; a mismatch falls back to the template explanation."

Figures are bucketed by unit and checked only against trace figures of the same kind.
A single undifferentiated pool is far too weak: with hundreds of legitimate values
spanning dollars, megawatts and percentages, a fabricated "777.7 MW" matched an
unrelated cost figure of ~780 and passed.

Statistical labels are not figures. "P90" and "CVaR90" name a distribution band, so the
digits in them are checked against nothing and every explanation citing the scenario fan
was discarded -- including the template, which writes "P90 $..." itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Ordered so the longer unit wins: MWh before MW, Hz before h.
NUMBER = re.compile(
    r"(?P<dollar>-?\$)?(?P<num>-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>MWh|MW|m/s|Hz|%|t\b|h\b)?",
    re.IGNORECASE)
_STRIP = re.compile(r"[,$\s]")
# A digit-run touching a letter or underscore is part of a word, not a figure.
_IDENT = re.compile(r"[A-Za-z_]")

MONEY, POWER, ENERGY, FREQ, PERCENT, TONNES, HOURS, SPEED, BARE = (
    "money", "power", "energy", "frequency", "percent", "tonnes", "hours", "speed", "bare")

UNIT_OF = {
    "mwh": ENERGY, "mw": POWER, "hz": FREQ, "%": PERCENT,
    "t": TONNES, "h": HOURS, "m/s": SPEED,
}


@dataclass
class Verification:
    ok: bool
    checked: int
    unverified: list[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "checked": self.checked,
                "unverified": list(self.unverified), "reason": self.reason}


class FigureIndex:
    """Trace figures, bucketed by unit."""

    def __init__(self) -> None:
        self.buckets: dict[str, set[float]] = {}

    def add(self, kind: str, value) -> None:
        if value is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        bucket = self.buckets.setdefault(kind, set())
        for places in (0, 1, 2, 3):
            bucket.add(round(v, places))
            bucket.add(round(abs(v), places))

    def add_many(self, kind: str, values) -> None:
        for v in values:
            self.add(kind, v)

    def allowed(self, kind: str) -> set[float]:
        if kind == BARE:
            out: set[float] = set()
            for b in self.buckets.values():
                out |= b
            return out
        # A bare trace value may legitimately be quoted with a unit, so unit buckets
        # also admit the structural counts.
        return self.buckets.get(kind, set()) | self.buckets.get(BARE, set())


def build_index(decision, maintenance=None) -> FigureIndex:
    idx = FigureIndex()
    r = decision.committed
    snap = decision.snapshot
    cfg = decision.criteria

    idx.add_many(POWER, [
        r.demand_mw, r.import_mw, r.export_mw, r.grid_mw, r.unserved_mw,
        r.renewable_used_mw, r.solar_available_mw, r.wind_available_mw,
        r.total_curtailed_mw, r.solar_curtailed_mw, r.wind_curtailed_mw,
        r.battery_charge_mw, r.battery_discharge_mw, r.import_limit_mw,
        r.export_limit_mw, r.dr_mw, r.shift_mw, r.reserve_available_mw,
        r.reserve_shortfall_mw, r.forced_injection_mw,
        r.curtailed_by_corridor_mw, r.curtailed_by_export_mw,
        r.curtailed_at_negative_price_mw,
        snap.total_demand_mw, snap.net_load_mw, snap.available_renewable_mw,
        snap.total_solar_mw, snap.total_wind_mw, snap.municipal_mw,
        snap.import_limit_mw, snap.export_limit_mw,
    ])
    idx.add_many(POWER, r.battery_mw.values())
    idx.add_many(POWER, r.corridor_flow_mw.values())
    idx.add_many(POWER, r.corridor_limit_mw.values())
    idx.add_many(POWER, r.solar_curtailed_by_farm.values())
    idx.add_many(POWER, r.wind_curtailed_by_farm.values())
    idx.add_many(POWER, snap.solar_mw.values())
    idx.add_many(POWER, snap.wind_mw.values())
    idx.add_many(POWER, snap.consumer_mw.values())
    idx.add_many(POWER, snap.corridor_limit_mw.values())
    p = decision.snapshot
    idx.add_many(SPEED, snap.wind_speed_ms.values())
    idx.add_many(PERCENT, r.battery_soc_pct.values())
    idx.add_many(FREQ, [r.frequency_hz, snap.frequency_hz])

    s = decision.chosen_score
    idx.add_many(MONEY, [s.cost_p10, s.cost_p50, s.cost_p90,
                         r.price_per_mwh, snap.price_per_mwh,
                         snap.carbon_price_per_tonne])
    idx.add_many(MONEY, [s.objectives.get("net_cost"), s.objectives.get("battery_wear")])
    idx.add_many(ENERGY, [s.unserved_mwh, s.objectives.get("curtailment"),
                          s.objectives.get("reliability_risk")])
    idx.add_many(TONNES, [s.objectives.get("carbon")])
    for other in decision.scores[:10]:
        idx.add_many(MONEY, [other.cost_p10, other.cost_p90,
                             other.objectives.get("net_cost"),
                             other.objectives.get("battery_wear")])
        idx.add_many(ENERGY, [other.unserved_mwh, other.objectives.get("curtailment"),
                              other.objectives.get("reliability_risk")])
        idx.add_many(TONNES, [other.objectives.get("carbon")])
        idx.add(BARE, other.score)
    deltas = decision.objective_deltas()
    idx.add_many(MONEY, [deltas.get("net_cost"), deltas.get("battery_wear")])
    idx.add_many(ENERGY, [deltas.get("curtailment"), deltas.get("reliability_risk")])
    idx.add_many(TONNES, [deltas.get("carbon")])
    idx.add(BARE, deltas.get("grid_stability"))

    # Weights, multipliers and risk aversion read as bare decimals or percentages.
    for table in (cfg.base_weights, cfg.final_weights, cfg.multiplier_applied):
        for v in table.values():
            idx.add(BARE, v)
            idx.add(PERCENT, v * 100.0)
    idx.add_many(BARE, [cfg.risk_lambda, cfg.base_risk_lambda, s.score])

    # Counts and structural figures.
    idx.add_many(BARE, [decision.clusters_evaluated, decision.scenarios_evaluated,
                        decision.steps_evaluated, len(decision.surviving),
                        len(decision.pareto_front), decision.interval,
                        decision.guardrails_removed])
    idx.add_many(BARE, [g.removed_count for g in decision.guardrails])

    # Configured thresholds the trace quotes verbatim in its triggers, plus the storm
    # lead time, which the classifier derives rather than reads off the record.
    for regime in cfg.regimes:
        for value in _numbers_in(regime.trigger):
            idx.add(BARE, value)
            idx.add(POWER, value)
            idx.add(MONEY, value)
            idx.add(PERCENT, value)
            idx.add(HOURS, value)
            idx.add(FREQ, value)
    for g in decision.guardrails:
        for value in _numbers_in(g.note) | _numbers_in(g.rule):
            idx.add(BARE, value)
            idx.add(ENERGY, value)
            idx.add(PERCENT, value)
            idx.add(HOURS, value)
    # Maintenance is a separate planner on the same forecast, so its valuations and
    # configured bars are absent from `decision` -- yet plan_maintenance hands the model
    # window reasons quoting them ("...above the $2,500 bar"), which it then cites.
    for v in (maintenance.valuations if maintenance is not None else ()):
        idx.add_many(MONEY, [v.current_loss_usd, v.best_loss_usd, v.saving_usd])
        idx.add_many(BARE, [v.current_start, v.best_start])
        for value in _numbers_in(v.reason):
            idx.add(BARE, value)
            idx.add(MONEY, value)
    for note in (maintenance.notes if maintenance is not None else ()):
        for value in _numbers_in(note):
            idx.add(BARE, value)
            idx.add(MONEY, value)

    # Structural small integers: interval indices, hours, step and scenario counts.
    for i in range(0, 25):
        idx.add(BARE, float(i))
    return idx


def _numbers_in(text: str) -> set[float]:
    out: set[float] = set()
    for m in NUMBER.finditer(text or ""):
        v = _norm(m.group("num"))
        if v is not None:
            out.add(v)
            out.add(abs(v))
    return out


def _norm(raw: str) -> float | None:
    cleaned = _STRIP.sub("", raw or "")
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def trace_figures(decision) -> set[float]:
    """Flat view of every figure the decision legitimately contains."""
    return build_index(decision).allowed(BARE)


def verify(decision, text: str, *, tolerance: float = 0.02,
           maintenance=None) -> Verification:
    """A figure passes if it is within `tolerance` of a trace figure of the same unit.

    The tolerance is relative to the *trace* value, never to the claimed one: scaling
    it by the claim would let a large fabrication buy itself a large tolerance.
    """
    if not text.strip():
        return Verification(False, 0, reason="empty explanation")

    index = build_index(decision, maintenance)
    unverified: list[str] = []
    checked = 0
    for match in NUMBER.finditer(text):
        value = _norm(match.group("num"))
        if value is None:
            continue
        unit = (match.group("unit") or "").lower().strip()
        if match.group("dollar"):
            kind = MONEY
        elif unit:
            kind = UNIT_OF.get(unit, BARE)
        else:
            # A unitless digit-run glued to letters is a label, not a quantity claim:
            # "P90", "CVaR90", "90th", "BESS2". Reached only when no unit matched, so a
            # claim that carries one is still checked whatever it is glued to and this
            # cannot be used to smuggle a figure past the check.
            # Bound the digits themselves: the regex consumes trailing \s* even when
            # no unit follows, so match.end() would sit on the next word and skip the
            # bare figure in "80 clusters".
            start, end = match.span("num")
            if (start and _IDENT.match(text[start - 1])) \
                    or (end < len(text) and _IDENT.match(text[end])):
                continue
            kind = BARE
        checked += 1
        allowed = index.allowed(kind)
        if any(abs(value - a) <= max(tolerance * abs(a), 0.05) for a in allowed):
            continue
        unverified.append(match.group(0).strip())

    ok = not unverified
    return Verification(
        ok, checked, unverified,
        reason="" if ok else
        f"{len(unverified)} figure(s) not found in the decision trace: "
        f"{', '.join(unverified[:5])}")
