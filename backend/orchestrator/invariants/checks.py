"""The twelve invariants, as pure functions over a single interval's StepRecord.

PRD "Reliability test suite". These run on every build and nightly over randomised
episodes. Each returns (passed, detail) so a failure is reportable with evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..config_loader import Config
from ..decide.dispatch import StepRecord
from ..domain.state import StateSnapshot

TOL = 1e-6


@dataclass
class CheckOutcome:
    key: str
    label: str
    pass_condition: str
    passed: int = 0
    failed: int = 0
    applicable: int = 0
    failures: list[str] = field(default_factory=list)

    def record(self, ok: bool, detail: str = "") -> None:
        self.applicable += 1
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            if len(self.failures) < 5:
                self.failures.append(detail)

    @property
    def rate(self) -> float:
        return self.passed / self.applicable if self.applicable else 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key, "label": self.label,
            "pass_condition": self.pass_condition,
            "passed": self.passed, "failed": self.failed,
            "applicable": self.applicable,
            "rate": round(self.rate, 4),
            "failures": list(self.failures),
        }


@dataclass
class IntervalContext:
    """Everything an invariant needs about one committed interval."""

    config: Config
    snapshot: StateSnapshot
    record: StepRecord
    latency_ms: float
    storm_on_site: bool = False
    fleet_soc_pct: float = 0.0          # over AVAILABLE capacity
    available_capacity_fraction: float = 1.0
    chosen_index: int = -1


# ---- the twelve ------------------------------------------------------------
def energy_balance(c: IntervalContext) -> tuple[bool, str]:
    r = c.record
    served = r.demand_mw - r.dr_mw - r.shift_mw - r.unserved_mw
    lhs = r.renewable_used_mw + r.import_mw + r.battery_discharge_mw
    rhs = served + r.battery_charge_mw + r.export_mw
    ok = abs(lhs - rhs) <= 1e-5
    return ok, f"supply {lhs:.6f} MW vs demand {rhs:.6f} MW (residual {lhs - rhs:.2e})"


def soc_limits(c: IntervalContext) -> tuple[bool, str]:
    lim = c.config.portfolio.battery_limits
    bad = {b: v for b, v in c.record.battery_soc_pct.items()
           if v < lim.soc_min_pct - 1e-4 or v > lim.soc_max_pct + 1e-4}
    return not bad, f"outside {lim.soc_min_pct}-{lim.soc_max_pct}%: {bad}"


def power_ratings(c: IntervalContext) -> tuple[bool, str]:
    p = c.config.portfolio
    for b in p.batteries:
        mw = c.record.battery_mw.get(b.id, 0.0)
        if abs(mw) > b.power_mw + TOL:
            return False, f"{b.name} at {mw:.3f} MW against a {b.power_mw} MW rating"
        if not c.snapshot.battery_available.get(b.id, True) and abs(mw) > TOL:
            return False, f"tripped {b.name} dispatched {mw:.3f} MW"
    return True, ""


def transmission_limits(c: IntervalContext) -> tuple[bool, str]:
    r = c.record
    if r.import_mw > r.import_limit_mw + 1e-5:
        return False, f"import {r.import_mw:.3f} > {r.import_limit_mw} MW"
    if r.export_mw > r.export_limit_mw + 1e-5:
        return False, f"export {r.export_mw:.3f} > {r.export_limit_mw} MW"
    for corridor, flow in r.corridor_flow_mw.items():
        cap = r.corridor_limit_mw.get(corridor, float("inf"))
        if flow > cap + 1e-5:
            return False, f"{corridor} corridor {flow:.3f} > {cap} MW"
    return True, ""


def no_wash_trading(c: IntervalContext) -> tuple[bool, str]:
    r = c.record
    ok = not (r.import_mw > TOL and r.export_mw > TOL)
    return ok, f"import {r.import_mw:.3f} MW and export {r.export_mw:.3f} MW together"


def justified_curtailment(c: IntervalContext) -> tuple[bool, str | None]:
    r = c.record
    if r.total_curtailed_mw <= TOL:
        return None, ""                      # not applicable
    corridor_full = any(
        flow >= r.corridor_limit_mw.get(k, float("inf")) - 1e-5
        for k, flow in r.corridor_flow_mw.items())
    export_full = r.export_mw >= r.export_limit_mw - 1e-5
    negative = r.price_per_mwh < 0.0
    if not (corridor_full or export_full or negative):
        return False, (f"spilled {r.total_curtailed_mw:.3f} MW with corridor not full, "
                       f"export {r.export_mw:.1f}/{r.export_limit_mw} MW, "
                       f"price ${r.price_per_mwh:.2f}")
    # "with batteries absorbing what they can" applies to spill the batteries could
    # physically take. Corridor-constrained energy is stranded behind the corridor and
    # cannot reach storage on the far side of it, so only export-limit and
    # negative-price spill is checked here (see DECISIONS.md D18).
    absorbable = r.curtailed_by_export_mw + r.curtailed_at_negative_price_mw
    if absorbable <= TOL:
        return True, ""
    lim = c.config.portfolio.battery_limits
    for b in c.config.portfolio.batteries:
        if not c.snapshot.battery_available.get(b.id, True):
            continue
        charging = max(0.0, -r.battery_mw.get(b.id, 0.0))
        full = r.battery_soc_pct.get(b.id, 0.0) >= lim.soc_max_pct - 1e-3
        at_rating = charging >= b.power_mw - 1e-3
        discharging_for_reflex = r.forced_injection_mw > TOL
        preventing_shedding = r.import_mw >= r.import_limit_mw - 1e-5
        if not (full or at_rating or discharging_for_reflex or preventing_shedding):
            return False, (f"spilled {absorbable:.3f} absorbable MW while {b.name} "
                           f"absorbed only {charging:.1f}/{b.power_mw} MW at "
                           f"{r.battery_soc_pct.get(b.id, 0):.1f}% SoC")
    return True, ""


def no_discharge_while_curtailing(c: IntervalContext) -> tuple[bool, str | None]:
    """The PRD's open curtailment defect, as its own invariant."""
    r = c.record
    limit_driven = r.curtailed_by_corridor_mw + r.curtailed_by_export_mw
    if limit_driven <= TOL:
        return None, ""
    allowance = max(r.forced_injection_mw, 0.0)
    if r.import_mw >= r.import_limit_mw - 1e-5:
        return None, ""                      # discharging to avoid shedding
    ok = r.battery_discharge_mw <= allowance + 1e-5
    return ok, (f"discharged {r.battery_discharge_mw:.3f} MW while spilling "
                f"{limit_driven:.3f} MW for a transmission limit "
                f"(reflex allowance {allowance:.1f} MW)")


def unavoidable_shedding_only(c: IntervalContext) -> tuple[bool, str | None]:
    r = c.record
    if r.unserved_mw <= TOL:
        return None, ""
    if r.import_mw < r.import_limit_mw - 1e-5:
        return False, (f"shed {r.unserved_mw:.3f} MW with imports at "
                       f"{r.import_mw:.1f}/{r.import_limit_mw} MW")
    lim = c.config.portfolio.battery_limits
    for b in c.config.portfolio.batteries:
        if not c.snapshot.battery_available.get(b.id, True):
            continue
        mw = r.battery_mw.get(b.id, 0.0)
        at_rating = mw >= b.power_mw - 1e-3
        at_floor = r.battery_soc_pct.get(b.id, 0.0) <= lim.soc_min_pct + 1e-3
        if not (at_rating or at_floor):
            return False, (f"shed {r.unserved_mw:.3f} MW while {b.name} delivered only "
                           f"{mw:.1f}/{b.power_mw} MW at "
                           f"{r.battery_soc_pct.get(b.id, 0):.1f}% SoC")
    return True, ""


def frequency_response(c: IntervalContext) -> tuple[bool, str | None]:
    reflex = c.config.portfolio.grid.frequency_reflex_hz
    if c.record.frequency_hz >= reflex:
        return None, ""
    if not any(c.snapshot.battery_available.values()):
        return None, ""                      # nothing available to inject
    ok = c.record.battery_discharge_mw > TOL
    return ok, (f"{c.record.frequency_hz:.3f} Hz dip with "
                f"{c.record.battery_discharge_mw:.3f} MW injected")


def storm_readiness(c: IntervalContext) -> tuple[bool, str | None]:
    """Fleet charge level when a forecast storm arrives.

    Measured over *available* capacity. A tripped battery cannot be charged, so
    holding the agent to total nameplate while 60% of the fleet is offline would make
    the bar physically unreachable rather than merely strict -- which is exactly what
    happened under the contingency-cascade preset.
    """
    if not c.storm_on_site:
        return None, ""
    if c.available_capacity_fraction <= 1e-9:
        return None, ""                  # whole fleet offline; nothing to charge
    bar = c.config.portfolio.battery_limits.storm_readiness_pct
    ok = c.fleet_soc_pct >= bar - 1e-3
    note = f"available fleet at {c.fleet_soc_pct:.1f}% when the storm hit, bar {bar}%"
    if c.available_capacity_fraction < 0.999:
        note += (f" ({c.available_capacity_fraction:.0%} of storage capacity online)")
    return ok, note


def latency(c: IntervalContext) -> tuple[bool, str]:
    ok = c.latency_ms < 1000.0
    return ok, f"decision took {c.latency_ms:.1f} ms"


def critical_load_protected(c: IntervalContext) -> tuple[bool, str]:
    """PRD: "Critical loads (data center, municipal) are never curtailed by the agent"."""
    p = c.config.portfolio
    flex = p.total_dr_mw + p.total_shiftable_mw
    r = c.record
    used = r.dr_mw + max(0.0, r.shift_mw)
    ok = used <= flex + TOL
    return ok, (f"reduced {used:.2f} MW against {flex:.0f} MW of non-critical "
                f"flexibility, so critical load must have been touched")


CHECKS: dict[str, tuple[str, str, Callable]] = {
    "energy_balance": (
        "Energy balance",
        "Generation - curtailment + imports + discharge = load + charge + exports (+ unserved)",
        energy_balance),
    "soc_limits": ("SoC limits", "8% <= SoC <= 95%", soc_limits),
    "power_ratings": ("Power ratings", "No unit above its MW rating; tripped units at 0 MW",
                      power_ratings),
    "transmission_limits": ("Transmission limits",
                            "Imports, exports and corridor flow within current limits",
                            transmission_limits),
    "no_wash_trading": ("No wash trading", "Never import and export in the same interval",
                        no_wash_trading),
    "justified_curtailment": (
        "Justified curtailment",
        "Spill only if the corridor is full, the export line is full or the price is "
        "negative, with batteries absorbing what they can",
        justified_curtailment),
    "no_discharge_while_curtailing": (
        "No discharge while curtailing",
        "No battery discharge in an interval where renewables are curtailed for a "
        "transmission limit (except the frequency reflex or to avoid shedding)",
        no_discharge_while_curtailing),
    "unavoidable_shedding_only": (
        "Unavoidable shedding only",
        "Unserved load only with imports at the limit and batteries at full discharge",
        unavoidable_shedding_only),
    "frequency_response": ("Frequency response",
                           "Batteries inject during every dip below 49.9 Hz",
                           frequency_response),
    "storm_readiness": ("Storm readiness",
                        "Fleet >= 50% charged when a forecast storm arrives",
                        storm_readiness),
    "latency": ("Latency", "Every decision under 1 s", latency),
    "critical_load_protected": (
        "Critical load protected",
        "Critical consumers are never curtailed by the agent",
        critical_load_protected),
}


class InvariantSuite:
    """Accumulates results across intervals and episodes."""

    def __init__(self) -> None:
        self.outcomes = {
            key: CheckOutcome(key, label, condition)
            for key, (label, condition, _) in CHECKS.items()
        }
        self.determinism_pass: bool | None = None

    def check(self, context: IntervalContext) -> None:
        for key, (_, _, fn) in CHECKS.items():
            ok, detail = fn(context)
            if ok is None:
                continue                      # not applicable this interval
            self.outcomes[key].record(bool(ok),
                                      f"interval {context.snapshot.interval}: {detail}")

    @property
    def all_passed(self) -> bool:
        return all(o.failed == 0 for o in self.outcomes.values())

    def report(self) -> list[dict[str, object]]:
        rows = [o.as_dict() for o in self.outcomes.values()]
        rows.append({
            "key": "determinism", "label": "Determinism",
            "pass_condition": "Same inputs and seed produce the same cluster",
            "passed": 1 if self.determinism_pass else 0,
            "failed": 0 if self.determinism_pass in (True, None) else 1,
            "applicable": 0 if self.determinism_pass is None else 1,
            "rate": 1.0 if self.determinism_pass else 0.0,
            "failures": [] if self.determinism_pass is not False
                        else ["same seed produced a different cluster"],
        })
        return rows
