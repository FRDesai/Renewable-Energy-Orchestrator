"""Situation classifier: which regimes are active, and the measured trigger for each.

PRD F2.1: "Each regime shown with the measured trigger, e.g. 'Spot $310 vs typical
$52'." Rules over state and forecasts in v1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config_loader import Config
from ..domain.events import EventType
from ..domain.state import StateSnapshot
from .dispatch import ScenarioSet


@dataclass(frozen=True)
class ActiveRegime:
    key: str
    label: str
    rule: str                 # the configured rule, in words
    trigger: str              # the measured evidence, with units
    multipliers: dict[str, float]
    risk_aversion: float

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key, "label": self.label, "rule": self.rule,
            "trigger": self.trigger, "multipliers": self.multipliers,
            "risk_aversion": self.risk_aversion,
        }


def classify(
    config: Config, snapshot: StateSnapshot, scenarios: ScenarioSet | None = None
) -> tuple[ActiveRegime, ...]:
    o = config.objectives
    p = config.portfolio
    th = o.thresholds
    found: list[ActiveRegime] = []

    def add(key: str, trigger: str) -> None:
        r = o.regimes[key]
        found.append(ActiveRegime(key, r.label, r.trigger, trigger,
                                  dict(r.multipliers), r.risk_aversion))

    # 1. Grid stress -- storm alert or storm on site.
    storm = snapshot.storm_alert
    if storm is not None:
        if storm.on_site_at(snapshot.interval):
            add("grid_stress", "Storm on site now")
        else:
            lead = (storm.intervals_to_arrival(snapshot.interval) or 0) * p.interval_hours
            add("grid_stress",
                f"Storm alert, forecast arrival in {lead:.1f} h "
                f"(severity {storm.severity.value})")

    # 2. Frequency event.
    freq_threshold = float(o.guardrails["frequency_dip_hz"])
    if snapshot.frequency_hz < freq_threshold:
        add("frequency_event",
            f"{snapshot.frequency_hz:.3f} Hz against a {freq_threshold:.2f} Hz threshold")

    # 3. Asset contingency.
    offline = snapshot.batteries_offline
    if offline:
        names = ", ".join(p.battery(b).name for b in offline)
        lost = sum(p.battery(b).power_mw for b in offline)
        add("asset_contingency", f"{names} offline ({lost:.0f} MW unavailable)")

    # 4. Scarcity ahead -- deficit within 3 h above 75% of the import limit.
    if scenarios is not None:
        steps = max(1, int(th["scarcity_horizon_hours"] / p.interval_hours))
        steps = min(steps, scenarios.n_steps)
        renewable = scenarios.solar_mw[:, :steps] + sum(
            v[:, :steps] for v in scenarios.wind_mw.values())
        deficit = scenarios.demand_mw[:, :steps] - renewable
        worst = float(np.max(np.median(deficit, axis=0)))
        bar = th["scarcity_import_fraction"] * snapshot.import_limit_mw
        if worst > bar:
            add("scarcity_ahead",
                f"Deficit {worst:.0f} MW within "
                f"{th['scarcity_horizon_hours']:.0f} h against {bar:.0f} MW "
                f"({th['scarcity_import_fraction']:.0%} of the import limit)")

    # 5/6. Price spike and peak pricing.
    p70 = snapshot.price_percentile(th["price_spike_percentile"])
    spike_bar = max(th["price_spike_multiple"] * p70, th["price_spike_floor_per_mwh"])
    typical = config.market.price["typical_per_mwh"]
    if snapshot.price_per_mwh >= spike_bar:
        add("price_spike",
            f"Spot ${snapshot.price_per_mwh:.0f}/MWh vs typical ${typical:.0f}/MWh "
            f"(spike bar ${spike_bar:.0f})")
    peak_bar = snapshot.price_percentile(th["peak_price_percentile"])
    if snapshot.price_per_mwh >= peak_bar and snapshot.price_per_mwh > 0:
        add("peak_pricing",
            f"Spot ${snapshot.price_per_mwh:.0f}/MWh at or above the day's "
            f"{th['peak_price_percentile']:.0%} percentile (${peak_bar:.0f}/MWh)")

    # 7. Transmission constrained -- export derated or a corridor effectively full.
    reasons: list[str] = []
    if snapshot.export_limit_mw < p.grid.export_limit_mw - 1e-6:
        reasons.append(
            f"export derated to {snapshot.export_limit_mw:.0f} MW "
            f"from {p.grid.export_limit_mw:.0f} MW")
    for corridor in p.corridors:
        cap = snapshot.corridor_limit_mw.get(corridor, float("inf"))
        if not np.isfinite(cap):
            continue
        flow = sum(snapshot.wind_mw[f.id] for f in p.wind_farms if f.corridor == corridor)
        if flow >= th["corridor_full_fraction"] * cap:
            reasons.append(f"{corridor} corridor at {flow:.0f}/{cap:.0f} MW")
    if reasons:
        add("transmission_constrained", "; ".join(reasons))

    # 8. Renewable surplus.
    surplus = -snapshot.net_load_mw
    if surplus > th["renewable_surplus_mw"]:
        add("renewable_surplus",
            f"Renewables exceed load by {surplus:.0f} MW "
            f"(threshold {th['renewable_surplus_mw']:.0f} MW)")

    # 9. Negative prices.
    if snapshot.price_per_mwh < 0:
        add("negative_prices", f"Spot -${abs(snapshot.price_per_mwh):.2f}/MWh")

    # 10. DR program active.
    if snapshot.dr_call_active:
        pay = config.market.demand_response["iso_call_payment_per_mwh"]
        add("dr_program_active", f"ISO demand-response call active (pays ${pay:.0f}/MWh)")

    return tuple(found)
