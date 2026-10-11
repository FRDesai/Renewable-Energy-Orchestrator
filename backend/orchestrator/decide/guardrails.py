"""Hard safety guardrails, applied lexicographically before any trade-off.

PRD Step 1. These are engine modules and are deliberately not reachable by the LLM:
the orchestrator can choose what to look at, never what is allowed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config_loader import Config
from ..domain.state import StateSnapshot
from .dispatch import DispatchOutcome, ScenarioSet


@dataclass(frozen=True)
class GuardrailResult:
    key: str
    rule: str
    applied: bool                 # False when the situation does not call for it
    removed: tuple[int, ...]
    note: str = ""

    @property
    def removed_count(self) -> int:
        return len(self.removed)

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key, "rule": self.rule, "applied": self.applied,
            "removed_count": self.removed_count, "note": self.note,
        }


def apply_guardrails(
    config: Config,
    snapshot: StateSnapshot,
    scenarios: ScenarioSet,
    outcome: DispatchOutcome,
) -> tuple[tuple[int, ...], tuple[GuardrailResult, ...]]:
    """Returns (surviving cluster indices, one result per guardrail).

    Lexicographic: each guardrail filters what the previous one left. If a guardrail
    would remove every remaining cluster it is reported but not enforced -- an empty
    option set is worse than a breached preference, and the decision trace says so.
    """
    g = config.objectives.guardrails
    p = config.portfolio
    n_clusters = outcome.objectives["net_cost"].shape[0]
    alive = list(range(n_clusters))
    results: list[GuardrailResult] = []

    def enforce(key: str, rule: str, offending: np.ndarray, note: str) -> None:
        nonlocal alive
        bad = [i for i in alive if bool(offending[i])]
        if bad and len(bad) == len(alive):
            results.append(GuardrailResult(
                key, rule, True, (),
                note=f"{note} -- would have removed all {len(alive)} remaining "
                     f"clusters, so not enforced; the trade-off below is unguarded"))
            return
        alive = [i for i in alive if i not in set(bad)]
        results.append(GuardrailResult(key, rule, True, tuple(bad), note))

    # --- 1. Avoidable unserved load ---------------------------------------
    tol = float(g["avoidable_unserved_mwh"])
    unserved = outcome.unserved_mwh.mean(axis=1)
    best = float(unserved.min())
    enforce(
        "avoidable_unserved",
        f"Discard clusters with more than {tol} MWh of unserved load above the best",
        unserved > best + tol,
        note=f"best cluster sheds {best:.3f} MWh; bar {best + tol:.3f} MWh",
    )

    # --- 2. Storm readiness ------------------------------------------------
    storm = snapshot.storm_alert
    lim = p.battery_limits
    fleet_mwh = float(sum(b.energy_mwh for b in p.batteries))
    required = fleet_mwh * float(g["storm_soc_pct"]) / 100.0
    if storm is None or storm.arrival_interval is None:
        results.append(GuardrailResult(
            "storm_readiness",
            f"Under a storm alert, discard clusters below "
            f"{g['storm_soc_pct']}% fleet SoC at forecast arrival",
            False, (), note="no storm alert active"))
    else:
        lead_intervals = storm.arrival_interval - snapshot.interval
        lead_hours = lead_intervals * p.interval_hours
        if lead_hours > lim.storm_max_lead_hours:
            results.append(GuardrailResult(
                "storm_readiness", "Storm readiness", False, (),
                note=f"arrival {lead_hours:.1f} h away, beyond the "
                     f"{lim.storm_max_lead_hours:.0f} h alert window"))
        else:
            horizon_steps = scenarios.n_steps
            if 0 <= lead_intervals < horizon_steps:
                # Arrival is inside the look-ahead: read the SoC at that step.
                reachable = outcome.soc_path_mwh[:, :, lead_intervals].mean(axis=1)
                bar = required
                how = f"projected fleet SoC at arrival (step {lead_intervals})"
            else:
                # Arrival is beyond the look-ahead -- the case the PRD's open defect
                # was about. Asking only whether the bar is still *reachable* is
                # toothless: 200 MW of charging makes almost anything reachable, so
                # every cluster passed and the fleet arrived at 32%. Instead demand
                # *progress*: the cluster's end-of-horizon SoC must be at least on the
                # straight-line pre-charge ramp from here to the bar at arrival
                # (see DECISIONS.md D10).
                start_mwh = fleet_mwh * snapshot.fleet_soc_pct(
                    {b.id: b.energy_mwh for b in p.batteries}) / 100.0
                frac = min(1.0, config.objectives.lookahead_hours / max(lead_hours, 1e-9))
                bar = start_mwh + (required - start_mwh) * frac
                reachable = outcome.end_soc_mwh.mean(axis=1)
                how = (f"fleet SoC at the end of the look-ahead against the "
                       f"pre-charge ramp ({frac:.0%} of the way from "
                       f"{start_mwh:.0f} to {required:.0f} MWh)")
            enforce(
                "storm_readiness",
                f"Under a storm alert, discard clusters not on track to be at "
                f"{g['storm_soc_pct']}% fleet SoC when the storm arrives",
                reachable < bar - 1e-9,
                note=f"{how}; bar {bar:.0f} MWh, arrival in {lead_hours:.1f} h "
                     f"(target {required:.0f} MWh = {g['storm_soc_pct']}% of "
                     f"{fleet_mwh:.0f} MWh)",
            )

    # --- 3. Frequency response ---------------------------------------------
    threshold = float(g["frequency_dip_hz"])
    dipping = scenarios.frequency_hz < threshold          # (S, T)
    if not dipping.any():
        results.append(GuardrailResult(
            "frequency_response",
            f"During a dip below {threshold} Hz, discard clusters that do not inject",
            False, (), note="no dip in the horizon"))
    else:
        if not any(snapshot.battery_available.values()):
            results.append(GuardrailResult(
                "frequency_response", "Frequency response", False, (),
                note="every battery is offline; no cluster can inject"))
        else:
            mask = np.broadcast_to(dipping, outcome.injected_mw.shape[1:])
            injected = np.where(mask[None, :, :], outcome.injected_mw, np.inf)
            worst = injected.min(axis=(1, 2))
            enforce(
                "frequency_response",
                f"During a dip below {threshold} Hz, discard clusters that do not inject",
                worst <= 1e-9,
                note="the reflex is applied inside every cluster, so this guardrail is "
                     "defence in depth and normally removes nothing",
            )

    return tuple(alive), tuple(results)
