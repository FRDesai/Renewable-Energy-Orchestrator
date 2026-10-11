"""The optimizer: stages 2-7 of the decision loop, producing one Decision.

Everything the LLM orchestrator executes comes from here. The orchestrator chooses
*when* to call this; it never chooses the numbers or the guardrails.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config_loader import Config
from ..domain.actions import CLUSTERS, ActionCluster, CommittedActions, SetPoint
from ..domain.state import StateSnapshot
from ..forecast.scenarios import Envelope, build_scenarios
from .classifier import ActiveRegime, classify
from .dispatch import OBJECTIVES, DispatchEngine, DispatchOutcome, ScenarioSet, StepRecord
from .guardrails import GuardrailResult, apply_guardrails
from .score import ClusterScore, score_clusters
from .weights import AdaptedCriteria, adapt


@dataclass
class Decision:
    """One interval's decision, with everything needed to explain or replay it."""

    interval: int
    seed: int
    snapshot: StateSnapshot
    criteria: AdaptedCriteria
    guardrails: tuple[GuardrailResult, ...]
    surviving: tuple[int, ...]
    scores: list[ClusterScore]
    chosen_index: int
    held_by_hysteresis: bool
    trace: tuple[StepRecord, ...]          # the committed path, step 0 = this interval
    scenarios_evaluated: int
    clusters_evaluated: int
    steps_evaluated: int
    latency_ms: float
    outcome: DispatchOutcome | None = None
    explanation: str = ""
    explanation_source: str = "template"   # template | llm
    tool_calls: tuple[dict, ...] = ()
    fallback_reason: str | None = None
    auto_approved: bool = False

    @property
    def chosen(self) -> ActionCluster:
        return CLUSTERS[self.chosen_index]

    @property
    def chosen_score(self) -> ClusterScore:
        return next(s for s in self.scores if s.index == self.chosen_index)

    @property
    def runner_up(self) -> ClusterScore | None:
        for s in self.scores:
            if s.index != self.chosen_index and s.survived_guardrails:
                return s
        return None

    @property
    def top5(self) -> list[ClusterScore]:
        return [s for s in self.scores if s.survived_guardrails][:5]

    @property
    def pareto_front(self) -> tuple[int, ...]:
        return tuple(s.index for s in self.scores if s.on_pareto_front)

    @property
    def committed(self) -> StepRecord:
        return self.trace[0]

    @property
    def guardrails_removed(self) -> int:
        return sum(g.removed_count for g in self.guardrails)

    def material_alternative(self, *, min_rel: float = 0.02) -> ClusterScore | None:
        """The best alternative that is actually *different*.

        The nominal runner-up is often a sibling cluster differing only in a dimension
        that is a no-op in this state (market posture when batteries cannot help, the
        curtailment rule when prices are high), which makes "why this over that?"
        vacuous. This returns the best-scoring cluster that differs materially in at
        least one objective, preferring survivors and falling back to the best cluster
        a guardrail removed -- which is itself the more informative contrast.
        """
        mine = self.chosen_score.objectives
        def differs(other: ClusterScore) -> bool:
            for k in OBJECTIVES:
                scale = max(abs(mine[k]), abs(other.objectives[k]), 1e-9)
                if abs(mine[k] - other.objectives[k]) / scale > min_rel:
                    return True
            return False

        for s in self.scores:
            if s.index != self.chosen_index and s.survived_guardrails and differs(s):
                return s
        for s in self.scores:
            if s.index != self.chosen_index and differs(s):
                return s
        return None

    def objective_deltas(self, other: ClusterScore | None = None) -> dict[str, float]:
        """Chosen minus the alternative, per objective, in native units."""
        other = other or self.material_alternative() or self.runner_up
        if other is None:
            return {}
        mine = self.chosen_score.objectives
        return {k: mine[k] - other.objectives[k] for k in OBJECTIVES}

    def deciding_objective(self, other: ClusterScore | None = None) -> str | None:
        """The objective whose weighted contribution most favoured the winner."""
        other = other or self.material_alternative() or self.runner_up
        if other is None:
            return None
        mine = self.chosen_score.contributions
        diffs = {k: other.contributions[k] - mine[k] for k in OBJECTIVES}
        best = max(diffs, key=lambda k: diffs[k])
        return best if diffs[best] > 1e-12 else None


class Optimizer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.engine = DispatchEngine(config)

    def decide(
        self,
        snapshot: StateSnapshot,
        *,
        seed: int = 0,
        envelope: Envelope | None = None,
        previous_cluster_index: int | None = None,
        scenarios: ScenarioSet | None = None,
    ) -> Decision:
        t0 = time.perf_counter()
        o = self.config.objectives

        scen = scenarios or build_scenarios(
            self.config, snapshot, seed=seed, envelope=envelope)
        regimes = classify(self.config, snapshot, scen)
        criteria = adapt(self.config, regimes)

        outcome = self.engine.simulate(
            scen, snapshot.battery_soc_pct, snapshot.battery_available)
        surviving, guardrail_results = apply_guardrails(
            self.config, snapshot, scen, outcome)
        scores, chosen, held = score_clusters(
            self.config, outcome, criteria, surviving,
            previous_cluster_index=previous_cluster_index)

        # Re-run only the winner to capture the committed per-step detail, so the
        # actions executed are literally the ones that were scored.
        detailed = self.engine.simulate(
            scen, snapshot.battery_soc_pct, snapshot.battery_available,
            record=(chosen, 0))
        latency = (time.perf_counter() - t0) * 1000.0

        return Decision(
            interval=snapshot.interval,
            seed=seed,
            snapshot=snapshot,
            criteria=criteria,
            guardrails=guardrail_results,
            surviving=surviving,
            scores=scores,
            chosen_index=chosen,
            held_by_hysteresis=held,
            trace=detailed.trace(chosen, 0),
            scenarios_evaluated=scen.n_scenarios,
            clusters_evaluated=len(CLUSTERS),
            steps_evaluated=scen.n_steps,
            latency_ms=latency,
            outcome=outcome,
        )

    def decide_forced(
        self,
        snapshot: StateSnapshot,
        cluster_index: int,
        *,
        seed: int = 0,
        envelope: Envelope | None = None,
        scenarios: ScenarioSet | None = None,
    ) -> Decision:
        """Score the full field but commit a *named* cluster.

        Two callers need this: the benchmark baselines, so a rule-based run is
        interval-by-interval identical to the agent apart from the cluster chosen, and
        the fail-safe, which must dispatch the fixed rule when the search itself has
        failed. Both go through this engine, so the comparison is honest and the
        fallback is still bound by the same physics.
        """
        t0 = time.perf_counter()
        scen = scenarios or build_scenarios(
            self.config, snapshot, seed=seed, envelope=envelope)
        regimes = classify(self.config, snapshot, scen)
        criteria = adapt(self.config, regimes)
        outcome = self.engine.simulate(
            scen, snapshot.battery_soc_pct, snapshot.battery_available)
        surviving, guardrail_results = apply_guardrails(
            self.config, snapshot, scen, outcome)
        scores, _, _ = score_clusters(self.config, outcome, criteria, surviving)
        detailed = self.engine.simulate(
            scen, snapshot.battery_soc_pct, snapshot.battery_available,
            record=(cluster_index, 0))
        return Decision(
            interval=snapshot.interval,
            seed=seed,
            snapshot=snapshot,
            criteria=criteria,
            guardrails=guardrail_results,
            surviving=surviving,
            scores=scores,
            chosen_index=cluster_index,
            held_by_hysteresis=False,
            trace=detailed.trace(cluster_index, 0),
            scenarios_evaluated=scen.n_scenarios,
            clusters_evaluated=len(CLUSTERS),
            steps_evaluated=scen.n_steps,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            outcome=outcome,
        )

    # ---- turning the winning cluster into per-asset set-points ------------
    def set_points(self, decision: Decision) -> CommittedActions:
        """PRD F1.3: verb, asset, MW and reason for every set-point."""
        p = self.config.portfolio
        r = decision.committed
        cluster = decision.chosen
        regime_note = (", ".join(x.label for x in decision.criteria.regimes)
                       or "normal conditions")
        points: list[SetPoint] = []
        queued: list[SetPoint] = []

        for b in p.batteries:
            mw = r.battery_mw.get(b.id, 0.0)
            soc = r.battery_soc_pct.get(b.id, 0.0)
            if abs(mw) < 1e-6:
                points.append(SetPoint(
                    "hold", b.id, b.name, 0.0,
                    f"{cluster.battery.label} holds {b.name} at {soc:.0f}% SoC"))
                continue
            if r.forced_injection_mw > 1e-9 and mw > 0:
                reason = (f"Frequency {r.frequency_hz:.2f} Hz below "
                          f"{p.grid.frequency_reflex_hz} Hz forces injection")
            elif mw > 0:
                reason = f"{cluster.battery.label} at ${r.price_per_mwh:.0f}/MWh"
            else:
                reason = f"{cluster.battery.label} absorbing surplus"
            points.append(SetPoint(
                "discharge" if mw > 0 else "charge", b.id, b.name, abs(mw),
                f"{reason}; SoC now {soc:.0f}%"))

        if r.import_mw > 1e-6:
            points.append(SetPoint(
                "buy", "grid", "Grid import", r.import_mw,
                f"Cover {r.import_mw:.0f} MW deficit at "
                f"${r.price_per_mwh:.0f}/MWh + ${self.engine.spread:.0f} spread"))
        elif r.export_mw > 1e-6:
            points.append(SetPoint(
                "sell", "grid", "Grid export", r.export_mw,
                f"Sell surplus at ${r.price_per_mwh:.0f}/MWh"))

        for farm_id, mw in sorted(r.solar_curtailed_by_farm.items()):
            if mw > 1e-6:
                points.append(SetPoint(
                    "curtail", farm_id, p.solar(farm_id).name, mw,
                    self._curtail_reason(r)))
        for farm_id, mw in sorted(r.wind_curtailed_by_farm.items()):
            if mw > 1e-6:
                points.append(SetPoint(
                    "curtail", farm_id, p.wind(farm_id).name, mw,
                    self._curtail_reason(r)))

        if r.dr_mw > 1e-6:
            contracted = p.total_dr_mw
            for c in p.consumers:
                if c.dr_mw <= 0:
                    continue
                share = r.dr_mw * c.dr_mw / max(contracted, 1e-9)
                beyond = r.dr_mw > contracted + 1e-6
                sp = SetPoint(
                    "reduce", c.id, c.name, share,
                    f"Demand response at ${r.price_per_mwh:.0f}/MWh ({regime_note})",
                    requires_approval=beyond)
                (queued if beyond else points).append(sp)

        if abs(r.shift_mw) > 1e-6:
            for c in p.consumers:
                if c.shiftable_mw <= 0:
                    continue
                share = r.shift_mw * c.shiftable_mw / max(p.total_shiftable_mw, 1e-9)
                verb = "shift" if share > 0 else "restore"
                points.append(SetPoint(
                    verb, c.id, c.name, abs(share),
                    f"Shift {abs(share):.0f} MW off the ${r.price_per_mwh:.0f}/MWh "
                    f"peak, repaid off-peak" if share > 0
                    else f"Restore {abs(share):.0f} MW of previously shifted load "
                         f"at ${r.price_per_mwh:.0f}/MWh"))

        if r.unserved_mw > 1e-6:
            points.append(SetPoint(
                "shed", "load", "Unserved load", r.unserved_mw,
                f"Imports at the {r.import_limit_mw:.0f} MW limit and batteries at "
                f"full discharge; shedding is unavoidable",
                requires_approval=True))

        return CommittedActions(
            interval=decision.interval,
            cluster=cluster,
            set_points=tuple(points),
            queued_for_approval=tuple(queued),
            notes=(f"{len(decision.surviving)} of {len(CLUSTERS)} clusters survived "
                   f"the guardrails",),
        )

    @staticmethod
    def _curtail_reason(r: StepRecord) -> str:
        if r.curtailed_by_corridor_mw > 1e-6:
            north = r.corridor_flow_mw.get("north", 0.0)
            cap = r.corridor_limit_mw.get("north", 0.0)
            return f"North corridor at its {cap:.0f} MW limit (flow {north:.0f} MW)"
        if r.curtailed_at_negative_price_mw > 1e-6:
            return f"Price is -${abs(r.price_per_mwh):.2f}/MWh; exporting would pay to generate"
        return (f"Export line at its {r.export_limit_mw:.0f} MW limit with batteries "
                f"absorbing {r.battery_charge_mw:.0f} MW")
