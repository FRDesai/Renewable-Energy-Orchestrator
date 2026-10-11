"""The live session: one running portfolio, one simulated clock, one frame per interval.

This is the console's whole world. It owns the synthetic plant, the normaliser, the
optimizer, the LLM orchestrator, the actuation gateway, the maintenance planner and the
audit log, and it renders one JSON *frame* per interval -- everything the four tabs read.

Two rules shape the design:

**A plan is dispatched when it is made, and the plant responds when the clock moves.**
`step()` applies the standing plan to the world, advances 15 minutes, then plans the new
interval and dispatches it. So the frame always shows a plan that is already out to the
assets, which is what a control room actually looks like, and the 15-minute physical
response is visible on the next tick.

**An event re-plans the interval it arrives in.** PRD F1.4 requires a new plan within one
evaluation of an event rather than at the next tick, so `inject()` re-runs the decision
for the *current* interval without moving the clock, and reports the plan before against
the plan after. The superseded plan stays in the append-only audit log.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..act.gateway import ActuationGateway, GatewayResult
from ..act.rule_based import fallback as rule_based_fallback
from ..agent.client import LLMClient
from ..agent.loop import AgentOrchestrator, TickContext, TickOutcome
from ..agent.qa import AgentQA
from ..audit.store import AuditLog, decision_record
from ..config_loader import Config
from ..decide.optimizer import Decision, Optimizer
from ..domain.actions import CommittedActions, SetPoint
from ..domain.events import Event, EventType, Severity
from ..domain.state import StateSnapshot
from ..explain.template import situation as situation_text, criteria_text, options_text, \
    tradeoff_text
from ..forecast.quantile import QuantileForecaster
from ..forecast.scenarios import Envelope, ScenarioSet, build_scenarios
from ..ingest.extractor import BulletinExtractor, ExtractionResult
from ..ingest.normalize import STALE_ALARM_INTERVALS, Normalizer
from ..maintenance.planner import MaintenancePlan, MaintenancePlanner
from ..world.presets import DEFAULT_PRESET, DEMO_OPENING_INTERVAL, PRESETS
from ..world.world import AppliedDispatch, World

HISTORY_INTERVALS = 48          # PRD: "last 12 h actuals stacked by source"
DEMO_WARMUP_INTERVALS = 36      # open at 09:00 with the morning already behind us

# One button per situation (PRD "Event injector"), so the console sends only a type.
# Magnitudes are demo-visible but deliberately inside the reference scenario's range.
INJECTION_DEFAULTS: dict[EventType, dict[str, Any]] = {
    EventType.CLOUD_BANK: {"region": None, "magnitude": 0.85, "duration": 8},
    EventType.WIND_SURGE: {"corridor": None, "magnitude": 1.45, "duration": 10},
    EventType.PRICE_SPIKE: {"magnitude": 6.5, "duration": 10},
    EventType.BATTERY_TRIP: {"assets": None, "duration": 24},
    EventType.LINE_CONGESTION: {"corridor": None, "duration": 24},
    EventType.DEMAND_SURGE: {"magnitude": 1.18, "duration": 8},
    EventType.STORM_ALERT: {"severity": Severity.SEVERE, "lead": 20, "duration": 40},
    EventType.FREQUENCY_DIP: {"magnitude": 0.25, "duration": 2},
    EventType.ISO_DR_CALL: {"duration": 8},
    EventType.FORECAST_REVISION: {"magnitude": 1.0, "duration": 8},
    EventType.DATA_DROPOUT: {"channels": ("solar", "price"), "duration": 4},
}

EVENT_LABELS = {
    EventType.CLOUD_BANK: "Cloud bank",
    EventType.WIND_SURGE: "Wind surge",
    EventType.PRICE_SPIKE: "Price spike",
    EventType.BATTERY_TRIP: "Battery trip",
    EventType.LINE_CONGESTION: "Line congestion",
    EventType.DEMAND_SURGE: "Demand surge",
    EventType.STORM_ALERT: "Storm alert",
    EventType.FREQUENCY_DIP: "Frequency dip",
    EventType.ISO_DR_CALL: "ISO demand-response call",
    EventType.FORECAST_REVISION: "Forecast revision",
    EventType.DATA_DROPOUT: "Data dropout",
}

MODE_AGENT = "agent"
MODE_FALLBACK = "rule_based_fallback"
MODE_DEGRADED = "degraded"


def _clock(minute_of_day: int) -> str:
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"


def _f(value: Any, places: int = 2) -> float:
    return round(float(value), places)


@dataclass
class Plan:
    """One interval's standing plan: what was decided and what the gateway did with it."""

    interval: int
    mode: str
    reason: str
    snapshot: StateSnapshot
    envelope: Envelope | None
    scenarios: ScenarioSet | None
    decision: Decision | None
    actions: CommittedActions | None
    gate: GatewayResult | None
    outcome: TickOutcome | None = None
    maintenance: MaintenancePlan | None = None
    fallback_reason: str | None = None
    reserve_held_pct: float | None = None
    evaluation_ms: float = 0.0
    cycle_ms: float = 0.0

    @property
    def executed_battery_mw(self) -> dict[str, float]:
        """Only executed set-points move the plant."""
        out: dict[str, float] = {}
        if self.gate is None:
            return out
        for sp in self.gate.executed:
            if sp.verb == "discharge":
                out[sp.asset_id] = out.get(sp.asset_id, 0.0) + sp.mw
            elif sp.verb == "charge":
                out[sp.asset_id] = out.get(sp.asset_id, 0.0) - sp.mw
        return out

    @property
    def executed_dr_mw(self) -> float:
        return sum(sp.mw for sp in (self.gate.executed if self.gate else ())
                   if sp.verb == "reduce")

    def label(self) -> str:
        if self.decision is not None:
            return self.decision.chosen.short_label
        return "Hold all assets (degraded)"


class LiveSession:
    """One console session. Single-threaded: the API serialises access to it."""

    def __init__(
        self,
        config: Config,
        *,
        preset: str = DEFAULT_PRESET,
        seed: int = 7,
        open_at: int = DEMO_OPENING_INTERVAL,
        warmup_intervals: int = DEMO_WARMUP_INTERVALS,
        forecaster: QuantileForecaster | None = None,
        llm: LLMClient | None = None,
        audit_path: Path | None = None,
    ) -> None:
        if preset not in PRESETS:
            raise KeyError(f"unknown preset {preset!r}; options are {sorted(PRESETS)}")
        self.config = config
        self.preset = preset
        self.seed = int(seed)
        self.open_at = int(open_at)
        self.warmup_intervals = max(0, min(int(warmup_intervals), self.open_at))
        self.forecaster = forecaster
        self.llm = llm or LLMClient()
        self.audit_path = audit_path

        self.world = World(config, seed=self.seed, preset=preset,
                           start_interval=self.open_at - self.warmup_intervals)
        self.normalizer = Normalizer(config, substitute=self._substitute_factory())
        self.audit = AuditLog(path=audit_path, keep_in_memory=True)

        self.history: list[dict[str, Any]] = []
        self.log: list[dict[str, Any]] = []        # newest first
        self.work_orders: list[dict[str, Any]] = []
        self.maintenance_proposals: list[dict[str, Any]] = []
        self.pending_bulletin: ExtractionResult | None = None
        self.previous_cluster_index: int | None = None
        self.last_valid_decision: Decision | None = None
        self.current: Plan | None = None
        self.response_to_change: dict[str, Any] | None = None
        self.running = False
        self.speed_x = 1.0
        self._replans: dict[int, int] = {}
        self._proposal_counter = 0

        self._build_engine()
        self._warm_up()

    # ---- construction -----------------------------------------------------
    def _build_engine(self) -> None:
        """(Re)build everything that reads the config. Config is immutable, so an
        operator weight change produces a new one and these follow it."""
        self.optimizer = Optimizer(self.config)
        self.agent = AgentOrchestrator(self.config, self.llm)
        self.planner = MaintenancePlanner(self.config)
        self.qa = AgentQA(self.config, self.llm)
        self.extractor = BulletinExtractor(self.config, self.llm)
        if not hasattr(self, "gateway"):
            self.gateway = ActuationGateway(self.config, auto_approve=False)
        else:
            self.gateway.config = self.config

    def _substitute_factory(self):
        self._latest_snapshot: StateSnapshot | None = None

        def substitute(channel, interval, last_good):
            if self.forecaster is None or not self.forecaster.is_trained:
                return last_good
            return self.forecaster.substitute(
                self._latest_snapshot, channel, interval, last_good)
        return substitute

    def _warm_up(self) -> None:
        """Fill the chart with real history so the console never opens empty.

        The warm-up runs the optimizer directly, in batch mode: there is no operator
        to approve anything before the session exists (DECISIONS.md D9), and routing
        36 warm-up intervals through the LLM would cost 36 API calls to produce
        history nobody reads the prose of.
        """
        batch = ActuationGateway(self.config, auto_approve=True)
        live, self.gateway = self.gateway, batch
        try:
            for _ in range(self.warmup_intervals):
                self._plan(use_agent=False, reason="warm-up history (batch mode)")
                self._commit()
        finally:
            self.gateway = live
        if self.warmup_intervals:
            self._note("warmup", self.world.interval,
                       f"{self.warmup_intervals} interval(s) of history generated in "
                       f"batch mode, so the console opens at "
                       f"{_clock(self.world.minute_of_day)} already running")
        self._plan(use_agent=True, reason="session opened")

    # ---- the decision cycle ----------------------------------------------
    def _plan(self, *, use_agent: bool = True, reason: str = "scheduled interval") -> Plan:
        """Decide and dispatch the current interval. Does not move the clock."""
        t0 = time.perf_counter()
        cfg = self.config
        # Any approval still pending belongs to the plan about to be replaced, or to
        # an interval already dispatched. Either way it is no longer actionable.
        for req in self.gateway.withdraw_through(
                self.world.interval, f"superseded: {reason}"):
            self._note("approval_withdrawn", req.interval,
                       f"Withdrew {req.set_point.text}: the plan it belonged to was "
                       f"replaced", severity="warning")
        snapshot = self.normalizer.build(self.world.telemetry())
        self._latest_snapshot = snapshot
        seed = self.seed * 1000 + snapshot.interval
        envelope = (self.forecaster.envelope(snapshot, cfg.objectives.steps)
                    if self.forecaster is not None and self.forecaster.is_trained
                    else None)
        scenarios = build_scenarios(cfg, snapshot, seed=seed, envelope=envelope)

        # PRD fail-safe: a feed stale beyond two intervals is not a basis for
        # optimisation, so the fixed rule takes over before the search is even tried.
        stale_reason = None
        if self.normalizer.alarm:
            worst = self.normalizer.worst_stale_intervals
            channels = ", ".join(sorted(snapshot.quality.stale_fields)) or "a feed"
            stale_reason = (f"{channels} stale for {worst} intervals, beyond the "
                            f"{STALE_ALARM_INTERVALS}-interval limit")

        plan = (self._fallback_plan(snapshot, envelope, scenarios, seed, stale_reason,
                                    reason)
                if stale_reason is not None
                else self._agent_plan(snapshot, envelope, scenarios, seed,
                                      use_agent=use_agent, reason=reason))
        plan.cycle_ms = (time.perf_counter() - t0) * 1000.0

        plan.maintenance = self._plan_maintenance(plan)
        self._raise_work_orders(snapshot)
        self.current = plan
        if plan.decision is not None:
            self.last_valid_decision = plan.decision
        self._record(plan)
        return plan

    def _agent_plan(self, snapshot, envelope, scenarios, seed, *, use_agent: bool,
                    reason: str) -> Plan:
        try:
            if use_agent:
                return self._run_agent(snapshot, envelope, scenarios, seed, reason)
            return self._run_optimizer(snapshot, envelope, scenarios, seed, reason)
        except Exception as exc:
            return self._fallback_plan(snapshot, envelope, scenarios, seed,
                                       f"optimizer failed: {exc}", reason)

    def _run_agent(self, snapshot, envelope, scenarios, seed, reason: str) -> Plan:
        price_path, asset_output = self._median_paths(scenarios)
        ctx = TickContext(
            config=self.config, snapshot=snapshot, optimizer=self.optimizer,
            envelope=envelope, scenarios=scenarios, seed=seed,
            previous_cluster_index=self.previous_cluster_index,
            gateway=self.gateway, planner=self.planner,
            price_path=price_path, asset_output=asset_output,
            last_valid_decision=self.last_valid_decision)
        outcome = self.agent.run_tick(ctx)
        decision = outcome.decision
        decision.explanation = outcome.explanation
        decision.explanation_source = outcome.explanation_source
        decision.fallback_reason = outcome.fallback_reason
        gate = ctx.gateway_result
        if gate is None:
            raise RuntimeError("the tick produced no gateway result")
        return Plan(
            interval=snapshot.interval, mode=MODE_AGENT, reason=reason,
            snapshot=snapshot, envelope=envelope, scenarios=scenarios,
            decision=decision, actions=ctx.actions, gate=gate, outcome=outcome,
            fallback_reason=outcome.fallback_reason,
            evaluation_ms=decision.latency_ms)

    def _run_optimizer(self, snapshot, envelope, scenarios, seed, reason: str) -> Plan:
        """The warm-up path: optimizer and gateway only, template explanation."""
        from ..explain.template import explain as template_explain
        decision = self.optimizer.decide(
            snapshot, seed=seed, scenarios=scenarios,
            previous_cluster_index=self.previous_cluster_index)
        decision.explanation = template_explain(self.config, decision)
        decision.explanation_source = "template"
        actions = self.optimizer.set_points(decision)
        gate = self.gateway.submit(actions)
        return Plan(
            interval=snapshot.interval, mode=MODE_AGENT, reason=reason,
            snapshot=snapshot, envelope=envelope, scenarios=scenarios,
            decision=decision, actions=actions, gate=gate,
            evaluation_ms=decision.latency_ms)

    def _fallback_plan(self, snapshot, envelope, scenarios, seed, cause: str,
                       reason: str) -> Plan:
        """PRD: "falls back to the rule-based dispatch with reserve held, raises an
        alarm, and logs the cause"."""
        from ..explain.template import explain as template_explain
        fb = rule_based_fallback(self.config, snapshot, cause)
        try:
            decision = self.optimizer.decide_forced(
                snapshot, fb.cluster.index, seed=seed, scenarios=scenarios)
            decision.explanation = template_explain(self.config, decision)
            decision.explanation_source = "template"
            decision.fallback_reason = cause
            actions = self.optimizer.set_points(decision)
            gate = self.gateway.submit(actions)
            plan = Plan(
                interval=snapshot.interval, mode=MODE_FALLBACK, reason=reason,
                snapshot=snapshot, envelope=envelope, scenarios=scenarios,
                decision=decision, actions=actions, gate=gate,
                fallback_reason=cause, reserve_held_pct=fb.reserve_held_pct,
                evaluation_ms=decision.latency_ms)
        except Exception as exc:
            plan = self._degraded_plan(snapshot, envelope, fb.cluster,
                                       f"{cause}; rule-based dispatch also failed: {exc}",
                                       fb.reserve_held_pct, reason)
        self._note("fallback", snapshot.interval, plan.fallback_reason or cause,
                   severity="critical")
        return plan

    def _degraded_plan(self, snapshot, envelope, cluster, cause: str,
                       reserve_held_pct: float, reason: str) -> Plan:
        """Last resort: hold every asset where it is. Holding a known-safe state beats
        dispatching a figure no engine produced."""
        points = tuple(
            SetPoint("hold", b.id, b.name, 0.0,
                     f"Engine unavailable; {b.name} held at "
                     f"{snapshot.battery_soc_pct.get(b.id, 0.0):.0f}% SoC with reserve "
                     f"held at {reserve_held_pct:.0f}%")
            for b in self.config.portfolio.batteries)
        actions = CommittedActions(
            interval=snapshot.interval, cluster=cluster, set_points=points,
            notes=(cause,))
        gate = self.gateway.submit(actions)
        return Plan(
            interval=snapshot.interval, mode=MODE_DEGRADED, reason=reason,
            snapshot=snapshot, envelope=envelope, scenarios=None,
            decision=None, actions=actions, gate=gate,
            fallback_reason=cause, reserve_held_pct=reserve_held_pct)

    def _median_paths(self, scenarios: ScenarioSet) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Scenario 0 is the expected path, which is what the maintenance planner values
        a window against."""
        price = np.asarray(scenarios.price_per_mwh[0], dtype=float)
        output = {fid: np.asarray(series[0], dtype=float)
                  for fid, series in {**scenarios.solar_by_farm,
                                      **scenarios.wind_by_farm}.items()}
        return price, output

    def _plan_maintenance(self, plan: Plan) -> MaintenancePlan | None:
        if plan.scenarios is None:
            return None
        price_path, asset_output = self._median_paths(plan.scenarios)
        mp = self.planner.plan(plan.snapshot, price_path=price_path,
                              asset_output=asset_output)
        known = {p["window_id"] for p in self.maintenance_proposals}
        for v in mp.proposed_delays:
            if v.window_id in known:
                continue
            self._proposal_counter += 1
            self.maintenance_proposals.append({
                "id": f"mp-{plan.interval}-{self._proposal_counter}",
                "kind": "maintenance", "status": "pending",
                "window_id": v.window_id, "asset_name": v.asset_name,
                "interval": plan.interval,
                "summary": f"Move {v.asset_name} maintenance window: {v.reason}",
                "saving_usd": _f(v.saving_usd, 0),
                "current_start": v.current_start, "proposed_start": v.best_start,
            })
            self._note("maintenance_proposed", plan.interval,
                       f"{v.asset_name}: {v.reason}")
        return mp

    def _raise_work_orders(self, snapshot: StateSnapshot) -> None:
        """PRD fault tolerance: "Battery trip -> immediate re-plan and inspection
        dispatch". A system reflex, not something the LLM has to remember."""
        known = {o["id"] for o in self.work_orders}
        for order in self.planner.inspections_for(snapshot):
            if order.id in known:
                continue
            self.work_orders.append({
                "id": order.id, "kind": "crew", "status": "pending",
                "asset_id": order.asset_id, "reason": order.reason,
                "interval": order.raised_interval,
                "summary": f"Dispatch inspection crew to {order.asset_id}: {order.reason}",
            })
            self._note("crew_dispatch_proposed", snapshot.interval, order.reason)

    def _record(self, plan: Plan) -> None:
        """Append-only audit entry at dispatch time, plus the operator-facing log."""
        if plan.decision is not None:
            seen = self._replans.get(plan.interval, 0)
            if seen:
                self._note("replan", plan.interval,
                           f"plan superseded after {plan.reason}")
            self._replans[plan.interval] = seen + 1
            self.audit.append("decision", plan.interval, decision_record(
                plan.decision, plan.actions, plan.gate,
                preset=self.preset, world_seed=self.seed,
                config_version=self.config.portfolio.version))
        self._note("decision", plan.interval,
                   f"{plan.label()} -- {plan.reason}",
                   detail={"mode": plan.mode,
                           "evaluation_ms": _f(plan.evaluation_ms, 2),
                           "set_points": len(plan.gate.executed) if plan.gate else 0})

    def _commit(self) -> None:
        """Apply the standing plan to the plant and advance the clock 15 minutes."""
        plan = self.current
        assert plan is not None
        h = self.config.portfolio.interval_hours
        record = plan.decision.committed if plan.decision is not None else None
        battery_mw = plan.executed_battery_mw

        self.world.apply(AppliedDispatch(
            battery_mw={b.id: battery_mw.get(b.id, 0.0)
                        for b in self.config.portfolio.batteries},
            dr_mw=plan.executed_dr_mw,
            shift_mw=record.shift_mw if record is not None else 0.0,
            grid_mw=record.grid_mw if record is not None else 0.0,
            unserved_mwh=(record.unserved_mw * h) if record is not None else 0.0,
        ))
        self.history.append(self._history_row(plan, battery_mw))
        if len(self.history) > HISTORY_INTERVALS:
            del self.history[:-HISTORY_INTERVALS]
        if plan.decision is not None:
            self.previous_cluster_index = plan.decision.chosen_index
        self.world.step()
        self.response_to_change = None

    def _history_row(self, plan: Plan, battery_mw: dict[str, float]) -> dict[str, Any]:
        """What actually happened, built from executed set-points rather than proposals."""
        s = plan.snapshot
        r = plan.decision.committed if plan.decision is not None else None
        return {
            "interval": plan.interval,
            "time": _clock(s.minute_of_day),
            "solar_mw": _f(s.total_solar_mw, 1),
            "wind_mw": _f(s.total_wind_mw, 1),
            "curtailed_mw": _f(r.total_curtailed_mw if r else 0.0, 1),
            "battery_discharge_mw": _f(sum(v for v in battery_mw.values() if v > 0), 1),
            "battery_charge_mw": _f(sum(-v for v in battery_mw.values() if v < 0), 1),
            "import_mw": _f(r.import_mw if r else 0.0, 1),
            "export_mw": _f(r.export_mw if r else 0.0, 1),
            "demand_mw": _f(s.total_demand_mw, 1),
            "dr_mw": _f(plan.executed_dr_mw, 1),
            "unserved_mw": _f(r.unserved_mw if r else 0.0, 2),
            "price_per_mwh": _f(s.price_per_mwh, 2),
            "fleet_soc_pct": _f(self._fleet_soc_pct(self.world.battery_soc_pct), 1),
            "mode": plan.mode,
        }

    def _fleet_soc_pct(self, soc: dict[str, float]) -> float:
        p = self.config.portfolio
        total = p.battery_energy_mwh
        if total <= 0:
            return 0.0
        return sum(soc.get(b.id, 0.0) * b.energy_mwh for b in p.batteries) / total

    # ---- public controls --------------------------------------------------
    @property
    def frame(self) -> dict[str, Any]:
        return self._frame()

    def step(self, intervals: int = 1) -> dict[str, Any]:
        for _ in range(max(1, int(intervals))):
            self._commit()
            self._plan(reason="scheduled interval")
        return self._frame()

    def inject(self, kind: str | EventType, **spec: Any) -> dict[str, Any]:
        """One-click situation injection, then an immediate re-plan (PRD F1.4)."""
        event_type = kind if isinstance(kind, EventType) else EventType(kind)
        merged = {k: v for k, v in INJECTION_DEFAULTS[event_type].items()
                  if v is not None}
        merged.update(spec)
        if event_type is EventType.CLOUD_BANK and "region" not in merged:
            merged["region"] = self.config.portfolio.cloud_regions[0]
        if event_type in (EventType.WIND_SURGE, EventType.LINE_CONGESTION) \
                and "corridor" not in merged:
            merged["corridor"] = self.config.portfolio.corridors[0]
        if event_type is EventType.BATTERY_TRIP and "assets" not in merged:
            merged["assets"] = (self.config.portfolio.batteries[0].id,)

        event = self.world.inject(event_type, **merged)
        self._note("event_injected", self.world.interval,
                   f"{EVENT_LABELS[event_type]} injected",
                   detail=self._event_dict(event))
        self._replan_after(self._event_dict(event),
                           f"{EVENT_LABELS[event_type]} injected")
        return {"event": self._event_dict(event),
                "response_to_change": self.response_to_change}

    def _replan_after(self, event: dict[str, Any] | None, reason: str) -> None:
        """Re-decide the current interval and report the plan before against after."""
        before = self.current
        before_summary = self._plan_summary(before)
        t0 = time.perf_counter()
        after = self._plan(reason=reason)
        elapsed = (time.perf_counter() - t0) * 1000.0
        after_summary = self._plan_summary(after)
        self.response_to_change = {
            "trigger": reason,
            "event": event,
            "before": before_summary,
            "after": after_summary,
            "changed_actions": self._action_diff(before, after),
            # The PRD's acceptance is "a new plan within one evaluation (< 1 s)", so
            # the measured figure is the optimizer evaluation, reported beside the
            # full cycle including the orchestrator's narration.
            "evaluation_ms": _f(after.evaluation_ms, 2),
            "cycle_ms": _f(elapsed, 2),
        }

    def _plan_summary(self, plan: Plan | None) -> dict[str, Any]:
        if plan is None:
            return {"cluster": None}
        r = plan.decision.committed if plan.decision is not None else None
        return {
            "cluster": plan.label(),
            "cluster_index": plan.decision.chosen_index if plan.decision else None,
            "mode": plan.mode,
            "battery_mw": {k: _f(v, 1) for k, v in (r.battery_mw if r else {}).items()},
            "import_mw": _f(r.import_mw if r else 0.0, 1),
            "export_mw": _f(r.export_mw if r else 0.0, 1),
            "curtailed_mw": _f(r.total_curtailed_mw if r else 0.0, 1),
            "dr_mw": _f(r.dr_mw if r else 0.0, 1),
            "unserved_mw": _f(r.unserved_mw if r else 0.0, 2),
            "regimes": [x.label for x in plan.decision.criteria.regimes] if plan.decision else [],
        }

    @staticmethod
    def _action_diff(before: Plan | None, after: Plan) -> list[dict[str, Any]]:
        def key(sp: SetPoint) -> tuple[str, str]:
            return (sp.verb, sp.asset_id)

        old = {key(sp): sp for sp in (before.actions.set_points if before
                                      and before.actions else ())}
        new = {key(sp): sp for sp in (after.actions.set_points if after.actions else ())}
        changed: list[dict[str, Any]] = []
        for k in sorted(set(old) | set(new)):
            a, b = old.get(k), new.get(k)
            if a is not None and b is not None and abs(a.mw - b.mw) < 0.05:
                continue
            changed.append({
                "verb": k[0], "asset_id": k[1],
                "asset_name": (b or a).asset_name,
                "before_mw": _f(a.mw, 1) if a is not None else None,
                "after_mw": _f(b.mw, 1) if b is not None else None,
                "reason": (b or a).reason,
            })
        return changed

    # ---- bulletins (D2 text input) ---------------------------------------
    def submit_bulletin(self, text: str) -> dict[str, Any]:
        """Extract an event from pasted free text. Nothing reaches the optimizer yet:
        a high-impact or low-confidence extraction waits for the operator."""
        result = self.extractor.extract(text, interval=self.world.interval)
        body = result.as_dict()
        body["text"] = text
        # One event shape on the wire. The extractor's own dict omits the label, the
        # arrival countdown and the confidence-as-percentage that every other event in
        # the frame carries, and the console reads those -- so an extraction would have
        # arrived with three undefined fields.
        if result.event is not None:
            body["event"] = self._event_dict(result.event)
        if result.event is None:
            self.pending_bulletin = None
            self._note("bulletin_rejected", self.world.interval,
                       "No event extracted from the bulletin",
                       detail={"issues": list(result.issues)})
            return body
        self._note("bulletin_extracted", self.world.interval,
                   f"{EVENT_LABELS[result.event.type]} extracted from a bulletin "
                   f"(confidence {result.event.extraction_confidence:.2f})",
                   detail=body)
        if result.needs_confirmation:
            self.pending_bulletin = result
            return body
        self.pending_bulletin = None
        self._accept_event(result.event, "bulletin")
        return body

    def confirm_bulletin(self, accept: bool) -> dict[str, Any]:
        if self.pending_bulletin is None:
            raise LookupError("no bulletin is awaiting confirmation")
        result, self.pending_bulletin = self.pending_bulletin, None
        event = result.event
        assert event is not None
        if not accept:
            self._note("bulletin_rejected", self.world.interval,
                       f"Operator rejected the extracted "
                       f"{EVENT_LABELS[event.type]}", detail=result.as_dict())
            return {"accepted": False, "event": None,
                    "response_to_change": None}
        event.confirmed_by_operator = True
        self._accept_event(event, "bulletin confirmed by the operator")
        return {"accepted": True, "event": self._event_dict(event),
                "response_to_change": self.response_to_change}

    def _accept_event(self, event: Event, source: str) -> None:
        self.world.add_event(event)
        self._note("event_accepted", self.world.interval,
                   f"{EVENT_LABELS[event.type]} accepted ({source})",
                   detail=self._event_dict(event))
        self._replan_after(self._event_dict(event),
                           f"{EVENT_LABELS[event.type]} accepted ({source})")

    # ---- operator controls ----------------------------------------------
    def set_weights(self, weights: dict[str, float] | None = None, *,
                    risk_lambda: float | None = None,
                    note: str = "operator slider") -> dict[str, Any]:
        """F2.6: change base weights or risk appetite and see the decision change."""
        if not weights and risk_lambda is None:
            raise ValueError("nothing to change")
        before = self.current
        before_index = before.decision.chosen_index if before and before.decision else None
        t0 = time.perf_counter()
        if weights:
            self.config = self.config.with_base_weights(weights, note=note)
        if risk_lambda is not None:
            self.config = self.config.with_risk_lambda(risk_lambda, note=note)
        self._build_engine()
        self._replan_after(None, "operator changed the criteria")
        elapsed = (time.perf_counter() - t0) * 1000.0
        after = self.current
        self._note("weights_changed", self.world.interval,
                   f"Operator criteria changed ({note})",
                   detail={"weights": weights or {}, "risk_lambda": risk_lambda,
                           "elapsed_ms": _f(elapsed, 2)})
        return {
            "elapsed_ms": _f(elapsed, 2),
            "before_index": before_index,
            "after_index": after.decision.chosen_index if after and after.decision else None,
            "base_weights": {k: _f(v, 4)
                             for k, v in self.config.objectives.base_weights.items()},
            "risk_lambda": _f(self.config.objectives.risk["lambda_base"], 3),
            "changed": before_index != (after.decision.chosen_index
                                        if after and after.decision else None),
        }

    def proposals(self) -> list[dict[str, Any]]:
        """The approval queue: high-impact set-points, maintenance moves, crew
        dispatch. One list, because the operator works one queue."""
        out: list[dict[str, Any]] = []
        for req in self.gateway.queue:
            body = req.as_dict()
            body["kind"] = "set_point"
            body["summary"] = f"{req.set_point.text} -- {req.reason}"
            out.append(body)
        out.extend(self.maintenance_proposals)
        out.extend(self.work_orders)
        return out

    def pending_proposals(self) -> list[dict[str, Any]]:
        return [p for p in self.proposals() if p["status"] == "pending"]

    def decide_approval(self, proposal_id: str, approve: bool, *,
                        by: str = "operator") -> dict[str, Any]:
        for req in self.gateway.queue:
            if req.id == proposal_id:
                decided = self.gateway.decide(proposal_id, approve, by=by)
                self._note("approval", decided.interval,
                           f"{'Approved' if approve else 'Rejected'} "
                           f"{decided.set_point.text}",
                           detail=decided.as_dict())
                body = decided.as_dict()
                body["kind"] = "set_point"
                return body
        for pool, kind in ((self.maintenance_proposals, "maintenance"),
                           (self.work_orders, "crew")):
            for item in pool:
                if item["id"] != proposal_id:
                    continue
                if item["status"] != "pending":
                    raise ValueError(
                        f"proposal {proposal_id!r} is {item['status']} and can no "
                        f"longer be actioned")
                item["status"] = "approved" if approve else "rejected"
                item["decided_by"] = by
                # A rejection is preference data (PRD Trainability).
                self.gateway.overrides.append({
                    "kind": kind, "id": proposal_id, "interval": item["interval"],
                    "approved": approve, "by": by, "summary": item["summary"],
                })
                self._note("approval", item["interval"],
                           f"{'Approved' if approve else 'Rejected'} {item['summary']}",
                           detail=item)
                return dict(item)
        raise LookupError(f"no proposal {proposal_id!r}")

    def override(self, note: str, *, by: str = "operator") -> dict[str, Any]:
        """Operator override always wins and is always logged."""
        entry = self.gateway.override(self.world.interval, note, by=by)
        self.audit.append("override", self.world.interval, dict(entry))
        self._note("override", self.world.interval, f"Operator override: {note}",
                   detail=dict(entry))
        return {"note": note, "by": by, "interval": self.world.interval}

    def ask(self, question: str) -> dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("a question is required")
        plan = self.current
        if plan is None or plan.decision is None:
            raise LookupError("no decision trace is available to answer from")
        answer = self.qa.ask(plan.decision, question)
        body = answer.as_dict()
        body["question"] = question
        self._note("question", plan.interval, f"Operator asked: {question}",
                   detail={"source": answer.source})
        return body

    def replay(self, interval: int) -> dict[str, Any] | None:
        """Observability: any past interval, replayable from its stored inputs."""
        entries = [e for e in self.audit.entries
                   if e.kind == "decision" and e.interval == interval]
        if not entries:
            return None
        body = entries[-1].body
        return {
            "replay_key": {
                "interval": interval, "seed": body.get("seed"),
                "preset": body.get("preset"), "world_seed": body.get("world_seed"),
                "chosen_index": body.get("chosen_index"),
                "config_version": body.get("config_version"),
            },
            "superseded": len(entries) - 1,
            "decision": jsonable(body),
        }

    def set_running(self, running: bool, *, speed_x: float | None = None) -> None:
        self.running = bool(running)
        if speed_x is not None:
            self.speed_x = max(0.25, min(600.0, float(speed_x)))
        self._note("clock", self.world.interval,
                   f"{'Running' if self.running else 'Paused'} at "
                   f"{self.speed_x:g}x")

    @property
    def tick_seconds(self) -> float:
        """Wall-clock seconds between ticks at the current speed."""
        return self.config.portfolio.interval_minutes * 60.0 / self.speed_x

    # ---- logging ---------------------------------------------------------
    def _note(self, kind: str, interval: int, message: str, *,
              severity: str = "info", detail: dict[str, Any] | None = None) -> None:
        self.log.insert(0, {
            "kind": kind, "interval": interval, "time": _clock(
                (interval * self.config.portfolio.interval_minutes) % 1440),
            "severity": severity, "message": message,
            "detail": jsonable(detail or {}),
        })
        del self.log[400:]

    def _event_dict(self, event: Event) -> dict[str, Any]:
        return {
            "type": event.type.value,
            "label": EVENT_LABELS[event.type],
            "severity": event.severity.value,
            "start_interval": event.start_interval,
            "end_interval": event.end_interval,
            "arrival_interval": event.arrival_interval,
            "intervals_to_arrival": event.intervals_to_arrival(self.world.interval),
            "region": event.region, "corridor": event.corridor,
            "affected_assets": list(event.affected_assets),
            "note": event.note,
            "extracted_by": event.extracted_by,
            "extraction_confidence_pct": _f(100.0 * event.extraction_confidence, 0),
            "confirmed_by_operator": event.confirmed_by_operator,
            "needs_confirmation": event.needs_confirmation,
            "source_phrases": dict(event.source_phrases),
        }

    # ---- the frame -------------------------------------------------------
    def _frame(self) -> dict[str, Any]:
        plan = self.current
        assert plan is not None, "a session always has a standing plan"
        return {
            "clock": self._clock_block(),
            "scenario": {"preset": self.preset,
                         "label": PRESETS[self.preset].label,
                         "description": PRESETS[self.preset].description,
                         "seed": self.seed},
            "status": self._status_block(plan),
            "portfolio": self._portfolio_block(plan),
            "history": list(self.history),
            "forecast": self._forecast_block(plan),
            "trace": self._trace_block(plan),
            "actions": self._actions_block(plan),
            "agent": self._agent_block(plan),
            "approvals": self.pending_proposals(),
            "maintenance": self._maintenance_block(plan),
            "events": [self._event_dict(e) for e in self.world.active_events()],
            "data_quality": self._quality_block(plan),
            "alarms": self._alarms(plan),
            "response_to_change": self.response_to_change,
        }

    def _clock_block(self) -> dict[str, Any]:
        return {
            "interval": self.world.interval,
            "minute_of_day": self.world.minute_of_day,
            "time": _clock(self.world.minute_of_day),
            "day": self.world.day,
            "interval_minutes": self.config.portfolio.interval_minutes,
            "running": self.running,
            "speed_x": self.speed_x,
        }

    def _status_block(self, plan: Plan) -> dict[str, Any]:
        o = plan.outcome
        return {
            "mode": plan.mode,
            "committed": True,
            "reason": plan.reason,
            "llm_available": bool(o.llm_available) if o is not None else self.llm.available,
            "explanation_source": (plan.decision.explanation_source
                                   if plan.decision is not None else "template"),
            "fallback_reason": plan.fallback_reason,
            "reserve_held_pct": (None if plan.reserve_held_pct is None
                                 else _f(plan.reserve_held_pct, 1)),
            "evaluation_ms": _f(plan.evaluation_ms, 2),
            "cycle_ms": _f(plan.cycle_ms, 2),
            "verification": (o.verification.as_dict()
                             if o is not None and o.verification is not None else None),
        }

    def _portfolio_block(self, plan: Plan) -> dict[str, Any]:
        p = self.config.portfolio
        s = plan.snapshot
        r = plan.decision.committed if plan.decision is not None else None
        executed = plan.executed_battery_mw
        maint = {m.asset_id: m for m in s.maintenance if m.status == "in_progress"}

        def status_of(asset_id: str, curtailed: float, output: float) -> str:
            if asset_id in maint:
                return "maintenance"
            if curtailed > 0.05:
                return "curtailed"
            return "online" if output > 0.05 else "idle"

        solar = []
        for f in p.solar_farms:
            out = s.solar_mw.get(f.id, 0.0)
            cut = (r.solar_curtailed_by_farm.get(f.id, 0.0) if r else 0.0)
            solar.append({
                "id": f.id, "name": f.name, "capacity_mw": f.capacity_mw,
                "output_mw": _f(out, 1), "curtailed_mw": _f(cut, 1),
                "cloud_region": f.cloud_region,
                "cloud_cover_pct": _f(100.0 * s.cloud_cover.get(f.cloud_region, 0.0), 0),
                "utilisation_pct": _f(100.0 * out / f.capacity_mw, 0),
                "status": status_of(f.id, cut, out),
            })
        wind = []
        for f in p.wind_farms:
            out = s.wind_mw.get(f.id, 0.0)
            cut = (r.wind_curtailed_by_farm.get(f.id, 0.0) if r else 0.0)
            wind.append({
                "id": f.id, "name": f.name, "capacity_mw": f.capacity_mw,
                "output_mw": _f(out, 1), "curtailed_mw": _f(cut, 1),
                "corridor": f.corridor,
                "wind_speed_ms": _f(s.wind_speed_ms.get(f.id, 0.0), 1),
                "utilisation_pct": _f(100.0 * out / f.capacity_mw, 0),
                "status": status_of(f.id, cut, out),
            })
        batteries = []
        for b in p.batteries:
            mw = executed.get(b.id, 0.0)
            available = s.battery_available.get(b.id, True)
            batteries.append({
                "id": b.id, "name": b.name, "power_mw": b.power_mw,
                "energy_mwh": b.energy_mwh,
                "soc_pct": _f(s.battery_soc_pct.get(b.id, 0.0), 1),
                "stored_mwh": _f(b.energy_mwh * s.battery_soc_pct.get(b.id, 0.0) / 100.0, 1),
                "dispatch_mw": _f(mw, 1),
                "available": available,
                "status": ("tripped" if not available
                           else "discharging" if mw > 0.05
                           else "charging" if mw < -0.05 else "holding"),
            })
        consumers = []
        contracted = p.total_dr_mw
        for c in p.consumers:
            share_dr = (r.dr_mw * c.dr_mw / contracted) if r and contracted > 0 else 0.0
            share_shift = ((r.shift_mw * c.shiftable_mw / p.total_shiftable_mw)
                           if r and p.total_shiftable_mw > 0 else 0.0)
            consumers.append({
                "id": c.id, "name": c.name,
                "demand_mw": _f(s.consumer_mw.get(c.id, 0.0), 1),
                "contracted_dr_mw": c.dr_mw, "shiftable_mw": c.shiftable_mw,
                "dr_mw": _f(share_dr, 1), "shift_mw": _f(share_shift, 1),
                "critical": c.critical,
                "status": ("protected" if c.critical
                           else "reducing" if share_dr > 0.05
                           else "shifted" if share_shift > 0.05 else "normal"),
            })
        corridors = [{
            "corridor": name,
            "flow_mw": _f(r.corridor_flow_mw.get(name, 0.0) if r else 0.0, 1),
            "limit_mw": _f(s.corridor_limit_mw.get(name, 0.0), 0),
            "status": self._limit_status(r.corridor_flow_mw.get(name, 0.0) if r else 0.0,
                                        s.corridor_limit_mw.get(name, 0.0)),
        } for name in p.corridors]

        storms = [self._event_dict(e) for e in s.events
                  if e.type is EventType.STORM_ALERT]
        return {
            "solar": solar, "wind": wind, "batteries": batteries,
            "consumers": consumers, "corridors": corridors,
            "municipal_mw": _f(s.municipal_mw, 1),
            "grid": {
                "import_mw": _f(r.import_mw if r else 0.0, 1),
                "export_mw": _f(r.export_mw if r else 0.0, 1),
                "import_limit_mw": _f(s.import_limit_mw, 0),
                "export_limit_mw": _f(s.export_limit_mw, 0),
                "price_per_mwh": _f(s.price_per_mwh, 2),
                "carbon_price_per_tonne": _f(s.carbon_price_per_tonne, 2),
                "frequency_hz": _f(s.frequency_hz, 3),
                "import_status": self._limit_status(r.import_mw if r else 0.0,
                                                    s.import_limit_mw),
                "export_status": self._limit_status(r.export_mw if r else 0.0,
                                                    s.export_limit_mw),
                "frequency_status": ("low" if s.frequency_hz
                                     < p.grid.frequency_reflex_hz else "nominal"),
            },
            "demand_response": {
                "active": bool(r and r.dr_mw > 0.05),
                "iso_call": s.dr_call_active,
                "dr_mw": _f(r.dr_mw if r else 0.0, 1),
                "contracted_mw": _f(contracted, 0),
                "calls_today": s.dr_calls_today,
                "shift_debt_mwh": _f(s.shift_debt_mwh, 1),
            },
            "storm_alerts": storms,
            "totals": {
                "renewable_available_mw": _f(s.available_renewable_mw, 1),
                "renewable_used_mw": _f(r.renewable_used_mw if r else 0.0, 1),
                "demand_mw": _f(s.total_demand_mw, 1),
                "net_load_mw": _f(s.net_load_mw, 1),
                "curtailed_mw": _f(r.total_curtailed_mw if r else 0.0, 1),
                "unserved_mw": _f(r.unserved_mw if r else 0.0, 2),
                "fleet_soc_pct": _f(self._fleet_soc_pct(s.battery_soc_pct), 1),
            },
        }

    @staticmethod
    def _limit_status(flow: float, limit: float) -> str:
        if limit <= 0:
            return "idle"
        ratio = flow / limit
        if ratio >= 0.995:
            return "at_limit"
        if ratio >= 0.9:
            return "near_limit"
        return "normal" if flow > 0.05 else "idle"

    def _forecast_block(self, plan: Plan) -> dict[str, Any]:
        o = self.config.objectives
        steps = o.steps
        band: dict[str, Any] = {}
        env, scen = plan.envelope, plan.scenarios
        for name, series in (("solar", "solar_mw"), ("wind", "wind_mw"),
                             ("demand", "demand_mw"), ("price", "price_per_mwh")):
            arr = getattr(env, name, None) if env is not None else None
            if arr is not None:
                band[name] = {"p10": [_f(x, 1) for x in arr[0]],
                              "p50": [_f(x, 1) for x in arr[1]],
                              "p90": [_f(x, 1) for x in arr[2]],
                              "source": "forecaster"}
                continue
            # No trained forecaster: the fan the optimizer actually scored is the
            # honest band to show, rather than an invented one.
            if scen is None:
                band[name] = {"p10": [0.0] * steps, "p50": [0.0] * steps,
                              "p90": [0.0] * steps, "source": "unavailable"}
                continue
            mat = scen.solar_mw if name == "solar" else (
                sum(scen.wind_mw.values()) if name == "wind"
                else getattr(scen, series))
            mat = np.atleast_2d(np.asarray(mat, dtype=float))
            band[name] = {
                "p10": [_f(x, 1) for x in np.percentile(mat, 10, axis=0)],
                "p50": [_f(x, 1) for x in np.percentile(mat, 50, axis=0)],
                "p90": [_f(x, 1) for x in np.percentile(mat, 90, axis=0)],
                "source": "scenario_fan",
            }
        projected = [_f(self._fleet_soc_pct(rec.battery_soc_pct), 1)
                     for rec in (plan.decision.trace if plan.decision is not None else ())]
        projected = (projected + [projected[-1] if projected else 0.0] * steps)[:steps]
        return {
            "steps": steps,
            "horizon_hours": o.lookahead_hours,
            "scenarios": plan.scenarios.n_scenarios if plan.scenarios else 0,
            "projected_fleet_soc_pct": projected,
            **band,
        }

    def _trace_block(self, plan: Plan) -> dict[str, Any]:
        d = plan.decision
        if d is None:
            return {
                "situation": {"regimes": [], "text": "Decision engine unavailable."},
                "criteria": {}, "options": {"clusters_evaluated": 0, "surviving": 0,
                                            "guardrails": [], "top5": [],
                                            "pareto_front": [], "scatter": []},
                "committed": {}, "tradeoff": {},
                "explanation": plan.fallback_reason or "No decision trace.",
                "explanation_source": "template",
            }
        c = d.criteria
        r = d.committed
        alt = d.material_alternative()
        return {
            "situation": {
                "regimes": [x.as_dict() for x in c.regimes],
                "text": situation_text(d),
            },
            "criteria": {
                "base_weights": {k: _f(v, 4) for k, v in c.base_weights.items()},
                "final_weights": {k: _f(v, 4) for k, v in c.final_weights.items()},
                "multiplier_applied": {k: _f(v, 4)
                                       for k, v in c.multiplier_applied.items()},
                "shifted": {k: [_f(b, 4), _f(f, 4)] for k, (b, f) in c.shifted().items()},
                "risk_lambda": _f(c.risk_lambda, 3),
                "base_risk_lambda": _f(c.base_risk_lambda, 3),
                "text": criteria_text(d),
            },
            "options": {
                "clusters_evaluated": d.clusters_evaluated,
                "scenarios": d.scenarios_evaluated,
                "steps": d.steps_evaluated,
                "guardrails": [g.as_dict() for g in d.guardrails],
                "guardrails_removed": d.guardrails_removed,
                "surviving": len(d.surviving),
                "top5": [s.as_dict() for s in d.top5],
                "pareto_front": list(d.pareto_front),
                "scatter": [{
                    "index": s.index, "label": s.cluster.short_label,
                    "score": _f(s.score, 6),
                    "net_cost_usd": _f(s.objectives["net_cost"], 0),
                    "reliability_risk_mwh": _f(s.objectives["reliability_risk"], 3),
                    "carbon_tonnes": _f(s.objectives["carbon"], 2),
                    "survived_guardrails": s.survived_guardrails,
                    "on_pareto_front": s.on_pareto_front,
                } for s in d.scores],
                "text": options_text(d),
            },
            "committed": {
                "index": d.chosen_index,
                "cluster": d.chosen.as_dict(),
                "held_by_hysteresis": d.held_by_hysteresis,
                "objectives": {k: _f(v, 3) for k, v in d.chosen_score.objectives.items()},
                "contributions": {k: _f(v, 5)
                                  for k, v in d.chosen_score.contributions.items()},
                "cost_p10_usd": _f(d.chosen_score.cost_p10, 0),
                "cost_p50_usd": _f(d.chosen_score.cost_p50, 0),
                "cost_p90_usd": _f(d.chosen_score.cost_p90, 0),
                "interval": {
                    "battery_mw": {k: _f(v, 1) for k, v in r.battery_mw.items()},
                    "battery_soc_pct": {k: _f(v, 1) for k, v in r.battery_soc_pct.items()},
                    "import_mw": _f(r.import_mw, 1), "export_mw": _f(r.export_mw, 1),
                    "curtailed_mw": _f(r.total_curtailed_mw, 1),
                    "dr_mw": _f(r.dr_mw, 1), "shift_mw": _f(r.shift_mw, 1),
                    "unserved_mw": _f(r.unserved_mw, 2),
                    "price_per_mwh": _f(r.price_per_mwh, 2),
                    "frequency_hz": _f(r.frequency_hz, 3),
                    "forced_injection_mw": _f(r.forced_injection_mw, 1),
                    "reserve_shortfall_mw": _f(r.reserve_shortfall_mw, 2),
                },
            },
            "tradeoff": {
                "alternative": None if alt is None else {
                    "index": alt.index, "label": alt.cluster.short_label,
                    "survived_guardrails": alt.survived_guardrails,
                    "objectives": {k: _f(v, 3) for k, v in alt.objectives.items()},
                },
                "objective_deltas": {k: _f(v, 3) for k, v in d.objective_deltas().items()},
                "deciding_objective": d.deciding_objective(),
                "text": tradeoff_text(d),
            },
            "explanation": d.explanation,
            "explanation_source": d.explanation_source,
        }

    def _actions_block(self, plan: Plan) -> list[dict[str, Any]]:
        if plan.gate is None:
            return []
        return [g.as_dict() for g in plan.gate.gated]

    def _agent_block(self, plan: Plan) -> dict[str, Any]:
        o = plan.outcome
        return {
            "llm_available": bool(o.llm_available) if o else self.llm.available,
            "unavailable_reason": self.llm.unavailable_reason,
            "model": self.llm.model,
            "iterations": o.iterations if o else 0,
            "tool_calls": list(o.tool_calls) if o else [],
            "fallback_reason": plan.fallback_reason,
            "usage": self.llm.usage.as_dict(),
        }

    def _maintenance_block(self, plan: Plan) -> dict[str, Any]:
        mp = plan.maintenance
        return {
            "windows": [] if mp is None else [v.as_dict() for v in mp.valuations],
            "notes": [] if mp is None else list(mp.notes),
            "proposed_delays": [p for p in self.maintenance_proposals],
            "work_orders": [o for o in self.work_orders],
            "scheduled": [{
                "id": m.id, "asset_id": m.asset_id, "status": m.status,
                "start_interval": m.start_interval,
                "end_interval": m.end_interval,
                "deadline_interval": m.deadline_interval, "crew": m.crew,
            } for m in plan.snapshot.maintenance],
        }

    def _quality_block(self, plan: Plan) -> dict[str, Any]:
        q = plan.snapshot.quality
        return {
            "stale": list(q.stale_fields),
            "substituted": list(q.substituted_fields),
            "intervals_stale": dict(q.intervals_stale),
            "degraded": q.degraded,
            "alarm": self.normalizer.alarm,
            "limit_intervals": STALE_ALARM_INTERVALS,
        }

    def _alarms(self, plan: Plan) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        q = plan.snapshot.quality
        if q.degraded:
            out.append({
                "kind": "data_quality",
                "severity": "critical" if self.normalizer.alarm else "warning",
                "message": (
                    f"{', '.join(q.substituted_fields) or 'no channel'} replaced by the "
                    f"forecast; stale: {', '.join(q.stale_fields) or 'none'}"),
            })
        if plan.mode != MODE_AGENT:
            out.append({"kind": "fallback", "severity": "critical",
                        "message": plan.fallback_reason or "running in fallback"})
        r = plan.decision.committed if plan.decision is not None else None
        if r is not None and r.unserved_mw > 0.01:
            out.append({"kind": "unserved", "severity": "critical",
                        "message": f"{r.unserved_mw:.2f} MW of load unserved"})
        if r is not None and r.reserve_shortfall_mw > 0.01:
            out.append({"kind": "reserve", "severity": "warning",
                        "message": (f"fast reserve short by "
                                    f"{r.reserve_shortfall_mw:.1f} MW")})
        o = plan.outcome
        if o is not None and not o.llm_available:
            out.append({"kind": "llm", "severity": "info",
                        "message": (f"Orchestrator narration unavailable "
                                    f"({self.llm.unavailable_reason}); the optimizer "
                                    f"is running the loop with template explanations")})
        if o is not None and o.verification is not None and not o.verification.ok:
            out.append({"kind": "verification", "severity": "warning",
                        "message": (f"explanation failed number verification: "
                                    f"{o.verification.reason}")})
        for req in self.gateway.pending:
            out.append({"kind": "approval", "severity": "warning",
                        "message": f"awaiting approval: {req.set_point.text}"})
        return out


def jsonable(value: Any) -> Any:
    """Frames go out over JSON, so numpy and dataclasses must not survive."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "as_dict"):
        return jsonable(value.as_dict())
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return jsonable(asdict(value))
    return str(value)
