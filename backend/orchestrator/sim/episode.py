"""One episode: drive the world through the full decide-act loop, checking invariants.

Used by the nightly reliability run and by the Monte Carlo simulator, so both measure
exactly the loop the live console runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..act.gateway import ActuationGateway
from ..act.rule_based import no_storage_cluster, rule_based_cluster
from ..audit.store import AuditLog, decision_record
from ..config_loader import Config
from ..decide.dispatch import DispatchEngine
from ..decide.optimizer import Decision, Optimizer
from ..explain.template import explain
from ..forecast.quantile import QuantileForecaster
from ..forecast.scenarios import build_scenarios
from ..ingest.normalize import Normalizer
from ..invariants.checks import IntervalContext, InvariantSuite
from ..world.world import AppliedDispatch, World

STRATEGIES = ("agent", "rule_based", "no_storage")


@dataclass
class EpisodeKPIs:
    energy_cost_usd: float = 0.0
    carbon_tonnes: float = 0.0
    unserved_mwh: float = 0.0
    avoidable_unserved_mwh: float = 0.0
    curtailed_mwh: float = 0.0
    renewable_available_mwh: float = 0.0
    renewable_used_mwh: float = 0.0
    throughput_mwh: float = 0.0
    wear_usd: float = 0.0
    reserve_ok_intervals: int = 0
    intervals: int = 0
    dr_mwh: float = 0.0
    shift_mwh: float = 0.0
    start_soc_mwh: float = 0.0
    end_soc_mwh: float = 0.0
    mean_price_per_mwh: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def stored_energy_delta_mwh(self) -> float:
        return self.end_soc_mwh - self.start_soc_mwh

    @property
    def net_cost_usd(self) -> float:
        """The PRD's *objective* definition of net cost, which credits the change in
        the value of stored energy. The headline `energy_cost_usd` follows the PRD's
        *KPI* definition instead (cash only) -- the two differ at an episode boundary
        where the agent is deliberately holding charge it has already paid for."""
        return self.energy_cost_usd - self.stored_energy_delta_mwh * \
            self.mean_price_per_mwh * 0.93

    @property
    def renewable_utilisation(self) -> float:
        if self.renewable_available_mwh <= 0:
            return 1.0
        return self.renewable_used_mwh / self.renewable_available_mwh

    @property
    def reserve_compliance(self) -> float:
        return self.reserve_ok_intervals / self.intervals if self.intervals else 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "energy_cost_usd": round(self.energy_cost_usd, 1),
            "carbon_tonnes": round(self.carbon_tonnes, 2),
            "unserved_mwh": round(self.unserved_mwh, 4),
            "avoidable_unserved_mwh": round(self.avoidable_unserved_mwh, 4),
            "curtailed_mwh": round(self.curtailed_mwh, 3),
            "renewable_utilisation": round(self.renewable_utilisation, 4),
            "reserve_compliance": round(self.reserve_compliance, 4),
            "throughput_mwh": round(self.throughput_mwh, 2),
            "wear_usd": round(self.wear_usd, 1),
            "dr_mwh": round(self.dr_mwh, 2),
            "shift_mwh": round(self.shift_mwh, 2),
            "net_cost_usd": round(self.net_cost_usd, 1),
            "stored_energy_delta_mwh": round(self.stored_energy_delta_mwh, 2),
            "intervals": self.intervals,
            "latency_p95_ms": round(
                float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0, 2),
        }


@dataclass
class EpisodeResult:
    strategy: str
    preset: str
    seed: int
    kpis: EpisodeKPIs
    suite: InvariantSuite | None = None
    chosen_indices: list[int] = field(default_factory=list)
    soc_path: list[float] = field(default_factory=list)
    cost_path: list[float] = field(default_factory=list)
    price_path: list[float] = field(default_factory=list)
    unserved_path: list[float] = field(default_factory=list)
    start_interval: int = 0
    event_plan: tuple = ()        # part of the replay key: preset + seed alone is not
    audit: AuditLog | None = None


class EpisodeRunner:
    def __init__(self, config: Config, forecaster: QuantileForecaster | None = None) -> None:
        self.config = config
        self.optimizer = Optimizer(config)
        self.engine = DispatchEngine(config)
        self.forecaster = forecaster

    def run(
        self,
        *,
        strategy: str = "agent",
        preset: str = "typical_day",
        seed: int = 0,
        start_interval: int = 0,
        intervals: int = 96,
        check_invariants: bool = True,
        keep_audit: bool = False,
        decision_seed: int | None = None,
        event_plan: list[tuple[int, object, dict]] | None = None,
    ) -> EpisodeResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        cfg = self.config
        p = cfg.portfolio
        h = p.interval_hours
        world = World(cfg, seed=seed, preset=preset, start_interval=start_interval)
        normalizer = Normalizer(cfg, substitute=self._substitute_factory())
        gateway = ActuationGateway(cfg, auto_approve=True)   # batch mode (D9)
        suite = InvariantSuite() if check_invariants else None
        audit = AuditLog(keep_in_memory=True) if keep_audit else None
        kpis = EpisodeKPIs()
        result = EpisodeResult(strategy, preset, seed, kpis, suite,
                               start_interval=start_interval,
                               event_plan=tuple(event_plan or ()), audit=audit)

        pending = sorted(event_plan or [], key=lambda x: x[0])
        previous: int | None = None
        cumulative_cost = 0.0
        prices: list[float] = []
        kpis.start_soc_mwh = sum(
            b.energy_mwh * world.battery_soc_pct[b.id] / 100.0 for b in p.batteries)
        for step in range(intervals):
            # Inject any scheduled events before this interval is observed, so every
            # strategy in a benchmark sees exactly the same disturbances.
            while pending and pending[0][0] <= world.interval:
                _, kind, spec = pending.pop(0)
                world.inject(kind, **spec)
            snapshot = normalizer.build(world.telemetry())
            self._latest_snapshot = snapshot
            d_seed = (decision_seed if decision_seed is not None
                      else seed * 1000 + snapshot.interval)

            envelope = (self.forecaster.envelope(snapshot, cfg.objectives.steps)
                        if self.forecaster is not None and self.forecaster.is_trained
                        else None)
            scen = build_scenarios(cfg, snapshot, seed=d_seed, envelope=envelope)

            if strategy == "agent":
                decision = self.optimizer.decide(
                    snapshot, seed=d_seed, scenarios=scen,
                    previous_cluster_index=previous)
            else:
                cluster = (rule_based_cluster(cfg, snapshot) if strategy == "rule_based"
                           else no_storage_cluster(cfg, snapshot))
                decision = self._forced_decision(snapshot, scen, cluster.index, d_seed)
            previous = decision.chosen_index

            record = decision.committed
            actions = self.optimizer.set_points(decision)
            gate = gateway.submit(actions)

            if suite is not None:
                storm = snapshot.storm_alert
                online = {b.id: b.energy_mwh for b in p.batteries
                          if snapshot.battery_available.get(b.id, True)}
                fleet = snapshot.fleet_soc_pct(online) if online else 0.0
                suite.check(IntervalContext(
                    config=cfg, snapshot=snapshot, record=record,
                    latency_ms=decision.latency_ms,
                    storm_on_site=(storm is not None
                                   and storm.on_site_at(snapshot.interval)),
                    fleet_soc_pct=fleet,
                    available_capacity_fraction=(
                        sum(online.values()) / p.battery_energy_mwh),
                    chosen_index=decision.chosen_index))

            if audit is not None:
                decision.explanation = explain(cfg, decision)
                decision.auto_approved = True
                audit.append("decision", snapshot.interval, decision_record(
                    decision, actions, gate, preset=preset, world_seed=seed,
                    config_version=p.version))

            # ---- KPIs ------------------------------------------------------
            price = record.price_per_mwh
            cost = (record.import_mw * h * (price + self.engine.spread)
                    - record.export_mw * h * price
                    + record.dr_mw * h * self.engine.dr_pay
                    + max(0.0, record.shift_mw) * h * self.engine.shift_pay)
            kpis.energy_cost_usd += cost
            cumulative_cost += cost
            kpis.carbon_tonnes += record.import_mw * h * self.engine.carbon_intensity
            kpis.unserved_mwh += record.unserved_mw * h
            # PRD metric 1: "MWh shed while import or battery headroom existed".
            # Headroom must be *deliverable*: a battery at the hard floor has plenty of
            # unused power rating but no energy behind it, so counting the rating alone
            # reports phantom avoidable shedding.
            lim = p.battery_limits
            batt_headroom = 0.0
            for b in p.batteries:
                if not snapshot.battery_available.get(b.id, True):
                    continue
                soc_mwh = b.energy_mwh * record.battery_soc_pct.get(b.id, 0.0) / 100.0
                floor_mwh = b.energy_mwh * lim.soc_min_pct / 100.0
                from_energy = max(0.0, soc_mwh - floor_mwh) * b.one_way_efficiency / h
                spare_power = max(0.0, b.power_mw - record.battery_mw.get(b.id, 0.0))
                batt_headroom += min(spare_power, from_energy)
            headroom = (record.import_limit_mw - record.import_mw) + batt_headroom
            if record.unserved_mw > 1e-6 and headroom > 1e-3:
                kpis.avoidable_unserved_mwh += record.unserved_mw * h
            kpis.curtailed_mwh += record.total_curtailed_mw * h
            kpis.renewable_available_mwh += (
                record.solar_available_mw + record.wind_available_mw) * h
            kpis.renewable_used_mwh += record.renewable_used_mw * h
            tp = sum(abs(v) for v in record.battery_mw.values()) * h
            kpis.throughput_mwh += tp
            kpis.wear_usd += sum(
                abs(mw) * h * p.battery(bid).wear_cost_per_mwh
                for bid, mw in record.battery_mw.items())
            if record.reserve_shortfall_mw <= 1e-6:
                kpis.reserve_ok_intervals += 1
            kpis.dr_mwh += record.dr_mw * h
            kpis.shift_mwh += max(0.0, record.shift_mw) * h
            kpis.intervals += 1
            kpis.latencies_ms.append(decision.latency_ms)
            prices.append(price)
            kpis.end_soc_mwh = sum(
                b.energy_mwh * record.battery_soc_pct.get(b.id, 0.0) / 100.0
                for b in p.batteries)

            result.chosen_indices.append(decision.chosen_index)
            result.soc_path.append(
                100.0 * sum(record.battery_soc_pct.values()) / len(record.battery_soc_pct))
            result.cost_path.append(cumulative_cost)
            result.price_path.append(price)
            result.unserved_path.append(record.unserved_mw)

            # ---- advance the world by what the gateway executed -------------
            executed = {sp.asset_id: sp for sp in gate.executed}
            world.apply(AppliedDispatch(
                battery_mw={
                    b.id: (record.battery_mw.get(b.id, 0.0)
                           if b.id in executed else 0.0)
                    for b in p.batteries},
                dr_mw=record.dr_mw if any(
                    sp.verb == "reduce" for sp in gate.executed) else 0.0,
                shift_mw=record.shift_mw,
                grid_mw=record.grid_mw,
                unserved_mwh=record.unserved_mw * h,
            ))
            world.step()

        kpis.mean_price_per_mwh = float(np.mean(prices)) if prices else 0.0
        return result

    # ---- helpers ----------------------------------------------------------
    def _substitute_factory(self):
        self._latest_snapshot = None

        def substitute(channel, interval, last_good):
            if self.forecaster is None or not self.forecaster.is_trained:
                return last_good
            return self.forecaster.substitute(
                self._latest_snapshot, channel, interval, last_good)
        return substitute

    def _forced_decision(self, snapshot, scen, cluster_index: int, seed: int) -> Decision:
        """A baseline runs the same engine as the agent, committing a fixed cluster."""
        return self.optimizer.decide_forced(
            snapshot, cluster_index, seed=seed, scenarios=scen)
