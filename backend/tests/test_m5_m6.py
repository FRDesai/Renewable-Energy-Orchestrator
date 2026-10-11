"""M5: maintenance, gateway, audit/replay. M6: the invariant suite."""
import json
import sys

import numpy as np
import pytest

from orchestrator.act.gateway import ActuationGateway, Verdict
from orchestrator.act.rule_based import fallback, no_storage_cluster, rule_based_cluster
from orchestrator.audit.store import AuditLog, decision_record
from orchestrator.decide.optimizer import Optimizer
from orchestrator.domain.actions import CommittedActions, SetPoint
from orchestrator.domain.events import EventType, MaintenanceWindow
from orchestrator.forecast.quantile import QuantileForecaster
from orchestrator.invariants.checks import CHECKS, InvariantSuite
from orchestrator.maintenance.planner import MaintenancePlanner
from orchestrator.sim.episode import EpisodeRunner
from orchestrator.world.world import World
from orchestrator.ingest.normalize import Normalizer


@pytest.fixture(scope="module")
def forecaster():
    f = QuantileForecaster.load()
    if f is None or not f.is_trained:
        pytest.skip("run `make train` first")
    return f


@pytest.fixture(scope="module")
def runner(config, forecaster):
    return EpisodeRunner(config, forecaster)


def snap_at(config, preset, interval, seed=5, inject=None):
    w = World(config, seed=seed, preset=preset)
    n = Normalizer(config)
    s = None
    for i in range(interval + 1):
        if inject and i == interval:
            w.inject(inject[0], **inject[1])
        s = n.build(w.telemetry())
        if i < interval:
            w.step()
    return s


# ---- M5: maintenance -----------------------------------------------------
def _paths(config, snapshot, output_mw):
    T = config.objectives.steps
    price = np.linspace(20.0, 200.0, T)       # rising price: later is costlier to lose
    return price, {snapshot.maintenance[0].asset_id: np.full(T, output_mw)}


def test_delay_only_when_the_saving_clears_the_prd_bar(config):
    planner = MaintenancePlanner(config)
    bar = config.objectives.maintenance["min_saving_to_delay_usd"]
    assert bar == 2500.0

    snap = snap_at(config, "typical_day", 40)
    assert snap.maintenance, "the world should seed a planned window"
    # High output -> moving the window saves a lot -> recommend the delay.
    price, outputs = _paths(config, snap, 100.0)
    plan = planner.plan(snap, price_path=price, asset_output=outputs)
    assert plan.valuations
    v = plan.valuations[0]
    if v.saving_usd > bar:
        assert v.recommend_delay and "above the" in v.reason
    # Tiny output -> no material saving -> keep the window.
    price, outputs = _paths(config, snap, 0.2)
    v2 = planner.plan(snap, price_path=price, asset_output=outputs).valuations[0]
    assert not v2.recommend_delay
    assert "below the" in v2.reason
    assert v2.saving_usd < bar


def test_storm_overlap_forces_a_move_regardless_of_economics(config):
    """Crew safety is not a trade-off."""
    planner = MaintenancePlanner(config)
    snap = snap_at(config, "typical_day", 48,
                   inject=(EventType.STORM_ALERT, dict(duration=20, lead=4)))
    # Place the window squarely inside the storm.
    storm = snap.storm_alert
    window = MaintenanceWindow(
        id="mw-x", asset_id=config.portfolio.wind_farms[0].id,
        start_interval=storm.arrival_interval, duration_intervals=4,
        deadline_interval=storm.arrival_interval + 40, crew="crew-a")
    import dataclasses
    snap = dataclasses.replace(snap, maintenance=(window,))
    price, outputs = _paths(config, snap, 0.01)     # economics say "don't bother"
    v = planner.plan(snap, price_path=price, asset_output=outputs).valuations[0]
    assert v.storm_overlap and v.recommend_delay
    assert "crew safety" in v.reason.lower()
    assert v.best_start != v.current_start


def test_inspection_dispatched_after_a_battery_trip(config):
    planner = MaintenancePlanner(config)
    snap = snap_at(config, "typical_day", 40,
                   inject=(EventType.BATTERY_TRIP, dict(assets=("harbor",), duration=10)))
    orders = planner.inspections_for(snap)
    assert any(o.asset_id == "harbor" and "trip" in o.reason.lower() for o in orders)


def test_inspection_dispatched_after_a_turbine_cut_out(config):
    planner = MaintenancePlanner(config)
    snap = snap_at(config, "storm_passage", 72)
    curve = config.portfolio.wind_curve
    cut_out = [f.id for f in config.portfolio.wind_farms
               if snap.wind_speed_ms[f.id] >= curve.cut_out_ms]
    orders = planner.inspections_for(snap)
    for fid in cut_out:
        assert any(o.asset_id == fid for o in orders), f"no inspection for {fid}"


# ---- M5: gateway ---------------------------------------------------------
def actions_of(*set_points):
    from orchestrator.domain.actions import CLUSTERS
    return CommittedActions(interval=40, cluster=CLUSTERS[0], set_points=tuple(set_points))


def test_routine_set_points_execute_and_are_signed(config):
    gw = ActuationGateway(config)
    b = config.portfolio.batteries[0]
    res = gw.submit(actions_of(
        SetPoint("discharge", b.id, b.name, 50.0, "arbitrage")))
    assert [g.verdict for g in res.gated] == [Verdict.EXECUTED]
    assert len(res.gated[0].signature) == 16
    assert not res.any_queued


def test_signature_is_stable_and_instruction_specific(config):
    gw = ActuationGateway(config)
    b = config.portfolio.batteries[0]
    a = gw.submit(actions_of(SetPoint("discharge", b.id, b.name, 50.0, "x"))).gated[0]
    again = gw.submit(actions_of(SetPoint("discharge", b.id, b.name, 50.0, "y"))).gated[0]
    other = gw.submit(actions_of(SetPoint("discharge", b.id, b.name, 51.0, "x"))).gated[0]
    assert a.signature == again.signature, "signature must not depend on prose"
    assert a.signature != other.signature, "a different MW must sign differently"


def test_out_of_bounds_set_points_are_rejected_not_clamped(config):
    gw = ActuationGateway(config)
    b = config.portfolio.batteries[0]
    res = gw.submit(actions_of(
        SetPoint("discharge", b.id, b.name, b.power_mw + 25.0, "too much")))
    assert res.gated[0].verdict is Verdict.REJECTED
    assert "exceeds" in res.gated[0].reason
    assert not res.executed


def test_critical_consumers_cannot_be_curtailed_by_the_agent(config):
    gw = ActuationGateway(config)
    critical = next(c for c in config.portfolio.consumers if c.critical)
    res = gw.submit(actions_of(
        SetPoint("reduce", critical.id, critical.name, 5.0, "peak")))
    assert res.gated[0].verdict is Verdict.REJECTED
    assert "critical" in res.gated[0].reason.lower()


def test_high_impact_actions_queue_for_one_click_approval(config):
    gw = ActuationGateway(config)
    res = gw.submit(actions_of(
        SetPoint("shed", "load", "Unserved load", 10.0, "unavoidable",
                 requires_approval=True)))
    assert res.gated[0].verdict is Verdict.QUEUED
    assert len(gw.pending) == 1
    req = gw.pending[0]
    gw.decide(req.id, approve=True, by="operator-1")
    assert not gw.pending
    assert gw.overrides[-1]["approved"] is True
    assert gw.overrides[-1]["by"] == "operator-1"


def test_rejections_are_logged_as_preference_data(config):
    """PRD Trainability: "rejections are logged as preference data"."""
    gw = ActuationGateway(config)
    res = gw.submit(actions_of(
        SetPoint("reduce", "sinclair_chem", "Sinclair Chemicals", 12.0, "DR",
                 requires_approval=True)))
    gw.decide(res.queued[0].id, approve=False, by="operator-2")
    entry = gw.overrides[-1]
    assert entry["kind"] == "approval" and entry["approved"] is False
    assert entry["asset"] == "sinclair_chem"


def test_operator_override_always_wins_and_is_logged(config):
    gw = ActuationGateway(config)
    gw.override(41, "held Harbor manually for the evening peak", by="operator-3")
    assert gw.overrides[-1]["kind"] == "override"
    assert gw.overrides[-1]["by"] == "operator-3"


def test_batch_mode_auto_approves_and_says_so(config):
    """DECISIONS.md D9: Monte Carlo cannot wait for a human, and the trace must
    distinguish a simulated run from a live one."""
    gw = ActuationGateway(config, auto_approve=True)
    res = gw.submit(actions_of(
        SetPoint("shed", "load", "Unserved load", 10.0, "x", requires_approval=True)))
    assert res.gated[0].verdict is Verdict.EXECUTED
    assert "auto-approved in batch mode" in res.gated[0].reason
    assert not gw.pending


# ---- M5: audit and replay ------------------------------------------------
def test_decision_record_is_json_serialisable_and_complete(config, forecaster):
    snap = snap_at(config, "storm_passage", 48)
    opt = Optimizer(config)
    d = opt.decide(snap, seed=3, envelope=forecaster.envelope(snap, config.objectives.steps))
    actions = opt.set_points(d)
    gw = ActuationGateway(config)
    body = decision_record(d, actions, gw.submit(actions),
                           preset="storm_passage", world_seed=5, config_version=1)
    log = AuditLog(keep_in_memory=True)
    entry = log.append("decision", d.interval, body)
    parsed = json.loads(entry.to_json())
    # PRD Observability: input snapshot, seed, forecasts, regimes, weights, guardrail
    # removals, top 5 clusters, actions, tool calls -- all present.
    b = parsed["body"]
    for key in ("state", "seed", "criteria", "guardrails", "surviving", "top5",
                "actions", "committed", "chosen_index", "tool_calls", "pareto_front"):
        assert key in b, f"{key} missing from the audit record"
    assert len(b["top5"]) <= 5 and b["top5"]
    assert b["criteria"]["base_weights"] and b["criteria"]["final_weights"]
    assert len(b["guardrails"]) == 3


def test_audit_log_is_append_only(config, forecaster, tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("event", 1, {"a": 1})
    log.append("decision", 2, {"b": 2})
    assert len(log) == 2
    lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    log.append("event", 3, {"c": 3})
    lines2 = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
    assert lines2[:2] == lines, "earlier entries were rewritten"


def test_replay_reproduces_the_decision_bit_for_bit(config, runner):
    """PRD: "Every decision stored with its inputs snapshot, seed ... replayable
    bit-for-bit"."""
    kw = dict(strategy="agent", preset="storm_passage", seed=777,
              start_interval=40, intervals=10, keep_audit=True)
    first = runner.run(**kw)
    key = first.audit.replay_key(40)
    assert key is not None and key["seed"] is not None

    again = runner.run(**kw)
    assert first.chosen_indices == again.chosen_indices
    # `latency_ms` is wall-clock measurement, not an input or an output of the
    # decision, so it is excluded: everything that determines or describes the
    # decision itself must be identical.
    def comparable(entry):
        body = json.loads(entry.to_json())
        body["body"].pop("latency_ms", None)
        return body

    assert len(first.audit.entries) == len(again.audit.entries) == 10
    for a, b in zip(first.audit.entries, again.audit.entries):
        assert comparable(a) == comparable(b), \
            f"interval {a.interval} did not replay identically"


REPLAY_PROBE = """
import sys
sys.path.insert(0, %r)
from orchestrator.config_loader import load_config
from orchestrator.sim.episode import EpisodeRunner
r = EpisodeRunner(load_config()).run(
    strategy="agent", preset="evening_price_crisis", seed=2024,
    start_interval=0, intervals=10, check_invariants=False)
print(r.chosen_indices)
print([round(float(x), 9) for x in r.price_path])
print(round(float(r.kpis.energy_cost_usd), 9))
"""


def test_replay_is_reproducible_across_processes():
    """"Bit-for-bit" has to mean *across runs*, not merely twice inside one process.

    Python salts string hashing per process, so anything that derives a physical
    quantity from `hash()` silently makes every benchmark figure depend on which
    process produced it. The determinism invariant compares two runs in one
    interpreter and cannot see that, which is why this runs real subprocesses under
    different hash seeds.
    """
    import os
    import subprocess
    from pathlib import Path

    root = str(Path(__file__).resolve().parent.parent)
    outputs = []
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        proc = subprocess.run([sys.executable, "-c", REPLAY_PROBE % root],
                              capture_output=True, text=True, env=env, cwd=root)
        assert proc.returncode == 0, proc.stderr[-2000:]
        outputs.append(proc.stdout)

    assert outputs[0] == outputs[1] == outputs[2], (
        "the same preset and seed produced different physics in different processes:\n"
        + "\n---\n".join(outputs))


def test_fallback_controller_is_rule_based_with_reserve_held(config):
    snap = snap_at(config, "typical_day", 48)
    f = fallback(config, snap, reason="solver error")
    assert f.reason == "solver error"
    assert f.reserve_held_pct == config.portfolio.battery_limits.reserve_floor_pct
    assert f.cluster.index == rule_based_cluster(config, snap).index


def test_rule_based_baseline_follows_the_fixed_rule(config):
    surplus = snap_at(config, "typical_day", 48)      # midday surplus
    deficit = snap_at(config, "typical_day", 80)      # evening deficit
    assert surplus.net_load_mw < 0 < deficit.net_load_mw
    assert rule_based_cluster(config, surplus).battery.value == "soak_surplus"
    assert rule_based_cluster(config, deficit).battery.value == "discharge_now"
    # The no-storage baseline must not cycle the fleet.
    assert no_storage_cluster(config, deficit).battery.value == "hold_for_higher_price"


# ---- M6: the invariant suite --------------------------------------------
def test_suite_covers_the_prd_invariant_list():
    labels = {label for label, _, _ in CHECKS.values()}
    for required in ("Energy balance", "SoC limits", "Power ratings",
                     "Transmission limits", "No wash trading", "Justified curtailment",
                     "Unavoidable shedding only", "Frequency response",
                     "Storm readiness", "Latency"):
        assert required in labels, f"{required} is not in the suite"
    report = InvariantSuite().report()
    assert any(r["label"] == "Determinism" for r in report)
    for row in report:
        assert row["pass_condition"], f"{row['label']} has no stated pass condition"


def test_every_invariant_holds_over_a_short_episode(config, runner):
    res = runner.run(strategy="agent", preset="volatile_weather", seed=31,
                     start_interval=40, intervals=24)
    for row in res.suite.report():
        assert row["failed"] == 0, (
            f"{row['label']}: {row['failed']}/{row['applicable']} failed, "
            f"e.g. {row['failures'][:1]}"
        )


def test_invariants_actually_fire_rather_than_being_vacuous(config, runner):
    """A suite that never evaluates anything proves nothing."""
    res = runner.run(strategy="agent", preset="contingency_cascade", seed=12,
                     start_interval=30, intervals=32)
    applicable = {r["key"]: r["applicable"] for r in res.suite.report()}
    for key in ("energy_balance", "soc_limits", "power_ratings",
                "transmission_limits", "no_wash_trading", "latency"):
        assert applicable[key] == 32, f"{key} evaluated {applicable[key]}/32 intervals"


def test_rule_based_baseline_sheds_load_where_the_agent_does_not(config, runner):
    """The headline claim. If the baseline never sheds, the comparison is empty."""
    agent_shed = rule_shed = 0.0
    for preset, start, seed in (("evening_price_crisis", 56, 3),
                                ("storm_passage", 60, 4),
                                ("contingency_cascade", 56, 5)):
        kw = dict(preset=preset, seed=seed, start_interval=start, intervals=32,
                  check_invariants=False)
        agent_shed += runner.run(strategy="agent", **kw).kpis.unserved_mwh
        rule_shed += runner.run(strategy="rule_based", **kw).kpis.unserved_mwh
    assert rule_shed > 0, "rule-based never shed, so there is nothing to beat"
    assert agent_shed < rule_shed * 0.25, (
        f"agent shed {agent_shed:.2f} MWh against rule-based {rule_shed:.2f} MWh -- "
        f"below the PRD's -75% target"
    )


def test_agent_never_sheds_avoidably(config, runner):
    """PRD's first metric: avoidable load shedding target 0."""
    for preset, start in (("evening_price_crisis", 56), ("storm_passage", 60)):
        res = runner.run(strategy="agent", preset=preset, seed=9,
                         start_interval=start, intervals=32, check_invariants=False)
        assert res.kpis.avoidable_unserved_mwh == pytest.approx(0.0, abs=1e-6)


def test_baselines_see_identical_disturbances(config, runner):
    """A benchmark is only fair if every strategy faces the same world."""
    plan = [(42, EventType.PRICE_SPIKE, dict(magnitude=5.0, duration=6)),
            (46, EventType.CLOUD_BANK, dict(region="region_a", magnitude=0.9, duration=6))]
    kw = dict(preset="typical_day", seed=21, start_interval=40, intervals=20,
              check_invariants=False, event_plan=plan)
    a = runner.run(strategy="agent", **kw)
    r = runner.run(strategy="rule_based", **kw)
    assert a.price_path[:3] == pytest.approx(r.price_path[:3]), \
        "strategies saw different prices at the start, so the world diverged"
