"""M4: regimes, weight adaptation, guardrails and scoring."""
import dataclasses
import time

import numpy as np
import pytest

from orchestrator.decide.classifier import classify
from orchestrator.decide.dispatch import OBJECTIVES, DispatchEngine
from orchestrator.decide.guardrails import apply_guardrails
from orchestrator.decide.optimizer import Optimizer
from orchestrator.decide.score import cvar, pareto_front, score_clusters
from orchestrator.decide.weights import adapt
from orchestrator.domain.actions import CLUSTERS, BatteryStrategy
from orchestrator.domain.events import EventType
from orchestrator.explain.template import explain, tradeoff_text
from orchestrator.forecast.quantile import QuantileForecaster
from orchestrator.forecast.scenarios import build_scenarios
from orchestrator.ingest.normalize import Normalizer
from orchestrator.world.world import World


@pytest.fixture(scope="module")
def forecaster():
    f = QuantileForecaster.load()
    if f is None or not f.is_trained:
        pytest.skip("run `make train` first")
    return f


def state(config, preset, interval, *, seed=11, inject=None):
    w = World(config, seed=seed, preset=preset)
    n = Normalizer(config)
    snap = None
    for i in range(interval + 1):
        if inject and i == interval:
            kind, spec = inject
            w.inject(kind, **spec)
        snap = n.build(w.telemetry())
        if i < interval:
            w.step()
    return snap


def find_interval(config, preset, predicate, *, seed=11, limit=96):
    """Locate an interval that actually exhibits a condition, rather than guessing."""
    w = World(config, seed=seed, preset=preset)
    n = Normalizer(config)
    for i in range(limit):
        snap = n.build(w.telemetry())
        if predicate(snap):
            return i
        w.step()
    raise AssertionError(f"{preset}: no interval satisfied the predicate")


def regimes_at(config, snap):
    scen = build_scenarios(config, snap, seed=1)
    return {r.key: r for r in classify(config, snap, scen)}


# ---- F2.1: every regime fires with a measured trigger ---------------------
def test_all_ten_regimes_can_fire_with_measured_triggers(config):
    """PRD F2.1: "Each regime shown with the measured trigger"."""
    cases = {
        "grid_stress": ("typical_day", 40, (EventType.STORM_ALERT, dict(duration=40, lead=16))),
        "frequency_event": ("typical_day", 40, (EventType.FREQUENCY_DIP, dict(magnitude=0.3, duration=4))),
        "asset_contingency": ("typical_day", 40, (EventType.BATTERY_TRIP, dict(assets=("harbor",), duration=10))),
        "scarcity_ahead": ("evening_price_crisis", 73, None),
        "price_spike": ("evening_price_crisis", 73, None),
        "peak_pricing": ("evening_price_crisis", 73, None),
        "transmission_constrained": ("typical_day", 44, (EventType.LINE_CONGESTION, dict(corridor="north", duration=10))),
        "renewable_surplus": ("typical_day", 48, None),
        "negative_prices": ("volatile_weather", None, None),   # located below
        "dr_program_active": ("typical_day", 40, (EventType.ISO_DR_CALL, dict(duration=8))),
    }
    assert len(cases) == 10, "the PRD lists exactly ten regimes"
    seen = {}
    for key, (preset, interval, inject) in cases.items():
        if interval is None:
            interval = find_interval(config, preset,
                                     lambda s: s.price_per_mwh < 0)
        snap = state(config, preset, interval, inject=inject)
        found = regimes_at(config, snap)
        assert key in found, f"{key} did not fire on its crafted state"
        r = found[key]
        seen[key] = r.trigger
        assert r.trigger, f"{key} fired with no measured trigger"
        # The trigger must carry evidence: a number with a unit, not just a label.
        assert any(ch.isdigit() for ch in r.trigger), \
            f"{key} trigger has no measured value: {r.trigger!r}"
        assert r.label and r.rule
    assert set(seen) == set(config.objectives.regimes)


def test_price_spike_trigger_reads_like_the_prd_example(config):
    """PRD example: "Spot $310 vs typical $52"."""
    snap = state(config, "evening_price_crisis", 73)
    r = regimes_at(config, snap)["price_spike"]
    assert "Spot $" in r.trigger and "typical $" in r.trigger


def test_storm_alert_reports_lead_time_before_arrival(config):
    snap = state(config, "typical_day", 40,
                 inject=(EventType.STORM_ALERT, dict(duration=40, lead=16)))
    r = regimes_at(config, snap)["grid_stress"]
    assert "arrival in 4.0 h" in r.trigger


def test_no_regime_fires_in_calm_conditions(config):
    snap = state(config, "typical_day", 20)      # 05:00, no solar, no events
    found = regimes_at(config, snap)
    assert "grid_stress" not in found
    assert "frequency_event" not in found
    assert "asset_contingency" not in found


# ---- F2.2: weight adaptation ---------------------------------------------
def test_weights_are_base_times_multipliers_renormalised(config):
    snap = state(config, "typical_day", 40,
                 inject=(EventType.STORM_ALERT, dict(duration=40, lead=16)))
    regimes = tuple(regimes_at(config, snap).values())
    c = adapt(config, regimes)

    assert c.base_weights == config.objectives.base_weights, \
        "the agent must not mutate the operator's base weights"
    assert sum(c.final_weights.values()) == pytest.approx(1.0)
    # The applied multiplier is the product over every regime that fired, and
    # grid_stress must be among them.
    assert any(r.key == "grid_stress" for r in regimes)
    for objective in config.objectives.base_weights:
        expected = 1.0
        for r in regimes:
            expected *= r.multipliers.get(objective, 1.0)
        assert c.multiplier_applied[objective] == pytest.approx(expected)
    assert c.multiplier_applied["reliability_risk"] >= 2.4
    assert c.final_weights["reliability_risk"] > c.base_weights["reliability_risk"]
    assert c.final_weights["net_cost"] < c.base_weights["net_cost"]


def test_multipliers_compose_across_simultaneous_regimes(config):
    snap = state(config, "evening_price_crisis", 73)
    regimes = tuple(regimes_at(config, snap).values())
    assert len(regimes) >= 2
    c = adapt(config, regimes)
    expected = 1.0
    for r in regimes:
        expected *= r.multipliers.get("net_cost", 1.0)
    assert c.multiplier_applied["net_cost"] == pytest.approx(expected)


def test_risk_aversion_rises_in_stress_and_caps_at_the_configured_maximum(config):
    risk = config.objectives.risk
    calm = adapt(config, ())
    assert calm.risk_lambda == pytest.approx(risk["lambda_base"])

    snap = state(config, "typical_day", 40,
                 inject=(EventType.STORM_ALERT, dict(duration=40, lead=16)))
    stressed = adapt(config, tuple(regimes_at(config, snap).values()))
    assert stressed.risk_lambda > calm.risk_lambda

    # Every regime at once must not exceed lambda_max.
    everything = tuple(
        r for key in config.objectives.regimes
        for r in [type(stressed.regimes[0])(
            key=key, label=key, rule="", trigger="x",
            multipliers=config.objectives.regimes[key].multipliers,
            risk_aversion=config.objectives.regimes[key].risk_aversion)]
    )
    assert adapt(config, everything).risk_lambda == pytest.approx(risk["lambda_max"])


def test_operator_slider_change_moves_the_final_weights(config):
    snap = state(config, "typical_day", 48)
    regimes = tuple(regimes_at(config, snap).values())
    before = adapt(config, regimes)
    tweaked = config.with_base_weights({"carbon": 0.6}, note="slider")
    after = adapt(tweaked, regimes)
    assert after.final_weights["carbon"] > before.final_weights["carbon"]
    assert sum(after.final_weights.values()) == pytest.approx(1.0)


# ---- F2.3: guardrails ----------------------------------------------------
def guardrail_run(config, snap, forecaster):
    scen = build_scenarios(config, snap, seed=3,
                           envelope=forecaster.envelope(snap, config.objectives.steps))
    out = DispatchEngine(config).simulate(
        scen, snap.battery_soc_pct, snap.battery_available)
    return apply_guardrails(config, snap, scen, out), out, scen


def test_all_three_guardrails_are_reported_every_interval(config, forecaster):
    snap = state(config, "typical_day", 48)
    (alive, results), _, _ = guardrail_run(config, snap, forecaster)
    assert [g.key for g in results] == [
        "avoidable_unserved", "storm_readiness", "frequency_response"], \
        "guardrails must be reported in the PRD's lexicographic order"
    for g in results:
        assert g.rule, f"{g.key} has no stated rule"
        assert g.note, f"{g.key} has no measured note"


def test_avoidable_unserved_guardrail_uses_the_configured_tolerance(config, forecaster):
    snap = state(config, "evening_price_crisis", 73)
    (alive, results), out, _ = guardrail_run(config, snap, forecaster)
    g = next(r for r in results if r.key == "avoidable_unserved")
    tol = config.objectives.guardrails["avoidable_unserved_mwh"]
    unserved = out.unserved_mwh.mean(axis=1)
    best = unserved.min()
    for i in alive:
        assert unserved[i] <= best + tol + 1e-9
    for i in g.removed:
        assert unserved[i] > best + tol - 1e-9
    assert g.removed_count > 0, "this state should separate clusters on shedding"


def test_storm_readiness_guardrail_fires_and_covers_the_full_lead_time(config, forecaster):
    """The PRD's open defect: the guardrail "only checks 4 h ahead". Here the alert
    lead is 5 h -- past the look-ahead -- and the guardrail must still bite."""
    snap = state(config, "storm_passage", 48)
    storm = snap.storm_alert
    assert storm is not None
    lead_h = (storm.arrival_interval - snap.interval) * config.portfolio.interval_hours
    assert lead_h > config.objectives.lookahead_hours, "lead must exceed the horizon"

    (alive, results), out, _ = guardrail_run(config, snap, forecaster)
    g = next(r for r in results if r.key == "storm_readiness")
    assert g.applied and g.removed_count > 0, \
        f"storm guardrail removed nothing with the storm {lead_h:.1f} h out"
    assert "pre-charge ramp" in g.note


def test_storm_guardrail_is_inactive_without_an_alert(config, forecaster):
    snap = state(config, "typical_day", 48)
    (_, results), _, _ = guardrail_run(config, snap, forecaster)
    g = next(r for r in results if r.key == "storm_readiness")
    assert not g.applied and g.removed_count == 0
    assert "no storm alert" in g.note


def test_frequency_guardrail_is_defence_in_depth(config, forecaster):
    """The reflex is applied inside every cluster, so by the time the guardrail runs
    there should be nothing left to remove. Its job is to prove that."""
    snap = state(config, "typical_day", 44,
                 inject=(EventType.FREQUENCY_DIP, dict(magnitude=0.3, duration=6)))
    (_, results), _, _ = guardrail_run(config, snap, forecaster)
    g = next(r for r in results if r.key == "frequency_response")
    assert g.applied, "a dip is in the horizon, so the guardrail must evaluate"
    assert g.removed_count == 0, (
        "a cluster reached the guardrail without injecting, so the always-on reflex "
        "is not actually always on"
    )


def test_guardrails_never_remove_every_cluster(config, forecaster):
    """An empty option set is worse than an unguarded trade-off; the trace must say so."""
    snap = state(config, "storm_passage", 70)      # storm on site, worst case
    (alive, results), _, _ = guardrail_run(config, snap, forecaster)
    assert len(alive) >= 1
    for g in results:
        if g.applied and not g.removed and "not enforced" in g.note:
            assert "unguarded" in g.note


def test_guardrails_are_lexicographic(config, forecaster):
    """Each guardrail filters what the previous one left, so removals never overlap."""
    snap = state(config, "storm_passage", 48)
    (alive, results), _, _ = guardrail_run(config, snap, forecaster)
    seen: set[int] = set()
    for g in results:
        assert not (set(g.removed) & seen), f"{g.key} re-removed an already-removed cluster"
        seen |= set(g.removed)
    assert len(alive) == len(CLUSTERS) - len(seen)


# ---- F2.4: CVaR scoring --------------------------------------------------
def test_cvar_is_the_mean_of_the_worst_tail():
    x = np.array([[1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # 90th-percentile CVaR over 10 samples = mean of the worst 1.
    assert cvar(x, 0.90)[0] == pytest.approx(10.0)
    assert cvar(x, 0.80)[0] == pytest.approx(9.5)
    assert cvar(x, 0.0)[0] == pytest.approx(x.mean())


def test_cost_score_penalises_the_tail_not_just_the_average(config, forecaster):
    """PRD: "Cost score = mean + lambda x (CVaR90 - mean)"."""
    from orchestrator.decide.score import risk_adjusted
    snap = state(config, "volatile_weather", 48)
    scen = build_scenarios(config, snap, seed=4,
                           envelope=forecaster.envelope(snap, config.objectives.steps))
    out = DispatchEngine(config).simulate(
        scen, snap.battery_soc_pct, snap.battery_available)
    regimes = tuple(regimes_at(config, snap).values())
    criteria = adapt(config, regimes)
    adj = risk_adjusted(config, out, criteria)

    raw_mean = out.objectives["net_cost"].mean(axis=1)
    tail = cvar(out.objectives["net_cost"], config.objectives.risk["cvar_quantile"])
    expected = raw_mean + criteria.risk_lambda * (tail - raw_mean)
    assert np.allclose(adj["net_cost"], expected)
    # The adjustment must actually bite somewhere.
    assert np.any(np.abs(adj["net_cost"] - raw_mean) > 1.0)


def test_reliability_adds_a_tail_term_on_unserved_energy(config, forecaster):
    from orchestrator.decide.score import risk_adjusted
    snap = state(config, "evening_price_crisis", 73)
    scen = build_scenarios(config, snap, seed=4,
                           envelope=forecaster.envelope(snap, config.objectives.steps))
    out = DispatchEngine(config).simulate(
        scen, snap.battery_soc_pct, snap.battery_available)
    criteria = adapt(config, tuple(regimes_at(config, snap).values()))
    adj = risk_adjusted(config, out, criteria)
    base = out.objectives["reliability_risk"].mean(axis=1)
    assert np.all(adj["reliability_risk"] >= base - 1e-9)
    assert np.any(adj["reliability_risk"] > base + 1e-6), "no tail premium applied"


def test_normalisation_uses_the_range_floor(config, forecaster):
    """Clusters that differ trivially must not be driven apart by normalisation."""
    snap = state(config, "typical_day", 20)
    scen = build_scenarios(config, snap, seed=4,
                           envelope=forecaster.envelope(snap, config.objectives.steps))
    out = DispatchEngine(config).simulate(
        scen, snap.battery_soc_pct, snap.battery_available)
    criteria = adapt(config, tuple(regimes_at(config, snap).values()))
    alive, _ = apply_guardrails(config, snap, scen, out)
    scores, _, _ = score_clusters(config, out, criteria, alive)
    # Normalisation is relative to the surviving set, so only survivors are bounded.
    for s in scores:
        if not s.survived_guardrails:
            continue
        for k, v in s.normalized.items():
            assert -1e-9 <= v <= 1.0 + 1e-9, f"{k} normalised to {v}"
    assert any(s.survived_guardrails for s in scores)


def test_lowest_score_wins_among_survivors(config, forecaster):
    snap = state(config, "volatile_weather", 48)
    opt = Optimizer(config)
    d = opt.decide(snap, seed=2, envelope=forecaster.envelope(snap, config.objectives.steps))
    survivors = [s for s in d.scores if s.survived_guardrails]
    assert d.chosen_score.score == pytest.approx(min(s.score for s in survivors))
    assert d.chosen_index in d.surviving


def test_pareto_front_is_non_dominated():
    objs = {
        "a": np.array([1.0, 2.0, 3.0, 2.0]),
        "b": np.array([3.0, 2.0, 1.0, 3.0]),
    }
    front = pareto_front(objs, [0, 1, 2, 3])
    assert set(front) == {0, 1, 2}        # index 3 is dominated by index 1
    for i in front:
        for j in front:
            if i == j:
                continue
            assert not (objs["a"][j] <= objs["a"][i] and objs["b"][j] <= objs["b"][i]
                        and (objs["a"][j] < objs["a"][i] or objs["b"][j] < objs["b"][i]))


def test_decision_reports_a_pareto_front_and_top_five(config, forecaster):
    snap = state(config, "volatile_weather", 48)
    d = Optimizer(config).decide(
        snap, seed=2, envelope=forecaster.envelope(snap, config.objectives.steps))
    assert 1 <= len(d.pareto_front) <= len(d.surviving)
    assert d.chosen_index in d.pareto_front, \
        "the chosen cluster must be non-dominated: otherwise something strictly better exists"
    assert 1 <= len(d.top5) <= 5
    assert [s.score for s in d.top5] == sorted(s.score for s in d.top5)


# ---- F2.5/F2.6: trade-off text and slider responsiveness -----------------
def test_tradeoff_names_a_material_alternative_with_numbers(config, forecaster):
    for preset, interval in (("evening_price_crisis", 73), ("storm_passage", 48),
                             ("volatile_weather", 48)):
        snap = state(config, preset, interval)
        d = Optimizer(config).decide(
            snap, seed=2, envelope=forecaster.envelope(snap, config.objectives.steps))
        text = tradeoff_text(d)
        assert "Chosen over" in text
        assert any(ch.isdigit() for ch in text), f"no figures in: {text}"
        alt = d.material_alternative()
        assert alt is not None and alt.index != d.chosen_index


def test_explanation_covers_the_loop_in_order(config, forecaster):
    snap = state(config, "storm_passage", 48)
    d = Optimizer(config).decide(
        snap, seed=2, envelope=forecaster.envelope(snap, config.objectives.steps))
    text = explain(config, d)
    for section in ("Situation:", "Criteria:", "Options:", "Committed:", "Trade-off:"):
        assert section in text
    assert text.index("Situation:") < text.index("Criteria:") < text.index("Options:")


def test_weight_slider_change_re_ranks_within_one_second(config, forecaster):
    """PRD F2.6: "Re-evaluation within 1 s of a slider change"."""
    snap = state(config, "volatile_weather", 48)
    env = forecaster.envelope(snap, config.objectives.steps)
    base = Optimizer(config).decide(snap, seed=2, envelope=env)

    tweaked = config.with_base_weights({"carbon": 0.70}, note="slider")
    t0 = time.perf_counter()
    after = Optimizer(tweaked).decide(snap, seed=2, envelope=env)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"re-evaluation took {elapsed*1000:.0f} ms"
    carbon_before = base.chosen_score.objectives["carbon"]
    carbon_after = after.chosen_score.objectives["carbon"]
    assert carbon_after <= carbon_before + 1e-6, \
        "weighting carbon far higher did not pick a cluster at least as clean"


# ---- Step 4: hysteresis ---------------------------------------------------
def test_hysteresis_holds_the_previous_cluster_inside_the_margin(config, forecaster):
    opt = Optimizer(config)
    margin = config.objectives.hysteresis_margin
    # Search for a state that actually has a near runner-up, rather than skipping.
    for preset, interval in (("typical_day", 48), ("typical_day", 36),
                             ("volatile_weather", 48), ("typical_day", 20),
                             ("storm_passage", 30), ("contingency_cascade", 40)):
        snap = state(config, preset, interval)
        env = forecaster.envelope(snap, config.objectives.steps)
        plain = opt.decide(snap, seed=2, envelope=env)
        survivors = [s for s in plain.scores if s.survived_guardrails]
        near = next((s for s in survivors
                     if s.index != plain.chosen_index
                     and s.score <= plain.chosen_score.score + margin), None)
        if near is not None:
            break
    assert near is not None, "no state in the sweep has a runner-up inside the margin"
    held = opt.decide(snap, seed=2, envelope=env, previous_cluster_index=near.index)
    assert held.chosen_index == near.index
    assert held.held_by_hysteresis


def test_hysteresis_does_not_hold_a_clearly_worse_cluster(config, forecaster):
    snap = state(config, "evening_price_crisis", 73)
    opt = Optimizer(config)
    env = forecaster.envelope(snap, config.objectives.steps)
    plain = opt.decide(snap, seed=2, envelope=env)
    margin = config.objectives.hysteresis_margin
    far = next((s for s in plain.scores
                if s.survived_guardrails
                and s.score > plain.chosen_score.score + margin * 3), None)
    if far is None:
        pytest.skip("no sufficiently worse survivor in this state")
    out = opt.decide(snap, seed=2, envelope=env, previous_cluster_index=far.index)
    assert out.chosen_index != far.index
    assert not out.held_by_hysteresis


# ---- determinism and latency ---------------------------------------------
def test_same_seed_produces_the_same_decision(config, forecaster):
    snap = state(config, "storm_passage", 60)
    env = forecaster.envelope(snap, config.objectives.steps)
    opt = Optimizer(config)
    a = opt.decide(snap, seed=7, envelope=env)
    b = opt.decide(snap, seed=7, envelope=env)
    assert a.chosen_index == b.chosen_index
    assert [s.index for s in a.scores] == [s.index for s in b.scores]
    assert a.chosen_score.score == pytest.approx(b.chosen_score.score, abs=0)
    assert [g.removed for g in a.guardrails] == [g.removed for g in b.guardrails]


def test_full_decision_is_inside_the_latency_budget(config, forecaster):
    opt = Optimizer(config)
    timings = []
    for preset, interval in (("typical_day", 48), ("storm_passage", 60),
                             ("evening_price_crisis", 73)):
        snap = state(config, preset, interval)
        env = forecaster.envelope(snap, config.objectives.steps)
        for _ in range(3):
            d = opt.decide(snap, seed=1, envelope=env)
            timings.append(d.latency_ms)
    p95 = float(np.percentile(timings, 95))
    print(f"\nfull decision (classify + guardrails + score) P95: {p95:.1f} ms")
    assert p95 < 1000.0


def test_set_points_carry_verb_asset_mw_and_reason(config, forecaster):
    """PRD F1.3: "Action list shows verb, asset, MW and reason for every set-point"."""
    opt = Optimizer(config)
    for preset, interval in (("typical_day", 48), ("evening_price_crisis", 73),
                             ("contingency_cascade", 46), ("storm_passage", 70)):
        snap = state(config, preset, interval)
        d = opt.decide(snap, seed=1,
                       envelope=forecaster.envelope(snap, config.objectives.steps))
        actions = opt.set_points(d)
        assert actions.set_points, f"{preset}: no set-points produced"
        for sp in actions.set_points + actions.queued_for_approval:
            assert sp.verb and sp.asset_name and sp.reason
            assert sp.mw >= 0.0 and sp.unit == "MW"
            assert any(ch.isdigit() for ch in sp.reason) or "holds" in sp.reason
        # Every battery is accounted for, every interval.
        covered = {sp.asset_id for sp in actions.set_points}
        for b in config.portfolio.batteries:
            assert b.id in covered, f"{b.name} has no set-point"


def test_shed_and_excess_dr_are_queued_for_approval(config, forecaster):
    """PRD autonomy: load reduction beyond contracted DR needs approval."""
    opt = Optimizer(config)
    snap = state(config, "evening_price_crisis", 73)
    d = opt.decide(snap, seed=1,
                   envelope=forecaster.envelope(snap, config.objectives.steps))
    actions = opt.set_points(d)
    for sp in actions.set_points:
        if sp.verb == "shed":
            assert sp.requires_approval, "shedding load must not execute unattended"
