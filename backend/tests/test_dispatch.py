"""M2: the dispatch core must satisfy the PRD's physical invariants for every one
of the 80 clusters, across randomised states, and do it inside the latency budget."""
import time

import numpy as np
import pytest

from orchestrator.decide.dispatch import OBJECTIVES, DispatchEngine
from orchestrator.domain.actions import CLUSTERS, N_CLUSTERS, BatteryStrategy
from orchestrator.forecast.scenarios import build_scenarios
from orchestrator.ingest.normalize import Normalizer
from orchestrator.world.world import World

PRESETS = ["typical_day", "volatile_weather", "storm_passage",
           "evening_price_crisis", "contingency_cascade"]
TOL = 1e-7


def states(config, *, presets=PRESETS, intervals=(20, 36, 48, 60, 63, 73, 76, 84), seed=5):
    """A spread of real states: night, morning, midday surplus, evening peak."""
    out = []
    for preset in presets:
        w = World(config, seed=seed, preset=preset)
        n = Normalizer(config)
        for i in range(max(intervals) + 1):
            snap = n.build(w.telemetry())
            if i in intervals:
                out.append((preset, snap))
            w.step()
    return out


def all_paths(n_scenarios=3):
    return [(c, s) for c in range(N_CLUSTERS) for s in range(n_scenarios)]


@pytest.fixture(scope="module")
def swept(config):
    """Run every cluster against every state once; reuse across invariant tests."""
    engine = DispatchEngine(config)
    runs = []
    for preset, snap in states(config):
        scen = build_scenarios(config, snap, seed=hash(preset) % 1000, n_scenarios=3)
        out = engine.simulate(scen, snap.battery_soc_pct, snap.battery_available,
                              record=all_paths(3))
        runs.append((preset, snap, scen, out))
    return engine, runs


def every_record(runs):
    for preset, snap, scen, out in runs:
        for (ci, si), trace in out.records.items():
            for rec in trace:
                yield preset, snap, scen, out, ci, si, rec


# ---- invariant 1: energy balance ------------------------------------------
def test_energy_balance_holds_every_interval_for_every_cluster(config, swept):
    _, runs = swept
    checked = 0
    for _, _, _, _, ci, si, r in every_record(runs):
        served = r.demand_mw - r.dr_mw - r.shift_mw - r.unserved_mw
        lhs = r.renewable_used_mw + r.import_mw + r.battery_discharge_mw
        rhs = served + r.battery_charge_mw + r.export_mw
        assert lhs == pytest.approx(rhs, abs=1e-6), (
            f"cluster {ci} scenario {si} step {r.step}: "
            f"supply {lhs:.6f} != demand {rhs:.6f}"
        )
        checked += 1
    assert checked > 20_000, f"only {checked} intervals checked"


# ---- invariant 2: SoC limits ----------------------------------------------
def test_soc_stays_within_configured_limits(config, swept):
    _, runs = swept
    lim = config.portfolio.battery_limits
    for _, _, _, _, ci, si, r in every_record(runs):
        for bid, soc in r.battery_soc_pct.items():
            assert -TOL <= soc <= lim.soc_max_pct + TOL, (
                f"cluster {ci} step {r.step}: {bid} SoC {soc:.4f}%"
            )


def test_normal_dispatch_holds_above_the_reserve_floor(config, swept):
    """Normal dispatch holds above the 10% reserve floor. Only the emergency cascade,
    run to avoid shedding load, may go below it -- and never below the 8% hard floor."""
    _, runs = swept
    lim = config.portfolio.battery_limits
    breaches = 0
    for _, _, _, _, ci, si, r in every_record(runs):
        for bid, soc in r.battery_soc_pct.items():
            assert soc >= lim.soc_min_pct - 1e-6, (
                f"cluster {ci} step {r.step}: {bid} at {soc:.4f}% breached the "
                f"{lim.soc_min_pct}% hard floor"
            )
        # A sub-floor SoC persists after a legitimate emergency, so the invariant is
        # about *discharging* at or below the reserve floor, not about sitting below it.
        digging_in = [
            bid for bid, mw in r.battery_mw.items()
            if mw > 1e-9 and r.battery_soc_pct[bid] < lim.reserve_floor_pct - 1e-6
        ]
        if digging_in:
            breaches += 1
            assert r.import_mw >= r.import_limit_mw - 1e-6, (
                f"cluster {ci} step {r.step}: {digging_in} discharged below the "
                f"{lim.reserve_floor_pct}% reserve floor while the tie-line was only at "
                f"{r.import_mw:.1f}/{r.import_limit_mw} MW, so it was not an emergency"
            )
    assert breaches > 0, "the emergency cascade was never exercised"


# ---- invariant 3: power ratings -------------------------------------------
def test_no_unit_exceeds_its_rating_and_tripped_units_sit_at_zero(config, swept):
    _, runs = swept
    for _, snap, _, _, ci, si, r in every_record(runs):
        for b in config.portfolio.batteries:
            mw = r.battery_mw[b.id]
            assert abs(mw) <= b.power_mw + TOL, (
                f"cluster {ci} step {r.step}: {b.id} at {mw:.4f} MW "
                f"against a {b.power_mw} MW rating"
            )
            if not snap.battery_available[b.id]:
                assert mw == pytest.approx(0.0, abs=TOL), (
                    f"tripped battery {b.id} dispatched {mw} MW"
                )


# ---- invariant 4: transmission limits -------------------------------------
def test_imports_exports_and_corridor_flow_stay_within_current_limits(config, swept):
    _, runs = swept
    for _, _, _, _, ci, si, r in every_record(runs):
        assert r.import_mw <= r.import_limit_mw + 1e-6, (
            f"cluster {ci} step {r.step}: import {r.import_mw:.4f} MW "
            f"> limit {r.import_limit_mw}"
        )
        assert r.export_mw <= r.export_limit_mw + 1e-6, (
            f"cluster {ci} step {r.step}: export {r.export_mw:.4f} MW "
            f"> limit {r.export_limit_mw}"
        )
        for corridor, flow in r.corridor_flow_mw.items():
            cap = r.corridor_limit_mw.get(corridor, float("inf"))
            assert flow <= cap + 1e-6, (
                f"cluster {ci} step {r.step}: {corridor} corridor flow "
                f"{flow:.4f} MW > limit {cap}"
            )


# ---- invariant 5: no wash trading -----------------------------------------
def test_never_imports_and_exports_in_the_same_interval(config, swept):
    _, runs = swept
    for _, _, _, _, ci, si, r in every_record(runs):
        assert not (r.import_mw > 1e-9 and r.export_mw > 1e-9), (
            f"cluster {ci} step {r.step}: wash trade "
            f"{r.import_mw} MW in / {r.export_mw} MW out"
        )


# ---- invariant 6: justified curtailment -----------------------------------
def test_spill_only_happens_for_a_limit_or_a_negative_price(config, swept):
    """PRD: spill only if the corridor is full, the export line is full, or the price
    is negative, with batteries absorbing what they can."""
    _, runs = swept
    for _, _, _, _, ci, si, r in every_record(runs):
        if r.total_curtailed_mw <= 1e-9:
            continue
        corridor_full = any(
            flow >= r.corridor_limit_mw.get(c, float("inf")) - 1e-6
            for c, flow in r.corridor_flow_mw.items()
        )
        export_full = r.export_mw >= r.export_limit_mw - 1e-6
        negative = r.price_per_mwh < 0.0
        assert corridor_full or export_full or negative, (
            f"cluster {ci} step {r.step}: spilled {r.total_curtailed_mw:.4f} MW with "
            f"corridor not full, export {r.export_mw:.2f}/{r.export_limit_mw} MW, "
            f"price ${r.price_per_mwh:.2f}"
        )


def test_never_discharges_while_curtailing_for_a_transmission_limit(config, swept):
    """The PRD's open curtailment defect: "Battery discharging at full power while
    solar was curtailed". Forbidden by construction here."""
    _, runs = swept
    for _, _, _, _, ci, si, r in every_record(runs):
        limit_driven = r.curtailed_by_corridor_mw + r.curtailed_by_export_mw
        if limit_driven <= 1e-9:
            continue
        # Two reflexes legitimately outrank the rule, both of them PRD invariants:
        #   - a frequency dip forces injection (grid stability beats spilled energy);
        #   - batteries discharge before load is shed.
        allowance = max(r.forced_injection_mw, 0.0)
        preventing_shedding = r.import_mw >= r.import_limit_mw - 1e-6
        if preventing_shedding:
            continue
        assert r.battery_discharge_mw <= allowance + 1e-6, (
            f"cluster {ci} step {r.step}: discharging {r.battery_discharge_mw:.3f} MW "
            f"while spilling {limit_driven:.3f} MW for a transmission limit "
            f"(reflex allowance {allowance:.3f} MW, imports "
            f"{r.import_mw:.1f}/{r.import_limit_mw} MW)"
        )


def test_batteries_absorb_before_spilling_for_the_export_limit(config, swept):
    _, runs = swept
    for _, snap, _, _, ci, si, r in every_record(runs):
        if r.curtailed_by_export_mw <= 1e-6:
            continue
        # Every available battery must be at its charge rating or full, otherwise
        # there was headroom we should have soaked instead of spilling.
        for b in config.portfolio.batteries:
            if not snap.battery_available[b.id]:
                continue
            charging = max(0.0, -r.battery_mw[b.id])
            at_rating = charging >= b.power_mw - 1e-3
            full = r.battery_soc_pct[b.id] >= \
                config.portfolio.battery_limits.soc_max_pct - 1e-3
            assert at_rating or full, (
                f"cluster {ci} step {r.step}: spilled {r.curtailed_by_export_mw:.3f} MW "
                f"for the export limit while {b.id} was only absorbing "
                f"{charging:.2f}/{b.power_mw} MW at {r.battery_soc_pct[b.id]:.2f}% SoC"
            )


# ---- invariant 7: unavoidable shedding only -------------------------------
def test_load_is_shed_only_with_imports_maxed_and_batteries_at_the_floor(config, swept):
    _, runs = swept
    lim = config.portfolio.battery_limits
    shed_seen = 0
    for _, snap, _, _, ci, si, r in every_record(runs):
        if r.unserved_mw <= 1e-9:
            continue
        shed_seen += 1
        assert r.import_mw >= r.import_limit_mw - 1e-6, (
            f"cluster {ci} step {r.step}: shed {r.unserved_mw:.3f} MW with imports at "
            f"{r.import_mw:.2f}/{r.import_limit_mw} MW"
        )
        for b in config.portfolio.batteries:
            if not snap.battery_available[b.id]:
                continue
            mw = r.battery_mw[b.id]
            at_rating = mw >= b.power_mw - 1e-3
            at_hard_floor = r.battery_soc_pct[b.id] <= lim.soc_min_pct + 1e-3
            assert at_rating or at_hard_floor, (
                f"cluster {ci} step {r.step}: shed {r.unserved_mw:.3f} MW while {b.id} "
                f"was delivering only {mw:.2f}/{b.power_mw} MW at "
                f"{r.battery_soc_pct[b.id]:.2f}% SoC -- neither power- nor energy-limited"
            )
    # The sweep must actually reach shedding somewhere, or this proves nothing.
    assert shed_seen > 0, "no interval in the sweep ever shed load"


# ---- invariant 8: frequency response --------------------------------------
def test_batteries_inject_during_every_frequency_dip(config, swept):
    _, runs = swept
    reflex = config.portfolio.grid.frequency_reflex_hz
    dips = 0
    for _, snap, _, _, ci, si, r in every_record(runs):
        if r.frequency_hz >= reflex:
            continue
        if not any(snap.battery_available.values()):
            continue
        dips += 1
        assert r.battery_discharge_mw > 0.0, (
            f"cluster {ci} step {r.step}: {r.frequency_hz:.3f} Hz dip with no injection"
        )
    assert dips > 0, "sweep contained no frequency dips"


def test_frequency_reflex_applies_to_every_cluster_including_soak(config):
    """The reflex is applied inside every cluster, not an optional strategy."""
    from orchestrator.domain.events import EventType

    engine = DispatchEngine(config)
    w = World(config, seed=4, preset="typical_day", start_interval=48)
    w.inject(EventType.FREQUENCY_DIP, magnitude=0.3, duration=4)
    snap = Normalizer(config).build(w.telemetry())
    scen = build_scenarios(config, snap, seed=2, n_scenarios=1)
    soak = [c.index for c in CLUSTERS if c.battery is BatteryStrategy.SOAK_SURPLUS]
    out = engine.simulate(scen, snap.battery_soc_pct, snap.battery_available,
                          record=[(i, 0) for i in soak])
    assert snap.frequency_hz < config.portfolio.grid.frequency_reflex_hz
    for i in soak:
        first = out.trace(i, 0)[0]
        assert first.battery_discharge_mw > 0.0, (
            f"soak-surplus cluster {i} failed to inject during a dip"
        )


# ---- invariant 9: critical load is never curtailed ------------------------
def test_critical_load_is_never_shed_by_demand_response_or_shifting(config, swept):
    _, runs = swept
    critical = sum(c.base_mw for c in config.portfolio.consumers if c.critical)
    non_critical_flex = config.portfolio.total_dr_mw + config.portfolio.total_shiftable_mw
    assert critical > 0 and non_critical_flex > 0
    for _, _, _, _, ci, si, r in every_record(runs):
        assert r.dr_mw <= config.portfolio.total_dr_mw + TOL
        assert abs(r.shift_mw) <= config.portfolio.total_shiftable_mw + TOL
        # Flexibility only ever comes from non-critical consumers.
        assert r.dr_mw + max(0.0, r.shift_mw) <= non_critical_flex + TOL


# ---- the commitment matches what was scored -------------------------------
def test_recorded_trace_reproduces_the_scored_objectives_exactly(config):
    """The decisive consistency check: the per-step detail the operator sees and the
    (cluster, scenario) objective the optimizer scored must be the same run."""
    engine = DispatchEngine(config)
    _, snap = states(config, presets=["volatile_weather"], intervals=(48,))[0]
    scen = build_scenarios(config, snap, seed=3, n_scenarios=4)
    out = engine.simulate(scen, snap.battery_soc_pct, snap.battery_available,
                          record=all_paths(4))
    h = config.portfolio.interval_hours
    m = config.market
    for ci in range(0, N_CLUSTERS, 7):
        for si in range(4):
            trace = out.trace(ci, si)
            scored = out.objective_at(ci, si)

            carbon = sum(r.import_mw * h for r in trace) * m.carbon["grid_intensity_t_per_mwh"]
            assert carbon == pytest.approx(scored["carbon"], rel=1e-9, abs=1e-9)

            curt = sum(r.total_curtailed_mw * h for r in trace)
            assert curt == pytest.approx(scored["curtailment"], rel=1e-7, abs=1e-7)

            wear = sum(
                abs(mw) * h * config.portfolio.battery(b).wear_cost_per_mwh
                for r in trace for b, mw in r.battery_mw.items()
            )
            assert wear == pytest.approx(scored["battery_wear"], rel=1e-9, abs=1e-9)

            unserved = sum(r.unserved_mw * h for r in trace)
            shortfall = sum(r.reserve_shortfall_mw * h for r in trace)
            expected = (config.objectives.reliability["unserved_weight"] * unserved
                        + config.objectives.reliability["reserve_shortfall_weight"] * shortfall)
            assert expected == pytest.approx(scored["reliability_risk"], rel=1e-9, abs=1e-9)


def test_all_six_objectives_are_produced_for_all_eighty_clusters(config, swept):
    _, runs = swept
    _, _, _, out = runs[0]
    assert tuple(out.objectives) == OBJECTIVES
    for key, arr in out.objectives.items():
        assert arr.shape[0] == N_CLUSTERS
        assert np.isfinite(arr).all(), f"{key} contains non-finite values"
    # Every objective must discriminate between clusters somewhere in the sweep,
    # otherwise scoring on it could never change a decision. (In any single state
    # some objectives legitimately tie -- e.g. corridor spill is forced, so it is
    # cluster-independent in a congested interval.)
    for key in OBJECTIVES:
        spreads = [
            float(np.ptp(o.objectives[key].mean(axis=1))) for _, _, _, o in runs
        ]
        assert max(spreads) > 0, f"{key} never discriminates between clusters"


def test_strategies_behave_distinguishably(config):
    """Soak must not discharge; discharge-now must discharge whenever it can."""
    engine = DispatchEngine(config)
    _, snap = states(config, presets=["typical_day"], intervals=(48,))[0]
    scen = build_scenarios(config, snap, seed=9, n_scenarios=1)
    by_strategy = {}
    for strategy in BatteryStrategy:
        idx = next(c.index for c in CLUSTERS
                   if c.battery is strategy and not c.demand_response and not c.load_shift)
        by_strategy[strategy] = idx
    out = engine.simulate(scen, snap.battery_soc_pct, snap.battery_available,
                          record=[(i, 0) for i in by_strategy.values()])

    soak = out.trace(by_strategy[BatteryStrategy.SOAK_SURPLUS], 0)
    assert all(r.battery_discharge_mw <= 1e-9 for r in soak), \
        "soak-surplus discharged outside a frequency dip"

    now = out.trace(by_strategy[BatteryStrategy.DISCHARGE_NOW], 0)
    assert sum(r.battery_discharge_mw for r in now) > 0

    hold = out.trace(by_strategy[BatteryStrategy.HOLD_FOR_HIGHER_PRICE], 0)
    assert sum(r.battery_discharge_mw for r in hold) <= \
        sum(r.battery_discharge_mw for r in now) + 1e-6


def test_charging_never_exceeds_available_headroom(config, swept):
    _, runs = swept
    lim = config.portfolio.battery_limits
    for _, _, _, _, ci, si, r in every_record(runs):
        for b in config.portfolio.batteries:
            if r.battery_mw[b.id] < -1e-9:
                assert r.battery_soc_pct[b.id] <= lim.soc_max_pct + TOL


# ---- determinism -----------------------------------------------------------
def test_same_inputs_and_seed_produce_the_same_decision(config):
    engine = DispatchEngine(config)
    _, snap = states(config, presets=["storm_passage"], intervals=(60,))[0]
    results = []
    for _ in range(3):
        scen = build_scenarios(config, snap, seed=42)
        out = engine.simulate(scen, snap.battery_soc_pct, snap.battery_available)
        results.append({k: v.copy() for k, v in out.objectives.items()})
    for key in OBJECTIVES:
        assert np.array_equal(results[0][key], results[1][key])
        assert np.array_equal(results[1][key], results[2][key])


def test_scenario_zero_is_the_expected_path(config):
    _, snap = states(config, presets=["typical_day"], intervals=(48,))[0]
    scen = build_scenarios(config, snap, seed=1)
    # Scenario 0 carries no shocks, so its spread over scenarios should be central.
    for name, arr in (("solar", scen.solar_mw), ("demand", scen.demand_mw)):
        lo = np.percentile(arr, 5, axis=0)
        hi = np.percentile(arr, 95, axis=0)
        assert np.all(arr[0] >= lo - 1e-6), f"{name} scenario 0 below the 5th percentile"
        assert np.all(arr[0] <= hi + 1e-6), f"{name} scenario 0 above the 95th percentile"


# ---- invariant 10: latency -------------------------------------------------
def test_decision_latency_is_inside_the_budget(config):
    """PRD: P95 < 1 s for 80 clusters x 24 scenarios x 16 steps."""
    engine = DispatchEngine(config)
    samples = states(config, intervals=(36, 48, 76))
    timings = []
    for _, snap in samples:
        scen = build_scenarios(config, snap, seed=1)
        for _ in range(4):
            t0 = time.perf_counter()
            engine.simulate(scen, snap.battery_soc_pct, snap.battery_available)
            timings.append((time.perf_counter() - t0) * 1000.0)
    p95 = float(np.percentile(timings, 95))
    assert p95 < 1000.0, f"P95 {p95:.1f} ms exceeds the 1 s budget"
    # Record the actual figure in the test output.
    print(f"\noptimizer P95 latency: {p95:.1f} ms over {len(timings)} runs")


def test_sweep_shape_matches_the_prd(config):
    o = config.objectives
    assert N_CLUSTERS == 80
    assert o.scenarios == 24
    assert o.steps == 16


def test_soak_surplus_actually_absorbs_an_available_surplus(config):
    """"Soak surplus: absorb everything spare" has to mean absorb it, not export it.

    Charging is the one dispatch direction no physical invariant can catch -- energy
    balance, SoC limits and the transmission caps all hold whether storage fills or the
    surplus goes out of the tie-line. So it needs its own assertion, and the PRD's whole
    stored-energy clause depends on it: a fleet that only fills as a side effect of the
    export cap arrives at the evening peak empty.
    """
    engine = DispatchEngine(config)
    checked = 0
    for preset, snap in states(config, intervals=(36, 40, 44, 48, 52)):
        if snap.net_load_mw >= -5.0:
            continue                      # no surplus worth absorbing in this state
        fleet = snap.fleet_soc_pct({b.id: b.energy_mwh for b in config.portfolio.batteries})
        if fleet > 85.0:
            continue                      # nowhere to put it
        scen = build_scenarios(config, snap, seed=404, n_scenarios=3)
        soak = next(c for c in CLUSTERS
                    if c.battery is BatteryStrategy.SOAK_SURPLUS
                    and not c.demand_response and not c.load_shift)
        out = engine.simulate(scen, snap.battery_soc_pct, snap.battery_available,
                              record=(soak.index, 0))
        r = out.trace(soak.index, 0)[0]
        if r.curtailed_by_corridor_mw > 1e-6 or r.export_limit_mw <= 1e-6:
            continue                      # the surplus cannot reach storage
        charged = -sum(min(0.0, v) for v in r.battery_mw.values())
        checked += 1
        assert charged > 0.0, (
            f"{preset} interval {snap.interval}: {abs(snap.net_load_mw):.1f} MW surplus, "
            f"fleet at {fleet:.0f}%, export limit {r.export_limit_mw:.0f} MW -- Soak "
            f"surplus stored nothing and exported {r.export_mw:.1f} MW")
    assert checked >= 3, f"only {checked} surplus states exercised; the test is too weak"


def test_price_arbitrage_charges_below_the_low_band(config):
    """The one strategy the PRD has charging from the grid on purpose must do it."""
    engine = DispatchEngine(config)
    arb = next(c for c in CLUSTERS
               if c.battery is BatteryStrategy.PRICE_ARBITRAGE
               and not c.demand_response and not c.load_shift
               and not c.curtailment.value.startswith("at_negative"))
    found = 0
    for preset, snap in states(config):
        scen = build_scenarios(config, snap, seed=505, n_scenarios=3)
        if snap.price_per_mwh >= float(scen.arbitrage_low_per_mwh[0]):
            continue
        fleet = snap.fleet_soc_pct({b.id: b.energy_mwh for b in config.portfolio.batteries})
        if fleet > 85.0:
            continue
        out = engine.simulate(scen, snap.battery_soc_pct, snap.battery_available,
                              record=(arb.index, 0))
        r = out.trace(arb.index, 0)[0]
        if r.frequency_hz < config.portfolio.grid.frequency_reflex_hz:
            continue                      # the reflex outranks arbitrage, correctly
        if r.import_mw >= r.import_limit_mw - 1e-6 or r.unserved_mw > 1e-6:
            continue                      # no import headroom to charge through
        found += 1
        charged = -sum(min(0.0, v) for v in r.battery_mw.values())
        assert charged > 0.0, (
            f"{preset} interval {snap.interval}: price ${snap.price_per_mwh:.2f} is below "
            f"the ${float(scen.arbitrage_low_per_mwh[0]):.2f} charge band with the fleet "
            f"at {fleet:.0f}%, but nothing was charged")
    if not found:
        pytest.skip("no state in the sweep sat below its own charge band")
