"""M1: the world simulator must be physically plausible and bit-for-bit reproducible."""
import pytest

from orchestrator.domain.events import EventType
from orchestrator.ingest.normalize import Normalizer
from orchestrator.world.presets import PRESETS, preset
from orchestrator.world.world import INTERVALS_PER_DAY, World

ALL_PRESETS = sorted(PRESETS)


def run_day(config, preset_key, seed=11, *, zero_noise=False):
    w = World(config, seed=seed, preset=preset_key)
    frames = []
    for _ in range(INTERVALS_PER_DAY):
        if zero_noise:
            w.demand_noise = 0.0
        frames.append(w.telemetry())
        w.step()
    return w, frames


def test_five_presets_exist_with_prd_names():
    assert ALL_PRESETS == [
        "contingency_cascade", "evening_price_crisis", "storm_passage",
        "typical_day", "volatile_weather",
    ]
    assert all(preset(k).label and preset(k).description for k in ALL_PRESETS)


@pytest.mark.parametrize("preset_key", ALL_PRESETS)
def test_solar_is_zero_at_night_and_peaks_at_midday(config, preset_key):
    _, frames = run_day(config, preset_key)
    for t in frames:
        total = sum(t.solar_mw.values())
        hour = t.minute_of_day / 60.0
        if hour < 5.5 or hour > 19.5:
            assert total == 0.0, f"solar {total} MW at {hour:.2f} h"
        assert total <= config.portfolio.solar_capacity_mw + 1e-6

    midday = [sum(t.solar_mw.values()) for t in frames if 11 <= t.minute_of_day / 60 <= 13]
    assert max(midday) > 0.4 * config.portfolio.solar_capacity_mw


@pytest.mark.parametrize("preset_key", ALL_PRESETS)
def test_wind_respects_the_power_curve(config, preset_key):
    curve = config.portfolio.wind_curve
    _, frames = run_day(config, preset_key)
    saw_output = False
    for t in frames:
        under_maintenance = {
            m.asset_id for m in t.maintenance if m.status == "in_progress"
        }
        for farm in config.portfolio.wind_farms:
            if farm.id in under_maintenance:
                assert t.wind_mw[farm.id] == 0.0, "a unit under maintenance must be at 0 MW"
                continue
            speed = t.wind_speed_ms[farm.id]
            mw = t.wind_mw[farm.id]
            assert -1e-9 <= mw <= farm.capacity_mw + 1e-6
            if speed < curve.cut_in_ms or speed >= curve.cut_out_ms:
                assert mw == pytest.approx(0.0, abs=1e-9), (
                    f"{farm.id} produced {mw} MW at {speed} m/s"
                )
            if speed >= curve.rated_ms and speed < curve.cut_out_ms:
                assert mw == pytest.approx(farm.capacity_mw, rel=1e-6)
            if mw > 0:
                saw_output = True
    assert saw_output


def test_cut_out_actually_occurs_in_the_storm_preset(config):
    """A storm that never reaches cut-out would not test reliability at all."""
    _, frames = run_day(config, "storm_passage")
    curve = config.portfolio.wind_curve
    cut_out_intervals = [
        t.interval for t in frames
        if any(s >= curve.cut_out_ms for s in t.wind_speed_ms.values())
    ]
    assert cut_out_intervals, "storm preset never drives wind past cut-out"


@pytest.mark.parametrize("preset_key", ["typical_day", "volatile_weather", "storm_passage"])
def test_deterministic_load_profile_sits_in_the_prd_band(config, preset_key):
    """PRD reference scenario: 220-330 MW of load.

    Checked on the deterministic profile (noise zeroed); injected demand surges
    deliberately exceed the band, so the two presets carrying DEMAND_SURGE are
    excluded here and covered by the surge test below.
    """
    _, frames = run_day(config, preset_key, zero_noise=True)
    loads = [sum(t.consumer_mw.values()) + t.municipal_mw for t in frames]
    assert 220.0 <= min(loads), f"min load {min(loads):.1f} MW below PRD band"
    assert max(loads) <= 330.0 + 1e-6, f"max load {max(loads):.1f} MW above PRD band"


def test_demand_surge_lifts_load_above_the_normal_band(config):
    _, frames = run_day(config, "evening_price_crisis", zero_noise=True)
    surging = [
        sum(t.consumer_mw.values()) + t.municipal_mw
        for t in frames if any(e.type is EventType.DEMAND_SURGE for e in t.events)
    ]
    assert surging and max(surging) > 330.0


def test_critical_loads_are_flat_and_never_zero(config):
    """The data centre is critical; it must never be curtailed or shaped away."""
    _, frames = run_day(config, "contingency_cascade")
    dc = [t.consumer_mw["cobalt_dc"] for t in frames]
    base = config.portfolio.consumer("cobalt_dc").base_mw
    assert min(dc) > 0.9 * base
    assert max(dc) <= base * 1.1


@pytest.mark.parametrize("preset_key", ALL_PRESETS)
def test_prices_stay_inside_the_configured_band(config, preset_key):
    lo = config.market.price["floor_per_mwh"]
    hi = config.market.price["cap_per_mwh"]
    _, frames = run_day(config, preset_key)
    prices = [t.price_per_mwh for t in frames]
    assert all(lo <= p <= hi for p in prices)
    assert len(set(round(p, 3) for p in prices)) > 10, "price must actually vary"


def test_negative_prices_occur_under_heavy_surplus(config):
    _, frames = run_day(config, "volatile_weather")
    negatives = [
        t for t in frames
        if t.price_per_mwh < 0
    ]
    assert negatives, "no negative prices, so the negative-price regime is untestable"
    for t in negatives:
        renewables = sum(t.solar_mw.values()) + sum(t.wind_mw.values())
        load = sum(t.consumer_mw.values()) + t.municipal_mw
        assert renewables > load, "negative price without a surplus"


def test_price_spike_preset_clears_the_spike_threshold(config):
    _, frames = run_day(config, "evening_price_crisis")
    floor = config.objectives.thresholds["price_spike_floor_per_mwh"]
    assert max(t.price_per_mwh for t in frames) >= floor


@pytest.mark.parametrize("preset_key", ALL_PRESETS)
def test_every_deficit_or_surplus_is_covered_or_stresses_the_tie_line(config, preset_key):
    """At least one preset interval must push past the import limit, otherwise the
    reliability guardrails and the battery fleet are never actually needed."""
    _, frames = run_day(config, preset_key)
    worst = max(
        sum(t.consumer_mw.values()) + t.municipal_mw
        - sum(t.solar_mw.values()) - sum(t.wind_mw.values())
        for t in frames
    )
    assert worst > config.portfolio.grid.import_limit_mw, (
        f"{preset_key}: worst deficit {worst:.1f} MW never exceeds the "
        f"{config.portfolio.grid.import_limit_mw} MW import limit"
    )


def test_frequency_sits_near_nominal_except_during_dips(config):
    nominal = config.portfolio.grid.nominal_frequency_hz
    _, frames = run_day(config, "storm_passage")
    dips = [t for t in frames if any(e.type is EventType.FREQUENCY_DIP for e in t.events)]
    calm = [t for t in frames if t not in dips]
    assert all(abs(t.frequency_hz - nominal) < 0.1 for t in calm)
    assert dips, "storm preset should include a frequency dip"
    assert min(t.frequency_hz for t in dips) < config.portfolio.grid.frequency_reflex_hz


def test_congestion_derates_export_and_the_corridor(config):
    _, frames = run_day(config, "contingency_cascade")
    g = config.portfolio.grid
    congested = [t for t in frames
                 if any(e.type is EventType.LINE_CONGESTION for e in t.events)]
    assert congested
    for t in congested:
        assert t.export_limit_mw == pytest.approx(g.export_derated_mw)
        assert t.corridor_limit_mw["north"] == pytest.approx(g.north_corridor_derated_mw)
    clear = [t for t in frames
             if not any(e.type is EventType.LINE_CONGESTION for e in t.events)]
    assert all(t.export_limit_mw == pytest.approx(g.export_limit_mw) for t in clear)


def test_battery_trip_marks_the_unit_unavailable_then_restores_it(config):
    _, frames = run_day(config, "contingency_cascade")
    tripped = [t for t in frames if not t.battery_available["harbor"]]
    assert tripped, "contingency preset should trip Harbor BESS"
    assert frames[0].battery_available["harbor"], "should start available"
    assert frames[-1].battery_available["harbor"], "should recover after the event"


def test_storm_alert_precedes_arrival_by_its_lead_time(config):
    _, frames = run_day(config, "storm_passage")
    alerts = [e for t in frames for e in t.events if e.type is EventType.STORM_ALERT]
    assert alerts
    ev = alerts[0]
    assert ev.arrival_interval is not None
    lead_intervals = ev.arrival_interval - ev.start_interval
    lead_hours = lead_intervals * config.portfolio.interval_hours
    # The guardrail must look past the 4 h optimiser horizon for this to matter.
    assert lead_hours > config.objectives.lookahead_hours
    assert lead_hours <= config.portfolio.battery_limits.storm_max_lead_hours
    assert not ev.on_site_at(ev.start_interval)
    assert ev.on_site_at(ev.arrival_interval)


def test_all_ten_prd_situations_can_be_injected(config):
    """Every situation in the PRD's list must be injectable in one click."""
    prd_situations = [
        EventType.CLOUD_BANK, EventType.WIND_SURGE, EventType.PRICE_SPIKE,
        EventType.BATTERY_TRIP, EventType.LINE_CONGESTION, EventType.DEMAND_SURGE,
        EventType.STORM_ALERT, EventType.FREQUENCY_DIP, EventType.ISO_DR_CALL,
        EventType.FORECAST_REVISION,
    ]
    assert len(prd_situations) == 10
    for kind in prd_situations:
        w = World(config, seed=5, preset="typical_day", start_interval=36)
        before = w.telemetry()
        ev = w.inject(kind, region="region_a", corridor="north",
                      assets=("harbor",), magnitude=1.5, duration=6, lead=8)
        after = w.telemetry()
        assert ev.type is kind
        assert any(e.type is kind for e in after.events)
        assert len(after.events) == len(before.events) + 1


def test_injected_cloud_bank_cuts_solar_in_its_region_only(config):
    w = World(config, seed=5, preset="typical_day", start_interval=44)  # 11:00
    before = w.telemetry()
    w.inject(EventType.CLOUD_BANK, region="region_a", magnitude=0.95, duration=8)
    w.step()
    after = w.telemetry()
    region_a = [f.id for f in config.portfolio.solar_farms if f.cloud_region == "region_a"]
    region_c = [f.id for f in config.portfolio.solar_farms if f.cloud_region == "region_c"]
    assert sum(after.solar_mw[f] for f in region_a) < \
        0.75 * sum(before.solar_mw[f] for f in region_a)
    # region_c has its own independent driver, so it should be largely unaffected.
    assert sum(after.solar_mw[f] for f in region_c) > \
        0.6 * sum(before.solar_mw[f] for f in region_c)


def test_injected_frequency_dip_breaches_the_reflex_threshold(config):
    w = World(config, seed=5, preset="typical_day", start_interval=36)
    w.inject(EventType.FREQUENCY_DIP, magnitude=0.25, duration=2)
    assert w.telemetry().frequency_hz < config.portfolio.grid.frequency_reflex_hz


@pytest.mark.parametrize("preset_key", ALL_PRESETS)
def test_same_seed_reproduces_the_trace_bit_for_bit(config, preset_key):
    """PRD "Determinism and audit": same inputs and seed produce the same result."""
    _, a = run_day(config, preset_key, seed=99)
    _, b = run_day(config, preset_key, seed=99)
    for ta, tb in zip(a, b):
        assert ta.solar_mw == tb.solar_mw
        assert ta.wind_mw == tb.wind_mw
        assert ta.consumer_mw == tb.consumer_mw
        assert ta.price_per_mwh == tb.price_per_mwh
        assert ta.frequency_hz == tb.frequency_hz


def test_different_seeds_diverge(config):
    _, a = run_day(config, "volatile_weather", seed=1)
    _, b = run_day(config, "volatile_weather", seed=2)
    assert [t.price_per_mwh for t in a] != [t.price_per_mwh for t in b]


def test_soc_is_clamped_to_configured_limits_when_dispatch_is_applied(config):
    from orchestrator.world.world import AppliedDispatch

    lim = config.portfolio.battery_limits
    w = World(config, seed=5, preset="typical_day", start_interval=36)
    # Charge hard for many intervals; SoC must stop at the configured ceiling.
    for _ in range(60):
        w.apply(AppliedDispatch(battery_mw={b.id: -b.power_mw for b in config.portfolio.batteries}))
        w.step()
    assert all(v <= lim.soc_max_pct + 1e-9 for v in w.battery_soc_pct.values())
    for _ in range(80):
        w.apply(AppliedDispatch(battery_mw={b.id: b.power_mw for b in config.portfolio.batteries}))
        w.step()
    assert all(v >= lim.soc_min_pct - 1e-9 for v in w.battery_soc_pct.values())


def test_charging_loses_energy_to_round_trip_efficiency(config):
    from orchestrator.world.world import AppliedDispatch

    w = World(config, seed=5, preset="typical_day", start_interval=36)
    b = config.portfolio.battery("harbor")
    start = w.battery_soc_pct[b.id]
    w.apply(AppliedDispatch(battery_mw={b.id: -40.0}))   # charge 40 MW for 15 min
    gained = w.battery_soc_pct[b.id] - start
    w.apply(AppliedDispatch(battery_mw={b.id: 40.0}))    # discharge the same power
    lost = (w.battery_soc_pct[b.id] - start) - gained
    # Discharging the same MW must cost more SoC than charging gained.
    assert abs(lost) > gained
    ideal = 100.0 * (40.0 * config.portfolio.interval_hours) / b.energy_mwh
    assert gained == pytest.approx(ideal * b.one_way_efficiency, rel=1e-6)


# ---- ingest / data quality -------------------------------------------------
def test_clean_telemetry_is_not_flagged(config):
    w = World(config, seed=3, preset="typical_day", start_interval=36)
    snap = Normalizer(config).build(w.telemetry())
    assert not snap.quality.degraded
    assert snap.quality.stale_fields == ()


def test_dropout_substitutes_and_flags_rather_than_zeroing(config):
    w = World(config, seed=3, preset="typical_day", start_interval=40)
    n = Normalizer(config)
    good = n.build(w.telemetry())
    assert good.total_solar_mw > 0

    w.inject(EventType.DATA_DROPOUT, channels=("solar", "price"), duration=10)
    degraded = n.build(w.telemetry())
    assert "solar" in degraded.quality.substituted_fields
    assert "price" in degraded.quality.substituted_fields
    # Held at the last good value, never zeroed.
    assert degraded.total_solar_mw == pytest.approx(good.total_solar_mw)
    assert degraded.price_per_mwh == pytest.approx(good.price_per_mwh)


def test_stale_beyond_two_intervals_raises_the_alarm(config):
    w = World(config, seed=3, preset="typical_day", start_interval=40)
    n = Normalizer(config)
    n.build(w.telemetry())
    w.inject(EventType.DATA_DROPOUT, channels=("solar",), duration=12)
    for expected in (1, 2, 3, 4):
        snap = n.build(w.telemetry())
        assert snap.quality.intervals_stale["solar"] == expected
        assert n.alarm is (expected > 2)
        w.step()


def test_feed_recovery_clears_the_alarm(config):
    w = World(config, seed=3, preset="typical_day", start_interval=40)
    n = Normalizer(config)
    w.inject(EventType.DATA_DROPOUT, channels=("solar",), duration=5)
    for _ in range(5):          # intervals 40..44 are all missing
        n.build(w.telemetry())
        w.step()
    assert n.alarm, "5 stale intervals should have tripped the alarm"
    snap = n.build(w.telemetry())   # interval 45: feed is back
    assert not n.alarm
    assert "solar" not in snap.quality.substituted_fields


def test_snapshot_derived_quantities(config):
    w = World(config, seed=3, preset="typical_day", start_interval=48)
    snap = Normalizer(config).build(w.telemetry())
    assert snap.available_renewable_mw == pytest.approx(
        snap.total_solar_mw + snap.total_wind_mw)
    assert snap.net_load_mw == pytest.approx(
        snap.total_demand_mw - snap.available_renewable_mw)
    assert snap.price_percentile(0.70) <= max(snap.price_history)


def test_rolling_price_history_supports_percentile_triggers(config):
    w = World(config, seed=3, preset="evening_price_crisis")
    n = Normalizer(config)
    snap = None
    for _ in range(80):
        snap = n.build(w.telemetry())
        w.step()
    assert len(snap.price_history) > 40
    p70 = snap.price_percentile(0.70)
    p50 = snap.price_percentile(0.50)
    assert p70 >= p50
