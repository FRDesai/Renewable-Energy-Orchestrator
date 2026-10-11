"""M3: the quantile forecasters must be calibrated, monotone and reproducible, and
the scenario fan must be consistent with their bands."""
import numpy as np
import pytest

from orchestrator.decide.dispatch import DispatchEngine
from orchestrator.domain.events import EventType
from orchestrator.forecast.features import N_FEATURES, TARGETS, build_row, target_value
from orchestrator.forecast.quantile import QUANTILES, QuantileForecaster
from orchestrator.forecast.scenarios import Envelope, build_scenarios
from orchestrator.forecast.train import calibration_samples, measure_coverage
from orchestrator.ingest.normalize import Normalizer
from orchestrator.world.world import INTERVALS_PER_DAY, World

# Seeds used nowhere in training (1-4) or calibration (31-32).
HOLDOUT_SEEDS = (77, 78)


@pytest.fixture(scope="module")
def forecaster():
    f = QuantileForecaster.load()
    if f is None or not f.is_trained:
        pytest.skip("run `make train` first to build artifacts/forecasters.pkl")
    return f


@pytest.fixture(scope="module")
def holdout(config, forecaster):
    return calibration_samples(config, forecaster, seeds=HOLDOUT_SEEDS)


def day(config, preset, seed=61, interval=None):
    w = World(config, seed=seed, preset=preset)
    n = Normalizer(config)
    snaps = []
    for _ in range(INTERVALS_PER_DAY):
        snaps.append(n.build(w.telemetry()))
        w.step()
    return snaps if interval is None else snaps[interval]


# ---- the models themselves -------------------------------------------------
def test_one_model_per_target_and_quantile(forecaster):
    assert set(forecaster.models) == {(t, q) for t in TARGETS for q in QUANTILES}
    assert QUANTILES == (0.10, 0.50, 0.90), "PRD specifies P10/P50/P90"
    assert forecaster.trained_rows > 10_000


def test_gradient_boosted_quantile_regression_is_what_is_used(forecaster):
    """PRD names the technique explicitly."""
    from sklearn.ensemble import GradientBoostingRegressor
    for (target, q), model in forecaster.models.items():
        assert isinstance(model, GradientBoostingRegressor)
        assert model.loss == "quantile"
        assert model.alpha == pytest.approx(q)


def test_feature_row_is_well_formed(config):
    snap = day(config, "typical_day", interval=48)
    row = build_row(snap, "solar", 4, history=[100.0, 110.0, 120.0, 130.0])
    assert row.shape == (N_FEATURES,)
    assert np.isfinite(row).all()


# ---- calibration on a third, never-seen seed set ---------------------------
def test_bands_are_monotone_everywhere(config, forecaster):
    for preset in ("typical_day", "storm_passage", "evening_price_crisis"):
        snaps = day(config, preset)
        for i in (12, 36, 48, 72, 84):
            for target in TARGETS:
                band = forecaster.predict(snaps[i], target, config.objectives.steps)
                assert band.shape == (3, config.objectives.steps)
                assert np.all(band[0] <= band[1] + 1e-9), f"{target}: p10 above p50"
                assert np.all(band[1] <= band[2] + 1e-9), f"{target}: p50 above p90"


def test_p10_p90_coverage_is_calibrated_on_unseen_data(config, forecaster, holdout):
    """PRD risk: "Forecast models under-state uncertainty -> plans too aggressive".
    Mitigation: calibrate the spread and monitor P10-P90 coverage. Checked here on
    seeds used in neither training nor calibration."""
    coverage = measure_coverage(holdout, forecaster)
    assert set(coverage) == set(TARGETS)
    for target, cov in sorted(coverage.items()):
        print(f"\n  {target:8s} held-out P10-P90 coverage {100 * cov:5.1f}%")
        assert cov >= 0.70, (
            f"{target} coverage {100*cov:.1f}% -- the band understates uncertainty, "
            f"which is the failure mode the PRD calls out"
        )
        assert cov <= 0.97, f"{target} coverage {100*cov:.1f}% -- band uselessly wide"


def test_calibration_improved_undercovered_targets(forecaster):
    """Wind and price were measurably under-covered before calibration."""
    assert forecaster.calibration, "no calibration was fitted"
    for target in ("wind", "price"):
        scale = forecaster.calibration[target]
        assert scale.max() > 1.0, f"{target} band was not widened"
    # Degenerate night-time solar bands must have been excluded, or every quantile
    # collapses to exactly 1.0 and calibration silently does nothing.
    assert forecaster.degenerate_skipped.get("solar", 0) > 0


def test_bands_respect_physical_bounds(config, forecaster):
    snaps = day(config, "volatile_weather")
    p = config.portfolio
    for i in (6, 30, 48, 66, 80):
        solar = forecaster.predict(snaps[i], "solar", config.objectives.steps)
        wind = forecaster.predict(snaps[i], "wind", config.objectives.steps)
        assert solar.min() >= -1e-9 and solar.max() <= p.solar_capacity_mw + 1e-6
        assert wind.min() >= -1e-9 and wind.max() <= p.wind_capacity_mw + 1e-6
        price = forecaster.predict(snaps[i], "price", config.objectives.steps)
        assert price.min() >= config.market.price["floor_per_mwh"] - 1e-6
        assert price.max() <= config.market.price["cap_per_mwh"] + 1e-6


def test_step_zero_is_the_measurement_not_a_prediction(config, forecaster):
    snaps = day(config, "typical_day")
    for i in (20, 48, 76):
        for target in TARGETS:
            band = forecaster.predict(snaps[i], target, 8)
            observed = target_value(snaps[i], target)
            assert band[0, 0] == pytest.approx(observed)
            assert band[1, 0] == pytest.approx(observed)
            assert band[2, 0] == pytest.approx(observed)


def test_spread_grows_with_lead_time(config, forecaster):
    """PRD: "Spread grows with lead time and with a volatility factor"."""
    snaps = day(config, "volatile_weather")
    T = config.objectives.steps
    for target in ("demand", "price", "wind"):
        widths = np.zeros(T)
        n = 0
        for i in range(8, 72, 4):
            band = forecaster.predict(snaps[i], target, T)
            widths += band[2] - band[0]
            n += 1
        widths /= n
        early = widths[1:4].mean()
        late = widths[T - 4:].mean()
        assert late > early, (
            f"{target}: band width {late:.2f} at long lead is not wider than "
            f"{early:.2f} at short lead"
        )


# ---- the scenario fan ------------------------------------------------------
def test_scenario_fan_is_consistent_with_the_forecast_bands(config, forecaster):
    """PRD: "The ML quantile forecasts set the P10-P90 envelope per step. Within it,
    each scenario evolves the underlying drivers". So the fan's own 10-90 width must
    track the forecast band, not merely be plausible.

    Demand is unconstrained and must match exactly. Solar is clipped per farm at 0 MW
    and nameplate, so saturation can narrow the fan -- but never widen it.
    """
    snap = day(config, "volatile_weather", interval=46)
    env = forecaster.envelope(snap, config.objectives.steps)
    scen = build_scenarios(config, snap, seed=5, envelope=env)

    def ratios(paths, band, cap):
        own = np.percentile(paths, 90, axis=0) - np.percentile(paths, 10, axis=0)
        want = band[2] - band[0]
        live = (want > 5.0) & (band[0] > 5.0) & (band[2] < cap - 5.0)
        assert live.sum() >= 8, f"only {live.sum()} comparable steps"
        return own[live] / want[live]

    demand_r = ratios(scen.demand_mw, env.demand, 1e9)
    assert np.allclose(demand_r, 1.0, atol=0.02), (
        f"demand fan width should match the band exactly "
        f"({demand_r.min():.3f}-{demand_r.max():.3f})"
    )

    solar_r = ratios(scen.solar_mw, env.solar, config.portfolio.solar_capacity_mw)
    assert solar_r.max() <= 1.05, (
        f"solar fan is wider than the forecast band ({solar_r.max():.3f}) -- "
        f"clipping can only narrow it"
    )
    assert solar_r.mean() >= 0.80, (
        f"solar fan width ({solar_r.mean():.3f} of the band) has drifted away from "
        f"the forecast rather than tracking it"
    )


def test_scenario_zero_tracks_the_forecast_median(config, forecaster):
    snap = day(config, "typical_day", interval=44)
    env = forecaster.envelope(snap, config.objectives.steps)
    scen = build_scenarios(config, snap, seed=5, envelope=env)
    # Scenario 0 carries no shocks, so after calibration it should sit on the median.
    assert np.allclose(scen.demand_mw[0], env.demand[1], rtol=0.02, atol=1.0)


def test_step_zero_of_the_fan_equals_the_measured_state(config, forecaster):
    """The committed interval must be the measured one in every scenario, or the
    agent would act on a forecast of the present."""
    snap = day(config, "contingency_cascade", interval=50)
    env = forecaster.envelope(snap, config.objectives.steps)
    scen = build_scenarios(config, snap, seed=3, envelope=env)
    assert np.allclose(scen.solar_mw[:, 0], snap.total_solar_mw, atol=1e-6)
    assert np.allclose(scen.demand_mw[:, 0], snap.total_demand_mw, rtol=1e-6)
    assert np.allclose(scen.price_per_mwh[:, 0], snap.price_per_mwh, atol=1e-9)
    assert np.allclose(scen.frequency_hz[:, 0], snap.frequency_hz, atol=1e-9)
    for farm in config.portfolio.solar_farms:
        assert scen.solar_by_farm[farm.id][:, 0] == pytest.approx(
            snap.solar_mw[farm.id], abs=1e-6)


def test_fan_is_reproducible_from_the_seed(config, forecaster):
    snap = day(config, "storm_passage", interval=50)
    env = forecaster.envelope(snap, config.objectives.steps)
    a = build_scenarios(config, snap, seed=12, envelope=env)
    b = build_scenarios(config, snap, seed=12, envelope=env)
    assert np.array_equal(a.solar_mw, b.solar_mw)
    assert np.array_equal(a.demand_mw, b.demand_mw)
    assert np.array_equal(a.price_per_mwh, b.price_per_mwh)
    c = build_scenarios(config, snap, seed=13, envelope=env)
    assert not np.array_equal(a.demand_mw, c.demand_mw)


def test_unsettled_weather_widens_the_fan(config):
    """PRD: spread rises "with a volatility factor that rises in unsettled weather"."""
    snap = day(config, "typical_day", interval=44)
    calm = build_scenarios(config, snap, seed=4, volatility=1.0)
    rough = build_scenarios(config, snap, seed=4, volatility=2.5)
    calm_w = np.percentile(calm.demand_mw, 90, axis=0) - np.percentile(calm.demand_mw, 10, axis=0)
    rough_w = np.percentile(rough.demand_mw, 90, axis=0) - np.percentile(rough.demand_mw, 10, axis=0)
    assert rough_w[1:].mean() > calm_w[1:].mean()


def test_announced_storm_arrival_is_jittered_across_scenarios(config):
    """PRD: "a storm with its forecast arrival jittered by about +/- 0.6 h per scenario"."""
    w = World(config, seed=9, preset="typical_day", start_interval=40)
    w.inject(EventType.STORM_ALERT, duration=40, lead=16, magnitude=1.0)
    snap = Normalizer(config).build(w.telemetry())
    scen = build_scenarios(config, snap, seed=2, n_scenarios=24)
    north = scen.wind_mw["north"]
    # Different scenarios must see the storm land at different steps, otherwise the
    # guardrail is being tested against a single deterministic arrival.
    onset = []
    for s in range(north.shape[0]):
        jumps = np.where(np.diff(north[s]) > 20.0)[0]
        if jumps.size:
            onset.append(int(jumps[0]))
    assert len(set(onset)) > 1, f"storm arrival identical in every scenario: {set(onset)}"


def test_unannounced_surprises_decay_with_the_configured_persistence(config):
    """PRD: an unannounced spike "is seen once it starts and assumed to decay with a
    typical persistence" -- 45 min for a price spike.

    Measured as the spike's own multiplier against an otherwise identical fan, so the
    diurnal price rise into the evening peak cannot mask the decay.
    """
    import dataclasses

    w = World(config, seed=9, preset="typical_day", start_interval=60)
    ev = w.inject(EventType.PRICE_SPIKE, magnitude=6.0, duration=20)
    snap = Normalizer(config).build(w.telemetry())
    baseline = dataclasses.replace(
        snap, events=tuple(e for e in snap.events
                           if e.type is not EventType.PRICE_SPIKE))

    spiked = build_scenarios(config, snap, seed=2, n_scenarios=1).price_per_mwh[0]
    plain = build_scenarios(config, baseline, seed=2, n_scenarios=1).price_per_mwh[0]
    uplift = spiked - plain          # $/MWh attributable to the spike

    # Step 0 is the measured present and is identical in both fans, so the decay is
    # measured over the forecast steps.
    tau_steps = int(config.objectives.persistence_hours["price_spike"]
                    / config.portfolio.interval_hours)
    assert uplift[0] == pytest.approx(0.0), "step 0 must be the measurement in both"
    assert uplift[1] > 50.0, f"spike absent at step 1 (uplift ${uplift[1]:.1f})"
    assert np.all(np.diff(uplift[1:]) <= 1e-6), "uplift must decay monotonically"
    # One persistence time should remove most of it.
    assert uplift[1 + tau_steps] < uplift[1] * 0.45
    assert uplift[-1] < uplift[1] * 0.05, f"never decayed (${uplift[-1]:.1f} left)"
    # A spike only ever raises the price.
    assert np.all(uplift >= -1e-9)


def test_announced_congestion_clears_at_its_known_end(config):
    """PRD: "a congestion derate ... with known end" enters all scenarios."""
    w = World(config, seed=9, preset="typical_day", start_interval=40)
    w.inject(EventType.LINE_CONGESTION, corridor="north", duration=6)
    snap = Normalizer(config).build(w.telemetry())
    scen = build_scenarios(config, snap, seed=2, n_scenarios=2)
    g = config.portfolio.grid
    assert scen.export_limit_mw[0] == pytest.approx(g.export_derated_mw)
    assert scen.corridor_limit_mw["north"][0] == pytest.approx(g.north_corridor_derated_mw)
    assert scen.export_limit_mw[-1] == pytest.approx(g.export_limit_mw)
    assert scen.corridor_limit_mw["north"][-1] == pytest.approx(g.north_corridor_limit_mw)


def test_dr_capacity_respects_the_daily_contract_cap(config):
    """PRD fairness control: "respect contract caps per day and month"."""
    snap = day(config, "typical_day", interval=70)
    scen = build_scenarios(config, snap, seed=1)
    assert scen.dr_capacity_mw == pytest.approx(config.portfolio.total_dr_mw)

    import dataclasses
    exhausted = dataclasses.replace(
        snap, dr_calls_today=int(config.market.demand_response["max_calls_per_day"]))
    scen2 = build_scenarios(config, exhausted, seed=1)
    assert scen2.dr_capacity_mw == 0.0, "DR offered beyond the contracted daily cap"


# ---- the forecaster as the substitute for a dead feed ----------------------
def test_forecast_substitution_replaces_a_dead_feed(config, forecaster):
    """PRD: stale or missing inputs are "replaced by the forecast", not zeroed."""
    w = World(config, seed=3, preset="typical_day", start_interval=40)
    holder = {}

    def substitute(channel, interval, last_good):
        return forecaster.substitute(holder.get("snap"), channel, interval, last_good)

    n = Normalizer(config, substitute=substitute)
    holder["snap"] = n.build(w.telemetry())
    good = holder["snap"]
    assert good.total_solar_mw > 0

    w.inject(EventType.DATA_DROPOUT, channels=("price",), duration=8)
    degraded = n.build(w.telemetry())
    assert "price" in degraded.quality.substituted_fields
    assert degraded.price_per_mwh != 0.0, "a dead price feed must never read as zero"
    assert np.isfinite(degraded.price_per_mwh)


def test_envelope_feeds_the_optimizer_end_to_end(config, forecaster):
    """The whole M1->M3 chain must produce a scoreable fan."""
    snap = day(config, "volatile_weather", interval=48)
    env = forecaster.envelope(snap, config.objectives.steps)
    scen = build_scenarios(config, snap, seed=7, envelope=env)
    out = DispatchEngine(config).simulate(
        scen, snap.battery_soc_pct, snap.battery_available)
    for key, arr in out.objectives.items():
        assert np.isfinite(arr).all(), f"{key} non-finite with a calibrated envelope"
    assert arr.shape == (80, config.objectives.scenarios)


def test_stored_energy_price_values_a_day_not_the_look_ahead(config, forecaster):
    """PRD: "Energy left in batteries at horizon end is valued at upper-quartile
    next-24h price x 0.93 -- without this the short horizon drains batteries before the
    evening peak."

    The look-ahead is only 4 h, and at midday those are the cheapest hours of the day. If
    the estimate is dominated by them, stored energy is valued at roughly spot, charging
    cannot pay for its own wear, and the agent exports the midday surplus instead of
    saving it for the peak -- exactly the failure the clause exists to prevent. So the
    figure must still describe a day once the day's own prices are known.
    """
    snaps = day(config, "typical_day", seed=61)
    haircut = config.objectives.stored_energy["efficiency_haircut"]
    for interval in (36, 44, 48, 52):            # 09:00 to 13:00: the surplus window
        snap = snaps[interval]
        scen = build_scenarios(config, snap, seed=1234,
                               envelope=forecaster.envelope(snap, config.objectives.steps))
        observed = float(np.median(snap.price_history))
        assert scen.price_quantile_next24 >= observed, (
            f"interval {interval}: the day's upper-quartile price came out at "
            f"${scen.price_quantile_next24:.2f}/MWh, below the ${observed:.2f}/MWh the "
            f"day has already traded at -- the cheap look-ahead is dominating it")

        wear = min(b.wear_cost_per_mwh for b in config.portfolio.batteries)
        stored_value = scen.price_quantile_next24 * haircut
        assert stored_value > snap.price_per_mwh + wear, (
            f"interval {interval}: storing a MWh is worth ${stored_value:.2f} against "
            f"${snap.price_per_mwh + wear:.2f} to buy and wear it, so the agent cannot "
            f"fill batteries from the midday surplus at all")


def test_the_charge_band_never_invites_a_loss_making_purchase(config, forecaster):
    """"Charge below this" has to mean cheap by the day's standard.

    The band is the 30th percentile of the look-ahead, and roughly 30% of any window
    sits below its own 30th percentile -- including a window of uniformly expensive
    evening hours. So the signal fires on "cheapest of an expensive afternoon" and the
    agent buys grid energy at a price above what storing it is worth, paying the spread
    and the wear on top and emitting 0.42 t/MWh for the privilege.

    The weakest form of the requirement is asserted: the band must never invite buying a
    MWh for more than a stored MWh is worth. Breaking even actually needs the spread and
    the wear covered too, so this leaves room rather than prescribing the threshold.
    """
    haircut = config.objectives.stored_energy["efficiency_haircut"]
    for preset in ("storm_passage", "evening_price_crisis", "typical_day"):
        snaps = day(config, preset, seed=5)
        for interval in range(0, INTERVALS_PER_DAY, 4):
            snap = snaps[interval]
            scen = build_scenarios(config, snap, seed=99,
                                   envelope=forecaster.envelope(snap, config.objectives.steps))
            stored_value = scen.price_quantile_next24 * haircut
            worst = float(np.max(scen.arbitrage_low_per_mwh))
            assert worst <= stored_value + 1e-6, (
                f"{preset} interval {interval}: the charge band reaches "
                f"${worst:.2f}/MWh while a stored MWh is worth ${stored_value:.2f} -- "
                f"buying anywhere in that range loses money and emits carbon")
