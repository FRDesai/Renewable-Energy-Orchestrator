"""Scenario generation (PRD "Uncertainty handling").

The ML quantile forecasts set the P10-P90 envelope per step. Within it, each scenario
evolves the underlying drivers -- cloud cover per region, wind speed per corridor,
demand noise, price noise -- as mean-reverting processes from the current measured
state. Scenario 0 is the expected path. Spread grows with lead time and with a
volatility factor that rises in unsettled weather.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config_loader import Config
from ..decide.dispatch import ScenarioSet
from ..domain.events import EventType
from ..domain.state import StateSnapshot
from ..world.drivers import MINUTES_PER_DAY, clear_sky_fraction, demand_shape, price_from_residual


@dataclass
class Envelope:
    """P10/P50/P90 bands from the quantile forecasters, per step.

    Supplied by `forecast.quantile` in v1. When absent the generator falls back to
    its own driver model, which is what the fallback controller uses.
    """

    solar: np.ndarray | None = None       # (3, T): p10, p50, p90 total MW
    wind: np.ndarray | None = None        # (3, T)
    demand: np.ndarray | None = None      # (3, T)
    price: np.ndarray | None = None       # (3, T)


def _calibrate(paths: np.ndarray, band: np.ndarray | None) -> np.ndarray:
    """Re-centre and re-scale driver-generated paths onto a forecast band.

    `paths` is (S, T); `band` is (3, T) of p10/p50/p90. Scenario 0 is pinned to the
    median, and the ensemble spread is scaled so its own 10-90 width matches the
    forecast's. This is what makes the fan "consistent with the quantiles" rather
    than merely plausible.
    """
    if band is None:
        return paths
    p10, p50, p90 = band[0], band[1], band[2]
    own_p10 = np.percentile(paths, 10, axis=0)
    own_p90 = np.percentile(paths, 90, axis=0)
    own_med = np.median(paths, axis=0)
    own_width = np.maximum(own_p90 - own_p10, 1e-6)
    want_width = np.maximum(p90 - p10, 0.0)
    scale = want_width / own_width
    return p50 + (paths - own_med) * scale


def _day_price_view(config: Config, snapshot: StateSnapshot,
                    price: np.ndarray) -> np.ndarray:
    """A 24-hour price path per scenario, shaped (S, intervals_per_day).

    Three signals are *day-scale* by definition: what a MWh left in storage is worth,
    and the prices below and above which charging and discharging make sense. All three
    were being read off the 4-hour look-ahead, which is not a day and is systematically
    unrepresentative of one — midday is the cheapest stretch there is and an evening peak
    the most expensive. Read from the window, the stored-energy value collapsed at midday
    (so the surplus was exported instead of saved, the very failure the PRD's clause
    exists to prevent) and the charge band rose to the cheapest hour of an expensive
    evening (so grid energy was bought above what storing it was worth).

    The day is therefore assembled at one value per interval: each scenario's own
    expected path for the intervals the look-ahead covers, and the rolling price window
    for the rest. Intervals with neither — early in a session, before a day of history
    exists — take the configured typical price, which is the stated prior for a day
    nobody has observed yet rather than a fill-in from the cheap hours.
    """
    intervals_per_day = int(round(24 * 60 / config.portfolio.interval_minutes))
    ahead = np.atleast_2d(np.asarray(price, dtype=float))[:, :intervals_per_day]
    remaining = intervals_per_day - ahead.shape[1]
    if remaining <= 0:
        return ahead

    history = np.asarray(snapshot.price_history or (), dtype=float)
    if history.size >= remaining:
        tail = history[-remaining:]
    else:
        tail = np.concatenate([
            history,
            np.full(remaining - history.size,
                    config.market.price["typical_per_mwh"], dtype=float)])
    return np.concatenate(
        [ahead, np.broadcast_to(tail, (ahead.shape[0], tail.size))], axis=1)


def build_scenarios(
    config: Config,
    snapshot: StateSnapshot,
    *,
    n_scenarios: int | None = None,
    n_steps: int | None = None,
    seed: int = 0,
    envelope: Envelope | None = None,
    volatility: float = 1.0,
) -> ScenarioSet:
    p = config.portfolio
    o = config.objectives
    m = config.market
    S = int(n_scenarios or o.scenarios)
    T = int(n_steps or o.steps)
    rng = np.random.default_rng(seed)
    h = p.interval_hours
    step_minutes = p.interval_minutes

    # Step 0 is the CURRENT interval, not the next one: the receding-horizon plan
    # executes only its first step, so that step has to be the one being decided.
    # Every scenario therefore starts from the measured state with no shock.
    minutes = np.array([
        (snapshot.minute_of_day + k * step_minutes) % MINUTES_PER_DAY
        for k in range(T)
    ])
    intervals = np.array([snapshot.interval + k for k in range(T)])

    # Spread grows with lead time; a cloudy, gusty state is "unsettled weather".
    unsettled = 1.0 + 0.8 * float(np.mean(list(snapshot.cloud_cover.values())))
    vol = volatility * unsettled
    lead = np.sqrt(np.arange(1, T + 1)).reshape(1, T)

    # --- cloud cover per region -> solar per farm --------------------------
    solar_by_farm: dict[str, np.ndarray] = {}
    clear = clear_sky_fraction(minutes).reshape(1, T)
    region_paths: dict[str, np.ndarray] = {}
    for region, now in snapshot.cloud_cover.items():
        shocks = rng.standard_normal((S, T)) * 0.075 * vol * lead
        shocks[0, :] = 0.0                      # scenario 0 is the expected path
        shocks[:, 0] = 0.0                      # step 0 is measured, not forecast
        drift = now + np.cumsum(shocks, axis=1)
        region_paths[region] = np.clip(drift, 0.0, 1.0)
    for farm in p.solar_farms:
        cover = region_paths[farm.cloud_region]
        solar_by_farm[farm.id] = np.clip(
            farm.capacity_mw * p.solar_response.fraction(clear, cover),
            0.0, farm.capacity_mw)

    # --- wind speed per corridor -> wind per farm --------------------------
    storm = snapshot.storm_alert
    corridor_paths: dict[str, np.ndarray] = {}
    for corridor in p.corridors:
        farms = [f for f in p.wind_farms if f.corridor == corridor]
        now = float(np.mean([snapshot.wind_speed_ms[f.id] for f in farms]))
        shocks = rng.standard_normal((S, T)) * 0.95 * vol * lead
        shocks[0, :] = 0.0
        shocks[:, 0] = 0.0
        path = np.clip(now + np.cumsum(shocks, axis=1), 0.0, 40.0)
        if storm is not None and storm.arrival_interval is not None:
            # A storm raises wind once it is on site, with arrival jittered per
            # scenario (PRD: about +/- 0.6 h).
            jitter_h = o.storm["arrival_jitter_hours"]
            jitter = rng.normal(0.0, jitter_h, size=S) / h
            jitter[0] = 0.0
            arrive = storm.arrival_interval + jitter.reshape(S, 1)
            on_site = intervals.reshape(1, T) >= arrive
            path = np.where(on_site, np.minimum(40.0, path * 1.9), path)
        corridor_paths[corridor] = path

    wind_by_farm: dict[str, np.ndarray] = {}
    wind_corridor_of: dict[str, str] = {}
    curve = p.wind_curve
    for farm in p.wind_farms:
        speed = corridor_paths[farm.corridor]
        frac = np.where(
            (speed < curve.cut_in_ms) | (speed >= curve.cut_out_ms), 0.0,
            np.where(speed >= curve.rated_ms, 1.0,
                     ((speed - curve.cut_in_ms) / (curve.rated_ms - curve.cut_in_ms)) ** 3),
        )
        wind_by_farm[farm.id] = farm.capacity_mw * frac
        wind_corridor_of[farm.id] = farm.corridor

    # --- demand ------------------------------------------------------------
    shape_now = max(1e-6, float(demand_shape(snapshot.minute_of_day)))
    level = snapshot.total_demand_mw / shape_now
    shocks = rng.standard_normal((S, T)) * 0.016 * vol * lead
    shocks[0, :] = 0.0
    shocks[:, 0] = 0.0
    demand = np.maximum(0.0, level * demand_shape(minutes).reshape(1, T) * (1.0 + shocks))
    surge = snapshot.event_of(EventType.DEMAND_SURGE)
    if surge is not None:
        active = np.array([surge.active_at(int(i)) for i in intervals]).reshape(1, T)
        demand = np.where(active, demand * surge.magnitude, demand)

    # --- calibrate the weather/demand fan onto the forecast envelope -------
    if envelope is not None:
        solar_total = sum(solar_by_farm.values())
        tuned = _calibrate(solar_total, envelope.solar)
        ratio = np.divide(tuned, np.maximum(solar_total, 1e-9),
                          out=np.ones_like(tuned), where=solar_total > 1e-9)
        for fid in solar_by_farm:
            cap = p.solar(fid).capacity_mw
            solar_by_farm[fid] = np.clip(solar_by_farm[fid] * ratio, 0.0, cap)

        wind_total = sum(wind_by_farm.values())
        tuned = _calibrate(wind_total, envelope.wind)
        ratio = np.divide(tuned, np.maximum(wind_total, 1e-9),
                          out=np.ones_like(tuned), where=wind_total > 1e-9)
        for fid in wind_by_farm:
            cap = p.wind(fid).capacity_mw
            wind_by_farm[fid] = np.clip(wind_by_farm[fid] * ratio, 0.0, cap)

        demand = np.maximum(0.0, _calibrate(demand, envelope.demand))

    # Pin step 0 to the measurement, per farm. The envelope rescaling above works on
    # fleet totals, which leaves per-farm values off the observation by the telemetry
    # rounding -- and step 0 is the interval that actually gets committed.
    for fid, series in solar_by_farm.items():
        series[:, 0] = snapshot.solar_mw.get(fid, series[0, 0])
    for fid, series in wind_by_farm.items():
        series[:, 0] = snapshot.wind_mw.get(fid, series[0, 0])
    demand[:, 0] = snapshot.total_demand_mw

    renewable = sum(solar_by_farm.values()) + sum(wind_by_farm.values())

    # --- price -------------------------------------------------------------
    residual = (demand - renewable) / max(1.0, p.grid.import_limit_mw)
    price = price_from_residual(residual, m.price["typical_per_mwh"], minutes.reshape(1, T))
    pshocks = rng.standard_normal((S, T)) * 0.085 * vol * lead
    pshocks[0, :] = 0.0
    pshocks[:, 0] = 0.0
    price = price * (1.0 + pshocks)
    spike = snapshot.event_of(EventType.PRICE_SPIKE)
    if spike is not None:
        # An unannounced spike is seen once it starts and assumed to decay with a
        # typical persistence (PRD: 45 min).
        tau = max(1e-6, o.persistence_hours["price_spike"] / h)
        decay = np.exp(-np.arange(0, T) / tau).reshape(1, T)
        # Relax from the spiked scarcity level back to the modelled price. Only ever
        # upward, so a negative surplus price is not amplified.
        spiked = m.price["typical_per_mwh"] * spike.magnitude
        price = price + np.maximum(0.0, spiked - price) * decay
    if envelope is not None:
        price = _calibrate(price, envelope.price)
    price = np.clip(price, m.price["floor_per_mwh"], m.price["cap_per_mwh"])
    price[:, 0] = snapshot.price_per_mwh          # measured, not modelled

    # --- frequency ---------------------------------------------------------
    freq = np.full((S, T), p.grid.nominal_frequency_hz)
    dip = snapshot.event_of(EventType.FREQUENCY_DIP)
    if dip is not None:
        active = np.array([dip.active_at(int(i)) for i in intervals]).reshape(1, T)
        freq = np.where(active, p.grid.nominal_frequency_hz - dip.magnitude, freq)
    freq[:, 0] = snapshot.frequency_hz

    # --- announced limits (same in every scenario) -------------------------
    imp = np.full(T, snapshot.import_limit_mw)
    exp = np.full(T, snapshot.export_limit_mw)
    corridor_limit = {c: np.full(T, snapshot.corridor_limit_mw.get(c, 1e6))
                      for c in p.corridors}
    congestion = snapshot.event_of(EventType.LINE_CONGESTION)
    if congestion is not None and congestion.end_interval is not None:
        clears = intervals >= congestion.end_interval
        exp = np.where(clears, p.grid.export_limit_mw, exp)
        target = congestion.corridor or "north"
        if target in corridor_limit:
            corridor_limit[target] = np.where(
                clears, p.grid.north_corridor_limit_mw, corridor_limit[target])

    # --- DR / peak flags ---------------------------------------------------
    p70 = snapshot.price_percentile(o.thresholds["peak_price_percentile"])
    peak = price >= max(p70, 1e-9)
    iso = np.zeros((S, T), dtype=bool)
    call = snapshot.event_of(EventType.ISO_DR_CALL)
    if call is not None:
        active = np.array([call.active_at(int(i)) for i in intervals]).reshape(1, T)
        iso = np.broadcast_to(active, (S, T)).copy()

    # DR respects contract caps per day.
    caps_left = max(0, int(m.demand_response["max_calls_per_day"]) - snapshot.dr_calls_today)
    dr_capacity = p.total_dr_mw if caps_left > 0 else 0.0

    # --- storm pre-charge target (PRD defect fix, applied in M6) -----------
    storm_target = np.zeros(T)
    if storm is not None and storm.arrival_interval is not None:
        lim = p.battery_limits
        lead_h = (storm.arrival_interval - snapshot.interval) * h
        if 0 <= lead_h <= lim.storm_max_lead_hours:
            # Ramp the requirement so the fleet is at the guardrail level on arrival.
            span = max(1.0, storm.arrival_interval - snapshot.interval)
            frac = np.clip((intervals - snapshot.interval) / span, 0.0, 1.0)
            start = snapshot.fleet_soc_pct({b.id: b.energy_mwh for b in p.batteries})
            # Straight-line pre-charge ramp to the guardrail level at arrival. This is
            # the "pre-charge target to the Reserve strategy" the PRD's fix asks for,
            # and it is the same ramp the guardrail measures against.
            storm_target = start + (lim.storm_guardrail_pct - start) * frac
            storm_target = np.maximum(storm_target, 0.0)

    # --- arbitrage bands and stored-energy price ---------------------------
    # All three are day-scale signals, so all three come off the same 24 h view.
    day_price = _day_price_view(config, snapshot, price)
    price_q24 = float(np.percentile(
        np.median(day_price, axis=0),
        o.stored_energy["price_quantile"] * 100.0))

    # The two bands are not the same kind of question, so they are not read off the same
    # window. *Buying* is a day-scale decision -- is this genuinely cheap, and is a
    # stored MWh worth more than it costs? -- so the charge band comes off the 24 h view
    # and is then capped at what storage is worth, because on a flat price view the
    # day's 30th percentile still sits a round-trip loss above it. *Selling* is a
    # receding-horizon decision: discharge at the best moment inside the window actually
    # visible, which is what the look-ahead's upper band expresses. Making the sell band
    # day-scale as well suppressed discharging across three strategies and cost 54% more
    # over 30 episodes, so it stays window-local.
    lowq = np.minimum(np.percentile(day_price, 30, axis=1),
                      price_q24 * o.stored_energy["efficiency_haircut"])
    highq = np.percentile(price, 70, axis=1)

    return ScenarioSet(
        solar_by_farm=solar_by_farm,
        wind_by_farm=wind_by_farm,
        wind_corridor_of=wind_corridor_of,
        demand_mw=demand,
        price_per_mwh=price,
        frequency_hz=freq,
        import_limit_mw=imp,
        export_limit_mw=exp,
        corridor_limit_mw=corridor_limit,
        peak_price=peak,
        iso_dr_call=iso,
        dr_capacity_mw=dr_capacity,
        shift_capacity_mw=p.total_shiftable_mw,
        storm_soc_target_pct=storm_target,
        price_quantile_next24=price_q24,
        arbitrage_low_per_mwh=lowq,
        arbitrage_high_per_mwh=highq,
    )
