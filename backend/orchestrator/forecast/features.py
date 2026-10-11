"""Feature construction for the quantile forecasters.

Direct multi-horizon: lead time is a feature, so one model per (target, quantile)
covers the whole 16-step look-ahead rather than 16 separate models.
"""
from __future__ import annotations

import numpy as np

from ..domain.state import StateSnapshot
from ..world.drivers import MINUTES_PER_DAY, clear_sky_fraction, demand_shape

TARGETS = ("solar", "wind", "demand", "price")

FEATURE_NAMES = (
    "lead",
    "sin_tod_at_lead",
    "cos_tod_at_lead",
    "clear_sky_at_lead",
    "demand_shape_at_lead",
    "value_now",
    "value_lag_1",
    "value_lag_4",
    "value_roll_4",
    "cloud_mean_now",
    "cloud_spread_now",
    "wind_speed_now",
    "demand_now",
    "price_now",
    "net_load_now",
)
N_FEATURES = len(FEATURE_NAMES)


def target_value(snapshot: StateSnapshot, target: str) -> float:
    if target == "solar":
        return snapshot.total_solar_mw
    if target == "wind":
        return snapshot.total_wind_mw
    if target == "demand":
        return snapshot.total_demand_mw
    if target == "price":
        return snapshot.price_per_mwh
    raise KeyError(target)


def build_row(
    snapshot: StateSnapshot,
    target: str,
    lead: int,
    *,
    history: list[float] | None = None,
    interval_minutes: int = 15,
) -> np.ndarray:
    """One feature row: predict `target` `lead` steps ahead from `snapshot`."""
    hist = history or []
    now = target_value(snapshot, target)
    lag1 = hist[-1] if len(hist) >= 1 else now
    lag4 = hist[-4] if len(hist) >= 4 else now
    roll4 = float(np.mean(hist[-4:])) if hist else now

    minute_at_lead = (snapshot.minute_of_day + lead * interval_minutes) % MINUTES_PER_DAY
    angle = 2.0 * np.pi * minute_at_lead / MINUTES_PER_DAY
    clouds = list(snapshot.cloud_cover.values()) or [0.0]
    speeds = list(snapshot.wind_speed_ms.values()) or [0.0]

    return np.array([
        float(lead),
        np.sin(angle),
        np.cos(angle),
        float(clear_sky_fraction(minute_at_lead)),
        float(demand_shape(minute_at_lead)),
        now,
        lag1,
        lag4,
        roll4,
        float(np.mean(clouds)),
        float(np.max(clouds) - np.min(clouds)),
        float(np.mean(speeds)),
        snapshot.total_demand_mw,
        snapshot.price_per_mwh,
        snapshot.net_load_mw,
    ], dtype=float)


def build_matrix(
    snapshot: StateSnapshot,
    target: str,
    leads: range | list[int],
    *,
    history: list[float] | None = None,
    interval_minutes: int = 15,
) -> np.ndarray:
    return np.vstack([
        build_row(snapshot, target, lead, history=history,
                  interval_minutes=interval_minutes)
        for lead in leads
    ])
