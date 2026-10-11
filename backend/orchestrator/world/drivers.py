"""Mean-reverting stochastic drivers.

Scenario generation (PRD "Uncertainty handling") evolves the same underlying drivers
as mean-reverting processes from the current measured state, so the world model and
the scenario model share this code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MINUTES_PER_DAY = 1440


@dataclass
class OUParams:
    """Ornstein-Uhlenbeck-style discrete mean reversion."""

    reversion: float      # 0..1 pull toward the mean each step
    volatility: float     # std-dev of the per-step shock
    lo: float
    hi: float


def ou_step(
    value: np.ndarray | float,
    mean: np.ndarray | float,
    params: OUParams,
    shock: np.ndarray | float,
) -> np.ndarray | float:
    nxt = value + params.reversion * (mean - value) + params.volatility * shock
    return np.clip(nxt, params.lo, params.hi)


def clear_sky_fraction(minute_of_day: int | np.ndarray) -> np.ndarray | float:
    """Clear-sky solar availability, 0..1. Sunrise 06:00, sunset 19:00."""
    sunrise, sunset = 360.0, 1140.0
    m = np.asarray(minute_of_day, dtype=float) % MINUTES_PER_DAY
    inside = (m >= sunrise) & (m <= sunset)
    phase = np.pi * (m - sunrise) / (sunset - sunrise)
    val = np.where(inside, np.sin(np.clip(phase, 0.0, np.pi)) ** 1.15, 0.0)
    return val if np.ndim(minute_of_day) else float(val)


def demand_shape(minute_of_day: int | np.ndarray) -> np.ndarray | float:
    """Municipal/commercial diurnal shape, 0..1, with a morning and evening peak."""
    m = np.asarray(minute_of_day, dtype=float) % MINUTES_PER_DAY
    h = m / 60.0
    night = 0.30
    morning = 0.55 * np.exp(-0.5 * ((h - 8.5) / 2.1) ** 2)
    evening = 0.95 * np.exp(-0.5 * ((h - 19.0) / 2.4) ** 2)
    midday = 0.35 * np.exp(-0.5 * ((h - 13.5) / 3.2) ** 2)
    val = np.clip(night + morning + evening + midday, 0.0, 1.0)
    return val if np.ndim(minute_of_day) else float(val)


def price_from_residual(
    residual_fraction: np.ndarray | float,
    typical: float,
    minute_of_day: int | np.ndarray,
) -> np.ndarray | float:
    """Real-time price as a convex function of residual load.

    `residual_fraction` is (demand - renewables) / import_limit, so it is negative
    under surplus and above 1.0 when the tie-line alone cannot cover the deficit.
    Prices go negative under heavy surplus and rise steeply into scarcity, which is
    what makes arbitrage and curtailment decisions non-trivial.
    """
    r = np.asarray(residual_fraction, dtype=float)
    h = (np.asarray(minute_of_day, dtype=float) % MINUTES_PER_DAY) / 60.0
    evening = 1.0 + 0.35 * np.exp(-0.5 * ((h - 19.0) / 2.0) ** 2)
    base = typical * evening * (0.45 + 0.85 * np.clip(r, -1.0, 3.0))
    scarcity = typical * 4.0 * np.clip(r - 0.85, 0.0, None) ** 2
    val = base + scarcity
    return val if np.ndim(residual_fraction) else float(val)
