"""Gradient-boosted quantile regression: P10/P50/P90 for solar, wind, demand, price.

PRD: "Accurate on tabular weather data, fast to retrain nightly, gives calibrated
uncertainty." Retrained by `forecast.train`; persisted so startup is instant.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from ..config_loader import Config
from ..domain.state import StateSnapshot
from ..world.drivers import clear_sky_fraction
from .features import N_FEATURES, TARGETS, build_matrix, target_value
from .scenarios import Envelope

QUANTILES = (0.10, 0.50, 0.90)
ARTIFACT = Path(__file__).resolve().parent.parent.parent / "artifacts" / "forecasters.pkl"

# Physical bounds so a forecast can never be unphysical, whatever the model says.
@dataclass
class Bounds:
    lo: float
    hi: float


@dataclass
class QuantileForecaster:
    """One model per (target, quantile). Lead time is a feature."""

    models: dict[tuple[str, float], GradientBoostingRegressor] = field(default_factory=dict)
    bounds: dict[str, Bounds] = field(default_factory=dict)
    # Per-target, per-lead band-width multipliers fitted on held-out data so the
    # empirical P10-P90 coverage actually lands on 80%. The PRD lists "forecast
    # models under-state uncertainty" as a top risk, with "calibrate scenario spread
    # against historical forecast errors" as the mitigation -- this is that step.
    calibration: dict[str, np.ndarray] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    degenerate_skipped: dict[str, int] = field(default_factory=dict)
    solar_floor_fraction: float | None = None   # 1 - max cloud attenuation
    trained_rows: int = 0
    version: str = "untrained"

    @property
    def is_trained(self) -> bool:
        return bool(self.models)

    # ---- training ---------------------------------------------------------
    def fit(self, data: dict[str, tuple[np.ndarray, np.ndarray]], *, version: str) -> None:
        for target, (X, y) in data.items():
            if X.shape[1] != N_FEATURES:
                raise ValueError(f"{target}: expected {N_FEATURES} features, got {X.shape[1]}")
            for q in QUANTILES:
                model = GradientBoostingRegressor(
                    loss="quantile", alpha=q,
                    n_estimators=90, max_depth=3, learning_rate=0.1,
                    subsample=0.85, random_state=0,
                )
                model.fit(X, y)
                self.models[(target, q)] = model
            self.trained_rows = max(self.trained_rows, int(X.shape[0]))
        self.version = version

    # ---- prediction -------------------------------------------------------
    def predict(
        self, snapshot: StateSnapshot, target: str, n_steps: int,
        *, history: list[float] | None = None, interval_minutes: int = 15,
    ) -> np.ndarray:
        """Returns (3, n_steps): p10, p50, p90. Monotone and inside physical bounds."""
        # Step 0 is the measured present, so it is forecast for leads 1..n_steps-1
        # and pinned to the observation at lead 0.
        leads = list(range(n_steps))
        X = build_matrix(snapshot, target, leads, history=history,
                         interval_minutes=interval_minutes)
        rows = []
        for q in QUANTILES:
            model = self.models.get((target, q))
            if model is None:
                raise RuntimeError(f"no model for {target} q={q}; train first")
            rows.append(model.predict(X))
        band = np.vstack(rows)

        # Quantile regressions are fitted independently and can cross; sorting is the
        # standard repair and preserves the median.
        band = np.sort(band, axis=0)

        scale = self.calibration.get(target)
        if scale is not None and scale.size:
            s = scale[np.minimum(np.arange(band.shape[1]), scale.size - 1)]
            median = band[1]
            band = np.vstack([
                median - s * (median - band[0]),
                median,
                median + s * (band[2] - median),
            ])

        b = self.bounds.get(target)
        if b is not None:
            band = np.clip(band, b.lo, b.hi)

        # Solar has a per-step physical floor and ceiling: clear-sky availability sets
        # the ceiling, and cloud attenuation is bounded, so daylight output cannot
        # collapse to zero. Without this the P10 model predicts 0 MW at midday, which
        # is unreachable and makes the agent hold reserve against a phantom risk.
        if target == "solar" and self.solar_floor_fraction is not None:
            leads = np.arange(band.shape[1])
            minute = (snapshot.minute_of_day + leads * interval_minutes) % 1440
            ceiling = np.asarray(clear_sky_fraction(minute)) * (b.hi if b else 1.0)
            floor = ceiling * self.solar_floor_fraction
            band = np.clip(band, floor, ceiling)
            band = np.sort(band, axis=0)

        observed = target_value(snapshot, target)
        band[:, 0] = observed      # the present is measured, not predicted
        return band

    def fit_calibration(
        self, samples: dict[str, list[tuple[np.ndarray, float, int]]],
        *, coverage_target: float = 0.80, lo: float = 1.0, hi: float = 5.0,
        min_width_fraction: float = 0.002,
    ) -> None:
        """Fit the band-width multipliers from held-out (band, actual, lead) samples.

        For each sample the minimal scale that would have covered it is computed, and
        the multiplier is the `coverage_target` quantile of those -- which makes the
        empirical coverage land on the target by construction.

        Samples whose band is already effectively zero-width are excluded: at night
        the solar forecast is a trivially-correct zero, and those intervals made up a
        third of the sample and pinned every quantile at exactly 1.0. There is no
        uncertainty there to calibrate.

        `lo` defaults to 1.0, so calibration only ever *widens* a band. Letting it
        sharpen an over-covered target looked attractive -- solar went 90% -> 81% on
        the calibration seeds -- but it is fitted in-sample, and on seeds used for
        neither training nor calibration the sharpened solar band collapsed to 67%
        coverage. Under-stating uncertainty is the failure mode the PRD singles out,
        so an over-wide band is the safe side to err on.
        """
        for target, rows in samples.items():
            if not rows:
                continue
            b = self.bounds.get(target)
            min_width = min_width_fraction * (b.hi - b.lo) if b else 0.0
            n_leads = max(lead for _, _, lead in rows) + 1
            needed: list[list[float]] = [[] for _ in range(n_leads)]
            skipped = 0
            for band, actual, lead in rows:
                p10, p50, p90 = band
                if (p90 - p10) < min_width:
                    skipped += 1
                    continue
                if actual >= p50:
                    side = max(p90 - p50, 1e-9)
                    needed[lead].append((actual - p50) / side)
                else:
                    side = max(p50 - p10, 1e-9)
                    needed[lead].append((p50 - actual) / side)
            self.degenerate_skipped[target] = skipped
            scale = np.ones(n_leads)
            for lead, vals in enumerate(needed):
                if vals:
                    scale[lead] = float(np.clip(
                        np.quantile(vals, coverage_target), lo, hi))
            # Uncertainty must not shrink with lead time; enforce monotonicity after
            # lead 1 so a noisy bucket cannot invert the fan.
            for lead in range(2, n_leads):
                scale[lead] = max(scale[lead], scale[lead - 1] * 0.95)
            self.calibration[target] = scale

    def envelope(
        self, snapshot: StateSnapshot, n_steps: int,
        *, history: dict[str, list[float]] | None = None, interval_minutes: int = 15,
    ) -> Envelope:
        hist = history or {}
        kw = dict(interval_minutes=interval_minutes)
        return Envelope(
            solar=self.predict(snapshot, "solar", n_steps,
                               history=hist.get("solar"), **kw),
            wind=self.predict(snapshot, "wind", n_steps,
                              history=hist.get("wind"), **kw),
            demand=self.predict(snapshot, "demand", n_steps,
                                history=hist.get("demand"), **kw),
            price=self.predict(snapshot, "price", n_steps,
                               history=hist.get("price"), **kw),
        )

    # ---- forecast substitution for missing feeds --------------------------
    def substitute(self, snapshot: StateSnapshot | None, channel: str,
                   interval: int, last_good):
        """Adapter for `ingest.Normalizer`: replace a dead feed with the forecast.

        Falls back to last-good persistence when the channel has no model, which is
        still a forecast in the degenerate sense -- and never a silent zero.
        """
        if snapshot is None or not self.is_trained:
            return last_good
        mapping = {"solar": "solar", "wind": "wind", "demand": "demand", "price": "price"}
        target = mapping.get(channel)
        if target is None:
            return last_good
        median = float(self.predict(snapshot, target, 2)[1, 1])
        if channel == "price":
            return median
        if channel in ("solar", "wind") and isinstance(last_good, dict):
            total = sum(last_good.values()) or 1.0
            return {k: v / total * median for k, v in last_good.items()}
        return last_good

    # ---- persistence ------------------------------------------------------
    def save(self, path: Path = ARTIFACT) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: Path = ARTIFACT) -> "QuantileForecaster | None":
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                obj = pickle.load(fh)
            return obj if isinstance(obj, QuantileForecaster) else None
        except Exception:
            return None


def physical_bounds(config: Config) -> dict[str, Bounds]:
    p = config.portfolio
    m = config.market
    peak_load = sum(c.base_mw for c in p.consumers) + p.municipal.max_mw
    return {
        "solar": Bounds(0.0, p.solar_capacity_mw),
        "wind": Bounds(0.0, p.wind_capacity_mw),
        "demand": Bounds(0.0, peak_load * 1.5),
        "price": Bounds(m.price["floor_per_mwh"], m.price["cap_per_mwh"]),
    }
