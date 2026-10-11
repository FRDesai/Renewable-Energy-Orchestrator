"""Nightly retrain of the quantile forecasters on new actuals (PRD: Trainability).

Training data comes from running the world model across presets and seeds, which is
the stand-in for "new actuals" until real telemetry history exists.
"""
from __future__ import annotations

import numpy as np

from ..config_loader import Config, load_config
from ..ingest.normalize import Normalizer
from ..world.presets import PRESETS
from ..world.world import INTERVALS_PER_DAY, World
from .features import TARGETS, build_row, target_value
from .quantile import QuantileForecaster, physical_bounds


def collect(
    config: Config,
    *,
    seeds: tuple[int, ...] = (1, 2, 3, 4),
    presets: tuple[str, ...] | None = None,
    n_steps: int | None = None,
    leads: tuple[int, ...] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build (features, target) pairs for each target.

    For every interval t and lead L we emit a row whose features are known at t and
    whose label is the actual value at t+L -- so the model learns the forecast task,
    not an interpolation.
    """
    presets = presets or tuple(sorted(PRESETS))
    T = int(n_steps or config.objectives.steps)
    lead_set = leads or tuple(range(1, T))
    rows: dict[str, list[np.ndarray]] = {t: [] for t in TARGETS}
    labels: dict[str, list[float]] = {t: [] for t in TARGETS}
    interval_minutes = config.portfolio.interval_minutes

    for preset in presets:
        for seed in seeds:
            w = World(config, seed=seed, preset=preset)
            n = Normalizer(config)
            snaps = []
            for _ in range(INTERVALS_PER_DAY):
                snaps.append(n.build(w.telemetry()))
                w.step()
            series = {t: [target_value(s, t) for s in snaps] for t in TARGETS}
            for i, snap in enumerate(snaps):
                for target in TARGETS:
                    hist = series[target][max(0, i - 4):i]
                    for lead in lead_set:
                        j = i + lead
                        if j >= len(snaps):
                            continue
                        rows[target].append(
                            build_row(snap, target, lead, history=list(hist),
                                      interval_minutes=interval_minutes))
                        labels[target].append(series[target][j])

    return {
        t: (np.vstack(rows[t]), np.asarray(labels[t], dtype=float))
        for t in TARGETS if rows[t]
    }


def calibration_samples(
    config: Config, forecaster: QuantileForecaster,
    *, seeds: tuple[int, ...], presets: tuple[str, ...] | None = None, stride: int = 3,
) -> dict[str, list[tuple[np.ndarray, float, int]]]:
    """Collect (band, actual, lead) triples on seeds the model never trained on."""
    presets = presets or tuple(sorted(PRESETS))
    T = config.objectives.steps
    out: dict[str, list[tuple[np.ndarray, float, int]]] = {t: [] for t in TARGETS}
    for preset in presets:
        for seed in seeds:
            w = World(config, seed=seed, preset=preset)
            n = Normalizer(config)
            snaps = []
            for _ in range(INTERVALS_PER_DAY):
                snaps.append(n.build(w.telemetry()))
                w.step()
            series = {t: [target_value(s, t) for s in snaps] for t in TARGETS}
            for i in range(0, len(snaps) - T, stride):
                for target in TARGETS:
                    hist = series[target][max(0, i - 4):i]
                    band = forecaster.predict(snaps[i], target, T, history=list(hist))
                    for lead in range(1, T):
                        out[target].append(
                            (band[:, lead].copy(), series[target][i + lead], lead))
    return out


def measure_coverage(samples, forecaster=None, *,
                     min_width_fraction: float = 0.002) -> dict[str, float]:
    """Empirical P10-P90 coverage. Degenerate (zero-width) bands are excluded for the
    same reason they are excluded from calibration -- a trivially-correct zero carries
    no information about whether the uncertainty estimate is honest."""
    result = {}
    for target, rows in samples.items():
        if not rows:
            continue
        min_width = 0.0
        if forecaster is not None:
            b = forecaster.bounds.get(target)
            if b is not None:
                min_width = min_width_fraction * (b.hi - b.lo)
        hits = [1.0 if band[0] - 1e-9 <= actual <= band[2] + 1e-9 else 0.0
                for band, actual, _ in rows if (band[2] - band[0]) >= min_width]
        if hits:
            result[target] = float(np.mean(hits))
    return result


def train(config: Config | None = None, *, save: bool = True,
          seeds: tuple[int, ...] = (1, 2, 3, 4),
          calibration_seeds: tuple[int, ...] = (31, 32),
          verbose: bool = False) -> QuantileForecaster:
    config = config or load_config()
    data = collect(config, seeds=seeds)
    f = QuantileForecaster()
    f.bounds = physical_bounds(config)
    f.solar_floor_fraction = 1.0 - config.portfolio.solar_response.max_cloud_attenuation
    version = f"gbqr-{len(seeds)}seeds-{sum(len(y) for _, y in data.values())}rows"
    f.fit(data, version=version)

    # Calibrate the band widths on seeds the model has never seen, then re-measure.
    raw = calibration_samples(config, f, seeds=calibration_seeds)
    before = measure_coverage(raw, f)
    f.fit_calibration(raw)
    after_samples = calibration_samples(config, f, seeds=calibration_seeds)
    f.coverage = measure_coverage(after_samples, f)
    if verbose:
        for t in sorted(before):
            print(f"  {t:8s} coverage {100*before[t]:5.1f}% -> "
                  f"{100*f.coverage.get(t, 0):5.1f}%  "
                  f"(scale {f.calibration[t].min():.2f}-{f.calibration[t].max():.2f})")
    if save:
        f.save()
    return f


def load_or_train(config: Config | None = None) -> QuantileForecaster:
    """Used at startup: load the persisted model, training it if absent."""
    existing = QuantileForecaster.load()
    if existing is not None and existing.is_trained:
        return existing
    return train(config)


if __name__ == "__main__":
    import time
    cfg = load_config()
    t0 = time.perf_counter()
    f = train(cfg, verbose=True)
    print(f"trained {len(f.models)} models on {f.trained_rows:,} rows "
          f"in {time.perf_counter() - t0:.1f}s -> {f.version}")
