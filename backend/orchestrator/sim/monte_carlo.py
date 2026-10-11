"""F3: 24-hour Monte Carlo over the same engine, in batch mode.

PRD F3.2: "N selectable (6/12/24); same random numbers across strategies for a fair
comparison." The common-random-numbers property is what makes the KPI table a real
comparison rather than three unrelated runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config_loader import Config
from ..domain.events import EventType, Severity
from ..forecast.quantile import QuantileForecaster
from ..world.presets import PRESETS
from .episode import STRATEGIES, EpisodeResult, EpisodeRunner

PATH_CHOICES = (6, 12, 24)
INTERVALS_PER_DAY = 96

# Per-preset extra disturbances layered on top of the preset's own schedule, so each
# path differs in event *timing* as well as weather noise (PRD F3.2).
JITTERABLE = {
    "typical_day": (EventType.CLOUD_BANK, dict(magnitude=0.8, duration=6)),
    "volatile_weather": (EventType.PRICE_SPIKE, dict(magnitude=4.0, duration=5)),
    "storm_passage": (EventType.FREQUENCY_DIP, dict(magnitude=0.25, duration=2)),
    "evening_price_crisis": (EventType.DEMAND_SURGE, dict(magnitude=1.15, duration=8)),
    "contingency_cascade": (EventType.BATTERY_TRIP, dict(assets=("valley",), duration=12)),
}


@dataclass
class StrategyKPIs:
    strategy: str
    runs: list[EpisodeResult] = field(default_factory=list)

    def series(self, field_name: str) -> np.ndarray:
        return np.array([getattr(r.kpis, field_name) for r in self.runs], dtype=float)

    def band(self, field_name: str) -> dict[str, float]:
        v = self.series(field_name)
        return {
            "mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
            "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
        }

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"strategy": self.strategy, "paths": len(self.runs)}
        for name in ("energy_cost_usd", "net_cost_usd", "carbon_tonnes", "unserved_mwh",
                     "avoidable_unserved_mwh", "curtailed_mwh", "throughput_mwh",
                     "wear_usd"):
            out[name] = {k: round(v, 2) for k, v in self.band(name).items()}
        for name in ("renewable_utilisation", "reserve_compliance"):
            v = np.array([getattr(r.kpis, name) for r in self.runs], dtype=float)
            out[name] = {"mean": round(float(v.mean()), 4),
                         "p10": round(float(np.percentile(v, 10)), 4),
                         "p90": round(float(np.percentile(v, 90)), 4)}
        return out

    def fan(self, attr: str) -> dict[str, list[float]]:
        """Per-interval P10/P50/P90 across paths, for the fan charts."""
        mat = np.array([getattr(r, attr) for r in self.runs], dtype=float)
        return {
            "p10": np.percentile(mat, 10, axis=0).round(3).tolist(),
            "p50": np.percentile(mat, 50, axis=0).round(3).tolist(),
            "p90": np.percentile(mat, 90, axis=0).round(3).tolist(),
        }

    def strategy_mix_by_hour(self, config: Config) -> list[dict[str, float]]:
        """PRD F3.4: "the agent's strategy mix by hour"."""
        from ..domain.actions import CLUSTERS
        per_hour = int(60 / config.portfolio.interval_minutes)
        n = min(len(r.chosen_indices) for r in self.runs)
        rows: list[dict[str, float]] = []
        for h in range(0, n, per_hour):
            counts: dict[str, int] = {}
            total = 0
            for r in self.runs:
                for idx in r.chosen_indices[h:h + per_hour]:
                    key = CLUSTERS[idx].battery.label
                    counts[key] = counts.get(key, 0) + 1
                    total += 1
            rows.append({"hour": h // per_hour,
                         **{k: round(v / max(total, 1), 4) for k, v in counts.items()}})
        return rows


@dataclass
class SimulationResult:
    preset: str
    paths: int
    intervals: int
    seeds: list[int]
    by_strategy: dict[str, StrategyKPIs]

    def kpi_table(self) -> list[dict[str, object]]:
        return [self.by_strategy[s].as_dict() for s in STRATEGIES
                if s in self.by_strategy]

    def comparison(self) -> dict[str, object]:
        agent = self.by_strategy["agent"]
        out: dict[str, object] = {}
        for other in ("rule_based", "no_storage"):
            if other not in self.by_strategy:
                continue
            o = self.by_strategy[other]
            a_cost = agent.series("energy_cost_usd")
            o_cost = o.series("energy_cost_usd")
            a_shed = agent.series("unserved_mwh")
            o_shed = o.series("unserved_mwh")
            a_co2 = agent.series("carbon_tonnes")
            o_co2 = o.series("carbon_tonnes")
            out[other] = {
                "cost_delta_pct": round(
                    100.0 * (a_cost.mean() - o_cost.mean()) / max(abs(o_cost.mean()), 1e-9), 1),
                "net_cost_delta_pct": round(
                    100.0 * (agent.series("net_cost_usd").mean() - o.series("net_cost_usd").mean())
                    / max(abs(o.series("net_cost_usd").mean()), 1e-9), 1),
                "shed_delta_pct": round(
                    100.0 * (a_shed.mean() - o_shed.mean()) / max(o_shed.mean(), 1e-9), 1)
                    if o_shed.mean() > 1e-9 else 0.0,
                "co2_delta_pct": round(
                    100.0 * (a_co2.mean() - o_co2.mean()) / max(abs(o_co2.mean()), 1e-9), 1),
                "win_rate_cost": round(float(np.mean(a_cost <= o_cost)), 3),
                "win_rate_net_cost": round(float(np.mean(
                    agent.series("net_cost_usd") <= o.series("net_cost_usd"))), 3),
            }
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset, "paths": self.paths, "intervals": self.intervals,
            "seeds": self.seeds,
            "kpis": self.kpi_table(),
            "comparison": self.comparison(),
            "fans": {
                s: {"cumulative_cost": k.fan("cost_path"),
                    "fleet_soc_pct": k.fan("soc_path"),
                    "price_per_mwh": k.fan("price_path")}
                for s, k in self.by_strategy.items()
            },
        }


class MonteCarlo:
    def __init__(self, config: Config, forecaster: QuantileForecaster | None = None) -> None:
        self.config = config
        self.runner = EpisodeRunner(config, forecaster)

    def run(
        self,
        *,
        preset: str = "typical_day",
        paths: int = 6,
        intervals: int = INTERVALS_PER_DAY,
        strategies: tuple[str, ...] = STRATEGIES,
        base_seed: int = 2024,
        progress=None,
    ) -> SimulationResult:
        if preset not in PRESETS:
            raise KeyError(f"unknown preset {preset!r}")
        if paths < 1:
            raise ValueError("paths must be >= 1")

        # Common random numbers: path p uses the same world seed and the same jittered
        # event timing for every strategy, so differences are attributable to the
        # strategy alone.
        seeds = [base_seed + p * 97 for p in range(paths)]
        plans = []
        for p, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            kind, spec = JITTERABLE[preset]
            at = int(rng.integers(8, max(9, intervals - 12)))
            plans.append([(at, kind, dict(spec))])

        by_strategy: dict[str, StrategyKPIs] = {}
        total = len(strategies) * paths
        done = 0
        for strategy in strategies:
            acc = StrategyKPIs(strategy)
            for p, seed in enumerate(seeds):
                acc.runs.append(self.runner.run(
                    strategy=strategy, preset=preset, seed=seed,
                    start_interval=0, intervals=intervals,
                    check_invariants=False, event_plan=plans[p]))
                done += 1
                if progress is not None:
                    progress(done, total, strategy, p)
            by_strategy[strategy] = acc

        return SimulationResult(preset=preset, paths=paths, intervals=intervals,
                                seeds=seeds, by_strategy=by_strategy)
