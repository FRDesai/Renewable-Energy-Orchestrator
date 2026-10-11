"""Nightly reliability run: 30 randomised 8-hour episodes = 960 decisions.

Matches the PRD's "Validation evidence" protocol: 30 randomised 8-hour episodes,
960 decisions, 2-4 random events each, on synthetic data -- plus the outcome
benchmark against rule-based dispatch.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.config_loader import load_config                  # noqa: E402
from orchestrator.domain.events import EventType, Severity          # noqa: E402
from orchestrator.forecast.quantile import QuantileForecaster       # noqa: E402
from orchestrator.invariants.checks import InvariantSuite           # noqa: E402
from orchestrator.sim.episode import EpisodeRunner                  # noqa: E402
from orchestrator.world.presets import PRESETS                      # noqa: E402

N_EPISODES = 30
INTERVALS = 32            # 8 hours
# Start intervals are randomised across the day. Fixing them at 09:00 produced
# episodes that never reached the evening peak, so reliability never had to pay for
# itself and the cost comparison was systematically unfavourable to the agent.
START_CHOICES = (8, 16, 24, 32, 40, 48, 56, 60)
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

INJECTABLE = [
    (EventType.CLOUD_BANK, lambda r: dict(region=r.choice(["region_a", "region_b", "region_c"]),
                                          magnitude=float(r.uniform(0.6, 0.95)), duration=int(r.integers(4, 10)))),
    (EventType.WIND_SURGE, lambda r: dict(corridor=r.choice(["north", "south"]),
                                          magnitude=float(r.uniform(1.3, 2.4)), duration=int(r.integers(4, 12)))),
    (EventType.PRICE_SPIKE, lambda r: dict(magnitude=float(r.uniform(3.0, 7.0)),
                                           duration=int(r.integers(3, 8)))),
    (EventType.BATTERY_TRIP, lambda r: dict(assets=(r.choice(["harbor", "valley"]),),
                                            duration=int(r.integers(8, 20)))),
    (EventType.LINE_CONGESTION, lambda r: dict(corridor="north", duration=int(r.integers(6, 16)))),
    (EventType.DEMAND_SURGE, lambda r: dict(magnitude=float(r.uniform(1.10, 1.25)),
                                            duration=int(r.integers(6, 12)))),
    (EventType.STORM_ALERT, lambda r: dict(severity=Severity.SEVERE, duration=30,
                                           lead=int(r.integers(8, 40)))),
    (EventType.FREQUENCY_DIP, lambda r: dict(magnitude=float(r.uniform(0.15, 0.35)),
                                             duration=2)),
    (EventType.ISO_DR_CALL, lambda r: dict(duration=int(r.integers(4, 10)))),
    (EventType.FORECAST_REVISION, lambda r: dict(duration=int(r.integers(4, 10)))),
]


def episode_plan(rng, start: int) -> list[tuple[int, EventType, dict]]:
    """2-4 random events at random intervals inside the episode."""
    n = int(rng.integers(2, 5))
    picks = rng.choice(len(INJECTABLE), size=n, replace=False)
    plan = []
    for i in picks:
        kind, spec = INJECTABLE[i]
        at = start + int(rng.integers(1, INTERVALS - 2))
        plan.append((at, kind, spec(rng)))
    return sorted(plan)


def main() -> int:
    config = load_config()
    forecaster = QuantileForecaster.load()
    if forecaster is None or not forecaster.is_trained:
        print("No trained forecaster; run `make train` first.", file=sys.stderr)
        return 2

    runner = EpisodeRunner(config, forecaster)
    suite = InvariantSuite()
    presets = sorted(PRESETS)
    rows = []
    agent_better = 0
    t0 = time.perf_counter()

    for ep in range(N_EPISODES):
        rng = np.random.default_rng(10_000 + ep)
        preset = presets[ep % len(presets)]
        world_seed = int(rng.integers(0, 100_000))
        start = int(rng.choice(START_CHOICES))
        plan = episode_plan(rng, start)

        results = {}
        for strategy in ("agent", "rule_based", "no_storage"):
            res = runner.run(
                strategy=strategy, preset=preset, seed=world_seed,
                start_interval=start, intervals=INTERVALS,
                check_invariants=(strategy == "agent"),
                event_plan=plan)
            results[strategy] = res
            if strategy == "agent" and res.suite is not None:
                for key, outcome in res.suite.outcomes.items():
                    target = suite.outcomes[key]
                    target.passed += outcome.passed
                    target.failed += outcome.failed
                    target.applicable += outcome.applicable
                    for f in outcome.failures:
                        if len(target.failures) < 8:
                            target.failures.append(f"ep{ep} {f}")

        a = results["agent"].kpis
        r = results["rule_based"].kpis
        n = results["no_storage"].kpis
        if a.energy_cost_usd <= r.energy_cost_usd:
            agent_better += 1
        rows.append({
            "episode": ep, "preset": preset, "world_seed": world_seed, "start": start,
            "events": [[at, k.value] for at, k, _ in plan],
            "agent": a.as_dict(), "rule_based": r.as_dict(), "no_storage": n.as_dict(),
        })
        print(f"  ep{ep:02d} {preset:21s} agent ${a.energy_cost_usd:9,.0f} "
              f"vs rule ${r.energy_cost_usd:9,.0f} "
              f"({'win ' if a.energy_cost_usd <= r.energy_cost_usd else 'LOSE'})  "
              f"shed {a.unserved_mwh:7.3f} vs {r.unserved_mwh:7.3f} MWh")

    # Determinism: same inputs and seed must reproduce the same cluster sequence.
    first = runner.run(strategy="agent", preset=presets[0], seed=4242,
                       start_interval=36, intervals=12, check_invariants=False)
    again = runner.run(strategy="agent", preset=presets[0], seed=4242,
                       start_interval=36, intervals=12, check_invariants=False)
    suite.determinism_pass = first.chosen_indices == again.chosen_indices

    elapsed = time.perf_counter() - t0
    decisions = N_EPISODES * INTERVALS
    agent_net = sum(r["agent"]["net_cost_usd"] for r in rows)
    rule_net = sum(r["rule_based"]["net_cost_usd"] for r in rows)
    agent_better_net = sum(
        1 for r in rows if r["agent"]["net_cost_usd"] <= r["rule_based"]["net_cost_usd"])
    agent_cost = sum(r["agent"]["energy_cost_usd"] for r in rows)
    rule_cost = sum(r["rule_based"]["energy_cost_usd"] for r in rows)
    nos_cost = sum(r["no_storage"]["energy_cost_usd"] for r in rows)
    agent_shed = sum(r["agent"]["unserved_mwh"] for r in rows)
    rule_shed = sum(r["rule_based"]["unserved_mwh"] for r in rows)
    agent_co2 = sum(r["agent"]["carbon_tonnes"] for r in rows)
    rule_co2 = sum(r["rule_based"]["carbon_tonnes"] for r in rows)
    nos_co2 = sum(r["no_storage"]["carbon_tonnes"] for r in rows)
    avoidable = sum(r["agent"]["avoidable_unserved_mwh"] for r in rows)
    win_rate = agent_better / N_EPISODES

    print(f"\n{'='*78}\nRELIABILITY TEST SUITE -- {N_EPISODES} randomised 8-hour episodes, "
          f"{decisions} decisions ({elapsed:.0f}s)\n{'='*78}")
    print(f"{'invariant':34s} {'passed':>8s} {'failed':>7s} {'rate':>7s}")
    report = suite.report()
    for row in report:
        if not row["applicable"]:
            print(f"{row['label']:34s} {'n/a':>8s} {'':>7s} {'':>7s}   (never arose)")
            continue
        print(f"{row['label']:34s} {row['passed']:8d} {row['failed']:7d} "
              f"{100*row['rate']:6.1f}%")
    for row in report:
        if row["failed"]:
            print(f"\n  ! {row['label']} -- {row['failed']} failures, e.g.:")
            for f in row["failures"][:3]:
                print(f"      {f}")

    print(f"\n{'outcome benchmark':34s}")
    print(f"{'  cost vs rule-based':34s} ${agent_cost:12,.0f} vs ${rule_cost:12,.0f}  "
          f"{100*(agent_cost-rule_cost)/abs(rule_cost):+.1f}%")
    print(f"{'  cost vs no-storage':34s} ${agent_cost:12,.0f} vs ${nos_cost:12,.0f}  "
          f"{100*(agent_cost-nos_cost)/abs(nos_cost):+.1f}%")
    print(f"{'  win rate vs rule-based':34s} {agent_better}/{N_EPISODES} "
          f"({100*win_rate:.0f}%)   target >= 75%")
    print(f"{'  net cost (incl. storage MtM)':34s} ${agent_net:12,.0f} vs ${rule_net:12,.0f}  "
          f"win rate {agent_better_net}/{N_EPISODES} ({100*agent_better_net/N_EPISODES:.0f}%)")
    print(f"{'  load shed vs rule-based':34s} {agent_shed:8.2f} vs {rule_shed:8.2f} MWh  "
          f"{100*(agent_shed-rule_shed)/max(rule_shed,1e-9):+.1f}%")
    print(f"{'  avoidable shedding':34s} {avoidable:8.3f} MWh   target 0")
    print(f"{'  CO2 vs rule-based':34s} {agent_co2:8.1f} vs {rule_co2:8.1f} t  "
          f"{100*(agent_co2-rule_co2)/max(rule_co2,1e-9):+.1f}%")
    print(f"{'  CO2 vs no-storage':34s} {agent_co2:8.1f} vs {nos_co2:8.1f} t  "
          f"{100*(agent_co2-nos_co2)/max(nos_co2,1e-9):+.1f}%")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = {
        "episodes": N_EPISODES, "intervals_each": INTERVALS, "decisions": decisions,
        "elapsed_s": round(elapsed, 1),
        "invariants": report,
        "benchmark": {
            "agent_cost_usd": agent_cost, "rule_based_cost_usd": rule_cost,
            "no_storage_cost_usd": nos_cost,
            "win_rate_vs_rule_based": win_rate,
            "agent_net_cost_usd": agent_net, "rule_based_net_cost_usd": rule_net,
            "win_rate_net_cost": agent_better_net / N_EPISODES,
            "agent_unserved_mwh": agent_shed, "rule_based_unserved_mwh": rule_shed,
            "avoidable_unserved_mwh": avoidable,
            "agent_co2_t": agent_co2, "rule_based_co2_t": rule_co2,
            "no_storage_co2_t": nos_co2,
        },
        "rows": rows,
    }
    (ARTIFACTS / "reliability.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten to artifacts/reliability.json")

    blocking = [r["label"] for r in report if r["failed"]]
    if blocking:
        print(f"\nFAILING INVARIANTS: {', '.join(blocking)}")
    if win_rate < 0.75:
        print(f"OUTCOME BENCHMARK MISS: win rate {100*win_rate:.0f}% below the 75% target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
