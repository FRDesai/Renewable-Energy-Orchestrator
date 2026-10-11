"""Optimiser latency against the PRD's budget: P95 < 1 s for 80 x 24 x 16."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.config_loader import load_config            # noqa: E402
from orchestrator.decide.dispatch import DispatchEngine        # noqa: E402
from orchestrator.forecast.scenarios import build_scenarios    # noqa: E402
from orchestrator.ingest.normalize import Normalizer           # noqa: E402
from orchestrator.world.presets import PRESETS                 # noqa: E402
from orchestrator.world.world import World                     # noqa: E402


def main() -> int:
    config = load_config()
    engine = DispatchEngine(config)
    o = config.objectives
    samples = []
    for preset in sorted(PRESETS):
        w = World(config, seed=17, preset=preset)
        n = Normalizer(config)
        for i in range(90):
            snap = n.build(w.telemetry())
            if i in (20, 36, 48, 60, 76, 84):
                samples.append((preset, snap))
            w.step()

    scen_ms, sim_ms, total_ms = [], [], []
    for _, snap in samples:
        for rep in range(5):
            t0 = time.perf_counter()
            scen = build_scenarios(config, snap, seed=rep)
            t1 = time.perf_counter()
            engine.simulate(scen, snap.battery_soc_pct, snap.battery_available)
            t2 = time.perf_counter()
            scen_ms.append((t1 - t0) * 1000)
            sim_ms.append((t2 - t1) * 1000)
            total_ms.append((t2 - t0) * 1000)

    def stats(name, xs):
        a = np.asarray(xs)
        print(f"  {name:<22} mean {a.mean():7.2f} ms   P50 {np.percentile(a,50):7.2f}   "
              f"P95 {np.percentile(a,95):7.2f}   max {a.max():7.2f}")

    print(f"Sweep: {80} clusters x {o.scenarios} scenarios x {o.steps} steps "
          f"= {80*o.scenarios*o.steps:,} cluster-scenario-steps per decision")
    print(f"Runs: {len(total_ms)} across {len(samples)} states\n")
    stats("scenario generation", scen_ms)
    stats("dispatch + scoring", sim_ms)
    stats("total per decision", total_ms)

    p95 = float(np.percentile(total_ms, 95))
    budget = 1000.0
    ok = p95 < budget
    print(f"\nPRD budget: decision P95 < {budget:.0f} ms (inside a 900 s interval)")
    print(f"Measured:   P95 = {p95:.1f} ms  ->  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
