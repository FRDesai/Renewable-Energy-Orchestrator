"""F3 scenario benchmark: 24 h x N paths x 5 presets x 3 strategies."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.config_loader import load_config               # noqa: E402
from orchestrator.forecast.quantile import QuantileForecaster    # noqa: E402
from orchestrator.sim.monte_carlo import MonteCarlo              # noqa: E402
from orchestrator.world.presets import PRESETS                   # noqa: E402

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
PATHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def main() -> int:
    config = load_config()
    f = QuantileForecaster.load()
    if f is None or not f.is_trained:
        print("run `make train` first", file=sys.stderr)
        return 2
    mc = MonteCarlo(config, f)
    t0 = time.perf_counter()
    out = {}
    print(f"24 h x {PATHS} paths x 5 presets x 3 strategies\n")
    print(f"{'preset':22s} {'agent $':>12s} {'rule $':>12s} {'no-stor $':>12s} "
          f"{'shed a/r':>16s} {'CO2 a/r':>16s}")
    for preset in sorted(PRESETS):
        res = mc.run(preset=preset, paths=PATHS, intervals=96)
        out[preset] = res.as_dict()
        a = res.by_strategy["agent"]
        r = res.by_strategy["rule_based"]
        n = res.by_strategy["no_storage"]
        print(f"{preset:22s} {a.series('energy_cost_usd').mean():12,.0f} "
              f"{r.series('energy_cost_usd').mean():12,.0f} "
              f"{n.series('energy_cost_usd').mean():12,.0f} "
              f"{a.series('unserved_mwh').mean():7.1f}/{r.series('unserved_mwh').mean():<8.1f} "
              f"{a.series('carbon_tonnes').mean():7.0f}/{r.series('carbon_tonnes').mean():<8.0f}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "simulation.json").write_text(json.dumps(out, indent=2))
    print(f"\n{time.perf_counter()-t0:.0f}s -> artifacts/simulation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
