"""Cluster -> numeric strategy encoding for the vectorised dispatch simulation."""
from __future__ import annotations

import numpy as np

from ..domain.actions import (
    CLUSTERS,
    N_CLUSTERS,
    ActionCluster,
    BatteryStrategy,
    CurtailmentRule,
    MarketPosture,
)

# Integer codes, fixed order, so the encoding is stable across runs and replays.
BATTERY_CODES = {s: i for i, s in enumerate(BatteryStrategy)}
SOAK, ARBITRAGE, HOLD, RESERVE, DISCHARGE = (
    BATTERY_CODES[BatteryStrategy.SOAK_SURPLUS],
    BATTERY_CODES[BatteryStrategy.PRICE_ARBITRAGE],
    BATTERY_CODES[BatteryStrategy.HOLD_FOR_HIGHER_PRICE],
    BATTERY_CODES[BatteryStrategy.RESERVE_FOR_EMERGENCIES],
    BATTERY_CODES[BatteryStrategy.DISCHARGE_NOW],
)


class ClusterEncoding:
    """Column vectors of shape (n_clusters, 1), broadcastable over scenarios."""

    def __init__(self, clusters: tuple[ActionCluster, ...] = CLUSTERS) -> None:
        self.clusters = clusters
        n = len(clusters)
        self.battery = np.array([BATTERY_CODES[c.battery] for c in clusters]).reshape(n, 1)
        self.batteries_before_imports = np.array(
            [c.market is MarketPosture.BATTERIES_BEFORE_IMPORTS for c in clusters]
        ).reshape(n, 1)
        self.curtail_at_negative = np.array(
            [c.curtailment is CurtailmentRule.AT_NEGATIVE_PRICES for c in clusters]
        ).reshape(n, 1)
        self.demand_response = np.array([c.demand_response for c in clusters]).reshape(n, 1)
        self.load_shift = np.array([c.load_shift for c in clusters]).reshape(n, 1)

    def __len__(self) -> int:
        return len(self.clusters)

    def is_strategy(self, code: int) -> np.ndarray:
        return self.battery == code


ENCODING = ClusterEncoding()
assert len(ENCODING) == N_CLUSTERS
