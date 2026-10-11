"""Rule-based dispatch: the PRD's fallback controller and its benchmark baseline.

"charge on surplus, discharge on deficit" with reserve held -- the fixed rule the PRD
says operators fall back on, and the baseline the agent is measured against.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config_loader import Config
from ..domain.actions import CLUSTERS, ActionCluster, BatteryStrategy, CurtailmentRule, MarketPosture
from ..domain.state import StateSnapshot


def rule_based_cluster(config: Config, snapshot: StateSnapshot) -> ActionCluster:
    """The fixed rule, expressed as one of the 80 clusters so it runs on the same
    engine and is therefore comparable interval by interval."""
    surplus = snapshot.net_load_mw < 0
    battery = BatteryStrategy.SOAK_SURPLUS if surplus else BatteryStrategy.DISCHARGE_NOW
    return next(
        c for c in CLUSTERS
        if c.battery is battery
        and c.market is MarketPosture.BATTERIES_BEFORE_IMPORTS
        and c.curtailment is CurtailmentRule.ONLY_WHEN_FORCED
        and not c.demand_response
        and not c.load_shift
    )


def no_storage_cluster(config: Config, snapshot: StateSnapshot) -> ActionCluster:
    """The no-storage baseline: batteries idle, spot market, forced curtailment only."""
    return next(
        c for c in CLUSTERS
        if c.battery is BatteryStrategy.HOLD_FOR_HIGHER_PRICE
        and c.market is MarketPosture.SPOT
        and c.curtailment is CurtailmentRule.ONLY_WHEN_FORCED
        and not c.demand_response
        and not c.load_shift
    )


@dataclass(frozen=True)
class FallbackDecision:
    cluster: ActionCluster
    reason: str
    reserve_held_pct: float


def fallback(config: Config, snapshot: StateSnapshot, reason: str) -> FallbackDecision:
    """PRD fail-safe: on a missed cycle, stale feed or solver error, fall back to
    rule-based dispatch with reserve held, raise an alarm, and log the cause."""
    return FallbackDecision(
        cluster=rule_based_cluster(config, snapshot),
        reason=reason,
        reserve_held_pct=config.portfolio.battery_limits.reserve_floor_pct,
    )
