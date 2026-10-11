"""The action space: one cluster per interval, turned into per-asset set-points.

5 battery strategies x 2 market postures x 2 curtailment rules x 2 DR x 2 shift
= 80 clusters (PRD "Action space").
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum


class BatteryStrategy(str, Enum):
    SOAK_SURPLUS = "soak_surplus"
    PRICE_ARBITRAGE = "price_arbitrage"
    HOLD_FOR_HIGHER_PRICE = "hold_for_higher_price"
    RESERVE_FOR_EMERGENCIES = "reserve_for_emergencies"
    DISCHARGE_NOW = "discharge_now"

    @property
    def label(self) -> str:
        return {
            "soak_surplus": "Soak surplus",
            "price_arbitrage": "Price arbitrage",
            "hold_for_higher_price": "Hold for higher price",
            "reserve_for_emergencies": "Reserve for emergencies",
            "discharge_now": "Discharge now",
        }[self.value]


class MarketPosture(str, Enum):
    SPOT = "spot"
    BATTERIES_BEFORE_IMPORTS = "batteries_before_imports"

    @property
    def label(self) -> str:
        return {
            "spot": "Spot (import/export freely)",
            "batteries_before_imports": "Batteries before imports",
        }[self.value]


class CurtailmentRule(str, Enum):
    AT_NEGATIVE_PRICES = "at_negative_prices"
    ONLY_WHEN_FORCED = "only_when_forced"

    @property
    def label(self) -> str:
        return {
            "at_negative_prices": "Curtail at negative prices",
            "only_when_forced": "Curtail only when forced",
        }[self.value]


@dataclass(frozen=True)
class ActionCluster:
    index: int
    battery: BatteryStrategy
    market: MarketPosture
    curtailment: CurtailmentRule
    demand_response: bool
    load_shift: bool

    @property
    def label(self) -> str:
        parts = [self.battery.label, self.market.label, self.curtailment.label]
        parts.append("DR on" if self.demand_response else "DR off")
        parts.append("Shift on" if self.load_shift else "Shift off")
        return " · ".join(parts)

    @property
    def short_label(self) -> str:
        return (
            f"{self.battery.label}"
            f"{' + DR' if self.demand_response else ''}"
            f"{' + Shift' if self.load_shift else ''}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "battery": self.battery.value,
            "market": self.market.value,
            "curtailment": self.curtailment.value,
            "demand_response": self.demand_response,
            "load_shift": self.load_shift,
            "label": self.label,
        }


def all_clusters() -> tuple[ActionCluster, ...]:
    """Enumerate all 80 clusters in a fixed, reproducible order."""
    combos = itertools.product(
        BatteryStrategy, MarketPosture, CurtailmentRule, (False, True), (False, True)
    )
    return tuple(
        ActionCluster(i, b, m, c, dr, sh) for i, (b, m, c, dr, sh) in enumerate(combos)
    )


CLUSTERS: tuple[ActionCluster, ...] = all_clusters()
N_CLUSTERS = len(CLUSTERS)


@dataclass(frozen=True)
class SetPoint:
    """A concrete instruction: verb, asset, MW and the reason for it."""

    verb: str           # charge | discharge | buy | sell | curtail | shed | shift | hold
    asset_id: str
    asset_name: str
    mw: float
    reason: str
    unit: str = "MW"
    requires_approval: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "verb": self.verb,
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "mw": round(self.mw, 2),
            "unit": self.unit,
            "reason": self.reason,
            "requires_approval": self.requires_approval,
        }

    @property
    def text(self) -> str:
        """Operator vocabulary, with units."""
        return f"{self.verb.capitalize()} {self.asset_name} {self.mw:.1f} {self.unit}"


@dataclass
class CommittedActions:
    interval: int
    cluster: ActionCluster
    set_points: tuple[SetPoint, ...] = ()
    queued_for_approval: tuple[SetPoint, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def executed_mw(self) -> float:
        return sum(abs(sp.mw) for sp in self.set_points)
