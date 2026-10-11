"""One normalised state snapshot per 15-minute interval.

Anything stale or missing is flagged and replaced by the forecast, never silently
zeroed (PRD "Inputs").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .events import Event, EventType, MaintenanceWindow


@dataclass
class DataQuality:
    """Staleness / substitution flags, surfaced to the operator as warnings."""

    stale_fields: tuple[str, ...] = ()
    substituted_fields: tuple[str, ...] = ()
    intervals_stale: dict[str, int] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return bool(self.stale_fields or self.substituted_fields)


@dataclass
class StateSnapshot:
    interval: int                       # index into the simulated clock
    minute_of_day: int
    day: int

    solar_mw: dict[str, float]          # available output per farm, pre-curtailment
    wind_mw: dict[str, float]
    wind_speed_ms: dict[str, float]
    cloud_cover: dict[str, float]       # 0..1 per cloud region

    consumer_mw: dict[str, float]       # per industrial consumer, pre-DR/shift
    municipal_mw: float

    battery_soc_pct: dict[str, float]
    battery_available: dict[str, bool]

    frequency_hz: float
    price_per_mwh: float
    carbon_price_per_tonne: float

    import_limit_mw: float              # current, possibly derated
    export_limit_mw: float
    corridor_limit_mw: dict[str, float]

    events: tuple[Event, ...] = ()
    maintenance: tuple[MaintenanceWindow, ...] = ()
    price_history: tuple[float, ...] = ()   # rolling window for percentile triggers
    shift_debt_mwh: float = 0.0             # shifted load awaiting off-peak repayment
    dr_calls_today: int = 0
    quality: DataQuality = field(default_factory=DataQuality)

    # ---- derived ----------------------------------------------------------
    @property
    def total_solar_mw(self) -> float:
        return sum(self.solar_mw.values())

    @property
    def total_wind_mw(self) -> float:
        return sum(self.wind_mw.values())

    @property
    def available_renewable_mw(self) -> float:
        return self.total_solar_mw + self.total_wind_mw

    @property
    def total_demand_mw(self) -> float:
        return sum(self.consumer_mw.values()) + self.municipal_mw

    @property
    def net_load_mw(self) -> float:
        """Positive = deficit to cover, negative = surplus to absorb."""
        return self.total_demand_mw - self.available_renewable_mw

    def event_of(self, kind: EventType) -> Event | None:
        for e in self.events:
            if e.type is kind and e.active_at(self.interval):
                return e
        return None

    def has_event(self, kind: EventType) -> bool:
        return self.event_of(kind) is not None

    @property
    def storm_alert(self) -> Event | None:
        return self.event_of(EventType.STORM_ALERT)

    @property
    def dr_call_active(self) -> bool:
        return self.has_event(EventType.ISO_DR_CALL)

    @property
    def batteries_offline(self) -> tuple[str, ...]:
        return tuple(bid for bid, ok in self.battery_available.items() if not ok)

    def price_percentile(self, q: float) -> float:
        """Percentile of the rolling price window; falls back to current price."""
        if not self.price_history:
            return self.price_per_mwh
        ordered = sorted(self.price_history)
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[idx]

    def fleet_soc_pct(self, energy_by_battery: dict[str, float]) -> float:
        """Energy-weighted fleet state of charge."""
        total = sum(energy_by_battery.values())
        if total <= 0:
            return 0.0
        return sum(
            self.battery_soc_pct[bid] * cap / total
            for bid, cap in energy_by_battery.items()
        )
