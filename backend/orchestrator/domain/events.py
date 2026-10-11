"""Situations the agent must handle. Each can be injected live in the demo."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    CLOUD_BANK = "cloud_bank"
    WIND_SURGE = "wind_surge"
    PRICE_SPIKE = "price_spike"
    BATTERY_TRIP = "battery_trip"
    LINE_CONGESTION = "line_congestion"
    DEMAND_SURGE = "demand_surge"
    STORM_ALERT = "storm_alert"
    FREQUENCY_DIP = "frequency_dip"
    ISO_DR_CALL = "iso_dr_call"
    FORECAST_REVISION = "forecast_revision"
    DATA_DROPOUT = "data_dropout"


# Events with a known end time enter all scenarios; the rest are seen once they
# start and assumed to decay with a typical persistence.
ANNOUNCED_TYPES = frozenset(
    {
        EventType.STORM_ALERT,
        EventType.ISO_DR_CALL,
        EventType.LINE_CONGESTION,
        EventType.FORECAST_REVISION,
    }
)

HIGH_IMPACT_TYPES = frozenset({EventType.STORM_ALERT})


class Severity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass
class Event:
    """A situation affecting the portfolio.

    `start_interval` / `end_interval` are indices into the simulated clock, so the
    engine never depends on wall-clock time (which would break replay).
    """

    type: EventType
    start_interval: int
    end_interval: int | None = None
    region: str | None = None
    corridor: str | None = None
    affected_assets: tuple[str, ...] = ()
    arrival_interval: int | None = None   # storm alerts: when it reaches site
    severity: Severity = Severity.MODERATE
    magnitude: float = 1.0
    # Provenance, for text-extracted events.
    source_phrases: dict[str, str] = field(default_factory=dict)
    extraction_confidence: float = 1.0
    extracted_by: str | None = None
    confirmed_by_operator: bool = False
    note: str = ""

    @property
    def announced(self) -> bool:
        return self.type in ANNOUNCED_TYPES

    @property
    def needs_confirmation(self) -> bool:
        """Low-confidence or high-impact extractions need operator confirmation
        before they reach the optimizer."""
        if self.extracted_by is None:
            return False
        if self.confirmed_by_operator:
            return False
        return self.extraction_confidence < 0.8 or self.type in HIGH_IMPACT_TYPES

    def on_site_at(self, interval: int) -> bool:
        """A storm alert is `active` from the moment it is issued, but only `on site`
        once the forecast arrival has passed."""
        if self.arrival_interval is None:
            return self.active_at(interval)
        return self.active_at(interval) and interval >= self.arrival_interval

    def intervals_to_arrival(self, interval: int) -> int | None:
        if self.arrival_interval is None:
            return None
        return self.arrival_interval - interval

    def active_at(self, interval: int) -> bool:
        if interval < self.start_interval:
            return False
        if self.end_interval is None:
            return True
        return interval < self.end_interval


@dataclass
class MaintenanceWindow:
    id: str
    asset_id: str
    start_interval: int
    duration_intervals: int
    deadline_interval: int
    crew: str
    status: str = "planned"  # planned | in_progress | complete | delayed
    lost_output_usd: float = 0.0

    @property
    def end_interval(self) -> int:
        return self.start_interval + self.duration_intervals

    def overlaps(self, start: int, end: int) -> bool:
        return self.start_interval < end and start < self.end_interval


@dataclass
class WorkOrder:
    id: str
    asset_id: str
    reason: str
    raised_interval: int
    kind: str = "inspection"  # inspection | repair
    status: str = "open"
