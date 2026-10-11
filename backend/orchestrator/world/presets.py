"""The five scenario presets (PRD F3.1) plus the live-demo opening state."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.events import Event, EventType, Severity


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    # Driver overrides
    cloud_volatility: float = 1.0
    wind_volatility: float = 1.0
    demand_volatility: float = 1.0
    price_volatility: float = 1.0
    base_wind_ms: float = 9.0
    base_cloud: float = 0.25
    # Events scheduled at fixed intervals (index into the 96-interval day)
    scheduled: tuple[tuple[int, EventType, dict], ...] = ()
    notes: str = ""


def _ev(**kw) -> dict:
    return kw


PRESETS: dict[str, Preset] = {
    "typical_day": Preset(
        key="typical_day",
        label="Typical day",
        description="Calm weather, ordinary price shape, no injected events.",
        cloud_volatility=0.8,
        wind_volatility=0.8,
        base_wind_ms=9.5,
        base_cloud=0.20,
    ),
    "volatile_weather": Preset(
        key="volatile_weather",
        label="Volatile weather",
        description="Fast-moving cloud banks and a wind surge; renewables swing hard.",
        cloud_volatility=2.6,
        wind_volatility=2.4,
        price_volatility=1.5,
        base_wind_ms=10.5,
        base_cloud=0.40,
        scheduled=(
            (38, EventType.CLOUD_BANK, _ev(region="region_a", magnitude=0.85, duration=8)),
            (52, EventType.WIND_SURGE, _ev(corridor="north", magnitude=1.45, duration=10)),
            (66, EventType.CLOUD_BANK, _ev(region="region_b", magnitude=0.75, duration=6)),
        ),
    ),
    "storm_passage": Preset(
        key="storm_passage",
        label="Storm passage",
        description="Storm alert 6 h ahead, then high winds into cut-out across the evening peak.",
        cloud_volatility=1.8,
        wind_volatility=2.0,
        base_wind_ms=11.0,
        base_cloud=0.45,
        scheduled=(
            # Alert issued at 11:00 with a 6 h lead, so the guardrail must look well
            # past the 4 h optimiser horizon. Storm lands in the evening peak.
            (44, EventType.STORM_ALERT, _ev(severity=Severity.SEVERE, duration=40, lead=24)),
            (66, EventType.CLOUD_BANK, _ev(region="region_b", magnitude=0.9, duration=12)),
            (68, EventType.WIND_SURGE, _ev(corridor="north", magnitude=2.6, duration=16)),
            (72, EventType.FREQUENCY_DIP, _ev(magnitude=0.25, duration=2)),
        ),
    ),
    "evening_price_crisis": Preset(
        key="evening_price_crisis",
        label="Evening price crisis",
        description="Sustained evening price spike with an ISO demand-response call.",
        price_volatility=2.2,
        base_wind_ms=7.5,
        base_cloud=0.30,
        scheduled=(
            (70, EventType.PRICE_SPIKE, _ev(magnitude=6.5, duration=10)),
            (72, EventType.ISO_DR_CALL, _ev(duration=8)),
            (76, EventType.DEMAND_SURGE, _ev(magnitude=1.18, duration=8)),
        ),
    ),
    "contingency_cascade": Preset(
        key="contingency_cascade",
        label="Contingency cascade",
        description="Battery trip, then corridor congestion, then an industrial demand surge.",
        cloud_volatility=1.4,
        wind_volatility=1.6,
        base_wind_ms=10.0,
        base_cloud=0.35,
        scheduled=(
            (34, EventType.BATTERY_TRIP, _ev(assets=("harbor",), duration=40)),
            (44, EventType.LINE_CONGESTION, _ev(corridor="north", duration=24)),
            (58, EventType.DEMAND_SURGE, _ev(magnitude=1.22, duration=12)),
            (62, EventType.FREQUENCY_DIP, _ev(magnitude=0.22, duration=2)),
        ),
    ),
}

DEFAULT_PRESET = "typical_day"
# The console must open in a realistic running state, never an empty screen:
# mid-morning with solar surplus already building (PRD "Demo and UX").
DEMO_OPENING_INTERVAL = 36  # 09:00


def preset(key: str) -> Preset:
    if key not in PRESETS:
        raise KeyError(f"unknown preset {key!r}; options are {sorted(PRESETS)}")
    return PRESETS[key]
