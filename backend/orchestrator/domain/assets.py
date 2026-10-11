"""Asset definitions. Everything here comes from versioned config, never hardcoded."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AssetKind(str, Enum):
    SOLAR = "solar"
    WIND = "wind"
    BATTERY = "battery"
    CONSUMER = "consumer"
    GRID = "grid"


@dataclass(frozen=True)
class SolarFarm:
    id: str
    name: str
    capacity_mw: float
    cloud_region: str


@dataclass(frozen=True)
class WindFarm:
    id: str
    name: str
    capacity_mw: float
    corridor: str


@dataclass(frozen=True)
class SolarResponse:
    max_cloud_attenuation: float

    def fraction(self, clear_sky: float, cloud_cover: float) -> float:
        return clear_sky * (1.0 - self.max_cloud_attenuation * cloud_cover)


@dataclass(frozen=True)
class WindCurve:
    cut_in_ms: float
    rated_ms: float
    cut_out_ms: float

    def output_fraction(self, speed_ms: float) -> float:
        """Fraction of rated output at a given wind speed.

        Cubic ramp between cut-in and rated, flat at rated, zero outside.
        """
        if speed_ms < self.cut_in_ms or speed_ms >= self.cut_out_ms:
            return 0.0
        if speed_ms >= self.rated_ms:
            return 1.0
        span = self.rated_ms - self.cut_in_ms
        return ((speed_ms - self.cut_in_ms) / span) ** 3


@dataclass(frozen=True)
class Battery:
    id: str
    name: str
    power_mw: float
    energy_mwh: float
    one_way_efficiency: float
    wear_cost_per_mwh: float

    @property
    def round_trip_efficiency(self) -> float:
        return self.one_way_efficiency**2


@dataclass(frozen=True)
class BatteryLimits:
    soc_min_pct: float
    soc_max_pct: float
    reserve_floor_pct: float
    storm_readiness_pct: float
    storm_guardrail_pct: float
    storm_max_lead_hours: float


@dataclass(frozen=True)
class IndustrialConsumer:
    id: str
    name: str
    base_mw: float
    shiftable_mw: float
    dr_mw: float
    critical: bool


@dataclass(frozen=True)
class MunicipalLoad:
    min_mw: float
    max_mw: float
    critical: bool


@dataclass(frozen=True)
class GridConfig:
    import_limit_mw: float
    export_limit_mw: float
    export_derated_mw: float
    north_corridor_limit_mw: float
    north_corridor_derated_mw: float
    nominal_frequency_hz: float
    frequency_reflex_hz: float


@dataclass(frozen=True)
class Portfolio:
    version: int
    interval_minutes: int
    solar_farms: tuple[SolarFarm, ...]
    wind_farms: tuple[WindFarm, ...]
    solar_response: SolarResponse
    wind_curve: WindCurve
    batteries: tuple[Battery, ...]
    battery_limits: BatteryLimits
    consumers: tuple[IndustrialConsumer, ...]
    municipal: MunicipalLoad
    grid: GridConfig

    @property
    def interval_hours(self) -> float:
        return self.interval_minutes / 60.0

    @property
    def solar_capacity_mw(self) -> float:
        return sum(f.capacity_mw for f in self.solar_farms)

    @property
    def wind_capacity_mw(self) -> float:
        return sum(f.capacity_mw for f in self.wind_farms)

    @property
    def battery_power_mw(self) -> float:
        return sum(b.power_mw for b in self.batteries)

    @property
    def battery_energy_mwh(self) -> float:
        return sum(b.energy_mwh for b in self.batteries)

    @property
    def cloud_regions(self) -> tuple[str, ...]:
        seen: list[str] = []
        for f in self.solar_farms:
            if f.cloud_region not in seen:
                seen.append(f.cloud_region)
        return tuple(seen)

    @property
    def corridors(self) -> tuple[str, ...]:
        seen: list[str] = []
        for f in self.wind_farms:
            if f.corridor not in seen:
                seen.append(f.corridor)
        return tuple(seen)

    def solar(self, farm_id: str) -> SolarFarm:
        return next(f for f in self.solar_farms if f.id == farm_id)

    def wind(self, farm_id: str) -> WindFarm:
        return next(f for f in self.wind_farms if f.id == farm_id)

    def battery(self, battery_id: str) -> Battery:
        return next(b for b in self.batteries if b.id == battery_id)

    def consumer(self, consumer_id: str) -> IndustrialConsumer:
        return next(c for c in self.consumers if c.id == consumer_id)

    @property
    def total_dr_mw(self) -> float:
        return sum(c.dr_mw for c in self.consumers)

    @property
    def total_shiftable_mw(self) -> float:
        return sum(c.shiftable_mw for c in self.consumers)
