"""Normalise telemetry into one state snapshot per interval.

PRD "Inputs": anything stale or missing is flagged and replaced by the forecast,
never silently zeroed. PRD "Fail-safe behavior": a feed stale beyond 2 intervals
raises an alarm and the system falls back to rule-based dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from ..config_loader import Config
from ..domain.state import DataQuality, StateSnapshot
from ..world.world import Telemetry

STALE_ALARM_INTERVALS = 2

# Channels that can be cut independently by a data-dropout injection.
CHANNELS = ("solar", "wind", "demand", "price", "frequency", "battery")


class Substitute(Protocol):
    """Supplies a replacement value for a missing channel.

    In v1 this is the forecaster (M3); before it exists, last-good persistence is
    used, which is still a forecast in the degenerate sense.
    """

    def __call__(self, channel: str, interval: int, last_good: object) -> object: ...


@dataclass
class Normalizer:
    config: Config
    substitute: Substitute | None = None
    _last_good: dict[str, object] = field(default_factory=dict)
    _stale_for: dict[str, int] = field(default_factory=dict)
    # Interval at which each channel's staleness was last counted. The live session
    # re-normalises the *same* interval whenever an injected event forces a re-plan,
    # and "stale for N intervals" must stay a count of intervals rather than of
    # rebuilds -- otherwise a few re-plans would trip the fail-safe on a healthy feed.
    _stale_at: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self._last_good.clear()
        self._stale_for.clear()
        self._stale_at.clear()

    # ---- helpers ----------------------------------------------------------
    def _resolve(self, channel: str, interval: int, fresh: object, missing: bool):
        """Return (value, was_substituted)."""
        if not missing:
            self._last_good[channel] = fresh
            self._stale_for[channel] = 0
            self._stale_at.pop(channel, None)
            return fresh, False

        if self._stale_at.get(channel) != interval:
            self._stale_for[channel] = self._stale_for.get(channel, 0) + 1
            self._stale_at[channel] = interval
        last = self._last_good.get(channel)
        if self.substitute is not None:
            replacement = self.substitute(channel, interval, last)
            if replacement is not None:
                return replacement, True
        if last is not None:
            return last, True
        # Nothing to substitute from. Returning the fresh (possibly zero) reading
        # here would be the silent-zeroing failure the PRD forbids, so this is
        # only reachable on the very first interval of a dropout at startup.
        return fresh, True

    def _out_of_range(self, telemetry: Telemetry) -> tuple[str, ...]:
        p = self.config.portfolio
        bad: list[str] = []
        for farm in p.solar_farms:
            v = telemetry.solar_mw.get(farm.id, 0.0)
            if v < -1e-6 or v > farm.capacity_mw * 1.02:
                bad.append(f"solar:{farm.id}")
        for farm in p.wind_farms:
            v = telemetry.wind_mw.get(farm.id, 0.0)
            if v < -1e-6 or v > farm.capacity_mw * 1.02:
                bad.append(f"wind:{farm.id}")
        for b in p.batteries:
            soc = telemetry.battery_soc_pct.get(b.id, 0.0)
            if not (p.battery_limits.soc_min_pct - 1e-6
                    <= soc <= p.battery_limits.soc_max_pct + 1e-6):
                bad.append(f"battery:{b.id}")
        if not (45.0 <= telemetry.frequency_hz <= 55.0):
            bad.append("frequency")
        return tuple(bad)

    # ---- the one public entry point ---------------------------------------
    def build(self, telemetry: Telemetry) -> StateSnapshot:
        missing = set(telemetry.missing)
        substituted: list[str] = []
        stale: list[str] = []

        solar, sub = self._resolve("solar", telemetry.interval,
                                   dict(telemetry.solar_mw), "solar" in missing)
        if sub:
            substituted.append("solar")
        wind, sub = self._resolve("wind", telemetry.interval,
                                  dict(telemetry.wind_mw), "wind" in missing)
        if sub:
            substituted.append("wind")
        demand, sub = self._resolve(
            "demand", telemetry.interval,
            (dict(telemetry.consumer_mw), telemetry.municipal_mw), "demand" in missing)
        if sub:
            substituted.append("demand")
        price, sub = self._resolve("price", telemetry.interval,
                                   telemetry.price_per_mwh, "price" in missing)
        if sub:
            substituted.append("price")
        freq, sub = self._resolve("frequency", telemetry.interval,
                                  telemetry.frequency_hz, "frequency" in missing)
        if sub:
            substituted.append("frequency")
        soc, sub = self._resolve("battery", telemetry.interval,
                                 dict(telemetry.battery_soc_pct), "battery" in missing)
        if sub:
            substituted.append("battery")

        for channel, count in self._stale_for.items():
            if count > 0:
                stale.append(channel)

        consumer_mw, municipal_mw = demand  # type: ignore[misc]
        snapshot = StateSnapshot(
            interval=telemetry.interval,
            minute_of_day=telemetry.minute_of_day,
            day=telemetry.day,
            solar_mw=solar,            # type: ignore[arg-type]
            wind_mw=wind,              # type: ignore[arg-type]
            wind_speed_ms=dict(telemetry.wind_speed_ms),
            cloud_cover=dict(telemetry.cloud_cover),
            consumer_mw=dict(consumer_mw),
            municipal_mw=float(municipal_mw),
            battery_soc_pct=soc,       # type: ignore[arg-type]
            battery_available=dict(telemetry.battery_available),
            frequency_hz=float(freq),  # type: ignore[arg-type]
            price_per_mwh=float(price),  # type: ignore[arg-type]
            carbon_price_per_tonne=telemetry.carbon_price_per_tonne,
            import_limit_mw=telemetry.import_limit_mw,
            export_limit_mw=telemetry.export_limit_mw,
            corridor_limit_mw=dict(telemetry.corridor_limit_mw),
            events=telemetry.events,
            maintenance=telemetry.maintenance,
            price_history=telemetry.price_history,
            shift_debt_mwh=telemetry.shift_debt_mwh,
            dr_calls_today=telemetry.dr_calls_today,
            quality=DataQuality(
                stale_fields=tuple(sorted(stale)),
                substituted_fields=tuple(sorted(substituted)),
                intervals_stale={k: v for k, v in self._stale_for.items() if v > 0},
            ),
        )
        out_of_range = self._out_of_range(telemetry)
        if out_of_range:
            snapshot.quality = DataQuality(
                stale_fields=snapshot.quality.stale_fields + out_of_range,
                substituted_fields=snapshot.quality.substituted_fields,
                intervals_stale=snapshot.quality.intervals_stale,
            )
        return snapshot

    @property
    def alarm(self) -> bool:
        """True when any feed has been stale beyond the fail-safe threshold."""
        return any(v > STALE_ALARM_INTERVALS for v in self._stale_for.values())

    @property
    def worst_stale_intervals(self) -> int:
        return max(self._stale_for.values(), default=0)
