"""The synthetic utility. Holds physical truth; the agent only ever sees telemetry.

This stands in for SCADA/EMS/ISO feeds (see DECISIONS.md D2). It is deliberately the
same object the Monte Carlo simulator drives in batch mode, so F3 inherits F1/F2
correctness rather than re-implementing physics.
"""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

from ..config_loader import Config
from ..domain.events import Event, EventType, MaintenanceWindow, Severity
from .drivers import (
    MINUTES_PER_DAY,
    OUParams,
    clear_sky_fraction,
    demand_shape,
    ou_step,
    price_from_residual,
)
from .presets import DEFAULT_PRESET, Preset, preset as get_preset

INTERVALS_PER_DAY = 96


def _stable_hash(text: str) -> int:
    """Process-independent replacement for `hash()` on asset identifiers."""
    return int.from_bytes(hashlib.blake2s(text.encode(), digest_size=8).digest(), "big")


@dataclass
class Telemetry:
    """Raw readings as they would arrive from the field, before normalisation."""

    interval: int
    minute_of_day: int
    day: int
    solar_mw: dict[str, float]
    wind_mw: dict[str, float]
    wind_speed_ms: dict[str, float]
    cloud_cover: dict[str, float]
    consumer_mw: dict[str, float]
    municipal_mw: float
    battery_soc_pct: dict[str, float]
    battery_available: dict[str, bool]
    frequency_hz: float
    price_per_mwh: float
    carbon_price_per_tonne: float
    import_limit_mw: float
    export_limit_mw: float
    corridor_limit_mw: dict[str, float]
    events: tuple[Event, ...]
    maintenance: tuple[MaintenanceWindow, ...]
    price_history: tuple[float, ...]
    shift_debt_mwh: float
    dr_calls_today: int
    missing: tuple[str, ...] = ()   # channels cut by a data-dropout injection


@dataclass
class AppliedDispatch:
    """What the gateway actually executed in the interval just decided."""

    battery_mw: dict[str, float] = field(default_factory=dict)  # + discharge, - charge
    curtailed_mw: dict[str, float] = field(default_factory=dict)
    dr_mw: float = 0.0
    shift_mw: float = 0.0      # + reduced now (debt accrues), - repaid now
    grid_mw: float = 0.0       # + import, - export
    unserved_mwh: float = 0.0


class World:
    def __init__(
        self,
        config: Config,
        *,
        seed: int = 0,
        preset: str | Preset = DEFAULT_PRESET,
        start_interval: int = 0,
    ) -> None:
        self.config = config
        self.preset = get_preset(preset) if isinstance(preset, str) else preset
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self.interval = int(start_interval)

        p = config.portfolio
        self.cloud: dict[str, float] = {
            r: float(np.clip(self.preset.base_cloud + 0.1 * self._rng.standard_normal(), 0, 1))
            for r in p.cloud_regions
        }
        self.wind_speed: dict[str, float] = {
            c: float(max(0.0, self.preset.base_wind_ms + 1.0 * self._rng.standard_normal()))
            for c in p.corridors
        }
        self.demand_noise = 0.0
        self.price_noise = 0.0
        self.frequency_hz = p.grid.nominal_frequency_hz

        self.battery_soc_pct: dict[str, float] = {b.id: 55.0 for b in p.batteries}
        self.battery_available: dict[str, bool] = {b.id: True for b in p.batteries}

        self.events: list[Event] = []
        self.maintenance: list[MaintenanceWindow] = []
        self.price_history: deque[float] = deque(maxlen=INTERVALS_PER_DAY)
        self.shift_debt_mwh = 0.0
        self.dr_calls_today = 0
        self.missing_channels: tuple[str, ...] = ()
        self._dropout_until: int | None = None

        self._schedule_preset_events()
        self._seed_maintenance()
        self._price = self._compute_price()
        self.price_history.append(self._price)

    # ---- construction helpers --------------------------------------------
    def _schedule_preset_events(self) -> None:
        for at, kind, spec in self.preset.scheduled:
            self.events.append(self._build_event(kind, at, spec))

    def _build_event(self, kind: EventType, at: int, spec: dict) -> Event:
        duration = int(spec.get("duration", 8))
        lead = spec.get("lead")
        return Event(
            type=kind,
            start_interval=at,
            end_interval=at + duration,
            region=spec.get("region"),
            corridor=spec.get("corridor"),
            affected_assets=tuple(spec.get("assets", ())),
            severity=spec.get("severity", Severity.MODERATE),
            magnitude=float(spec.get("magnitude", 1.0)),
            arrival_interval=(at + int(lead)) if lead is not None else None,
            note=spec.get("note", ""),
        )

    def _seed_maintenance(self) -> None:
        """One planned window per run, late in the day, so the maintenance planner
        has something real to value (and a storm can overlap it)."""
        p = self.config.portfolio
        asset = p.wind_farms[1].id if len(p.wind_farms) > 1 else p.wind_farms[0].id
        self.maintenance.append(
            MaintenanceWindow(
                id="mw-1",
                asset_id=asset,
                start_interval=52,
                duration_intervals=8,
                deadline_interval=92,
                crew="crew-a",
            )
        )

    # ---- event access -----------------------------------------------------
    def active_events(self) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.active_at(self.interval))

    def _active(self, kind: EventType) -> Event | None:
        for e in self.events:
            if e.type is kind and e.active_at(self.interval):
                return e
        return None

    def inject(self, kind: EventType, **spec) -> Event:
        """One-click live injection from the console."""
        ev = self._build_event(kind, self.interval, spec)
        self.events.append(ev)
        if kind is EventType.BATTERY_TRIP:
            for bid in ev.affected_assets or tuple(self.battery_available):
                self.battery_available[bid] = False
        if kind is EventType.FREQUENCY_DIP:
            self.frequency_hz = self.config.portfolio.grid.nominal_frequency_hz - ev.magnitude
        if kind is EventType.DATA_DROPOUT:
            channels = tuple(spec.get("channels", ("solar", "price")))
            self.missing_channels = channels
            self._dropout_until = self.interval + int(spec.get("duration", 4))
        return ev

    def add_event(self, event: Event) -> Event:
        """Accept an externally-built event (e.g. an extracted storm bulletin)."""
        self.events.append(event)
        return event

    # ---- physics ----------------------------------------------------------
    @property
    def minute_of_day(self) -> int:
        return (self.interval * self.config.portfolio.interval_minutes) % MINUTES_PER_DAY

    @property
    def day(self) -> int:
        return self.interval // INTERVALS_PER_DAY

    def _storm_on_site(self) -> Event | None:
        for e in self.events:
            if e.type is EventType.STORM_ALERT and e.on_site_at(self.interval):
                return e
        return None

    def solar_output(self) -> dict[str, float]:
        p = self.config.portfolio
        clear = clear_sky_fraction(self.minute_of_day)
        out: dict[str, float] = {}
        for f in p.solar_farms:
            cover = self.cloud[f.cloud_region]
            # Cloud attenuates but never fully blocks diffuse irradiance.
            avail = p.solar_response.fraction(clear, cover)
            if self._outage(f.id):
                avail = 0.0
            out[f.id] = round(max(0.0, f.capacity_mw * avail), 4)
        return out

    def wind_output(self) -> tuple[dict[str, float], dict[str, float]]:
        p = self.config.portfolio
        curve = p.wind_curve
        speeds: dict[str, float] = {}
        out: dict[str, float] = {}
        for f in p.wind_farms:
            # Small per-farm offset so farms in one corridor are not identical.
            # A *stable* digest, not `hash()`: Python salts string hashing per
            # process, so `hash(f.id)` made every wind figure -- and therefore every
            # benchmark number downstream -- depend on which interpreter produced it,
            # which breaks the PRD's bit-for-bit replay requirement.
            offset = 1.0 + 0.06 * ((_stable_hash(f.id) % 7) - 3) / 3.0
            # Round first, then derive output from the reported value, so telemetry is
            # self-consistent: an operator can check MW against the m/s on screen.
            speed = round(max(0.0, self.wind_speed[f.corridor] * offset), 3)
            speeds[f.id] = speed
            frac = curve.output_fraction(speed)
            if self._outage(f.id):
                frac = 0.0
            out[f.id] = round(f.capacity_mw * frac, 4)
        return out, speeds

    def _outage(self, asset_id: str) -> bool:
        for mw in self.maintenance:
            if mw.asset_id == asset_id and mw.status == "in_progress":
                return True
        return False

    def demand(self) -> tuple[dict[str, float], float]:
        p = self.config.portfolio
        shape = demand_shape(self.minute_of_day)
        surge = 1.0
        ev = self._active(EventType.DEMAND_SURGE)
        if ev is not None:
            surge = ev.magnitude
        noise = 1.0 + self.demand_noise

        consumers: dict[str, float] = {}
        for c in p.consumers:
            if c.critical:
                # Data centre: flat, barely diurnal.
                level = c.base_mw * (0.96 + 0.04 * shape)
            else:
                level = c.base_mw * (0.70 + 0.30 * shape) * surge
            consumers[c.id] = round(max(0.0, level * noise), 4)

        muni = p.municipal
        municipal = muni.min_mw + (muni.max_mw - muni.min_mw) * shape
        municipal = round(max(0.0, municipal * noise * surge), 4)
        return consumers, municipal

    def _compute_price(self) -> float:
        p = self.config.portfolio
        m = self.config.market
        solar = sum(self.solar_output().values())
        wind, _ = self.wind_output()
        consumers, municipal = self.demand()
        residual = (sum(consumers.values()) + municipal) - (solar + sum(wind.values()))
        frac = residual / max(1.0, p.grid.import_limit_mw)
        base = price_from_residual(frac, m.price["typical_per_mwh"], self.minute_of_day)
        price = base * (1.0 + self.price_noise)
        ev = self._active(EventType.PRICE_SPIKE)
        if ev is not None:
            # A spike lifts the price to a scarcity level. Multiplying would make an
            # already-negative surplus price *more* negative, which is not a spike.
            price = max(price, m.price["typical_per_mwh"] * ev.magnitude)
        return float(np.clip(price, m.price["floor_per_mwh"], m.price["cap_per_mwh"]))

    def current_limits(self) -> tuple[float, float, dict[str, float]]:
        g = self.config.portfolio.grid
        imp, exp = g.import_limit_mw, g.export_limit_mw
        corridors = {c: g.north_corridor_limit_mw if c == "north" else 1e6
                     for c in self.config.portfolio.corridors}
        ev = self._active(EventType.LINE_CONGESTION)
        if ev is not None:
            exp = g.export_derated_mw
            target = ev.corridor or "north"
            if target in corridors:
                corridors[target] = g.north_corridor_derated_mw
        return imp, exp, corridors

    # ---- the interface the agent sees -------------------------------------
    def telemetry(self) -> Telemetry:
        p = self.config.portfolio
        wind, speeds = self.wind_output()
        consumers, municipal = self.demand()
        imp, exp, corridors = self.current_limits()
        return Telemetry(
            interval=self.interval,
            minute_of_day=self.minute_of_day,
            day=self.day,
            solar_mw=self.solar_output(),
            wind_mw=wind,
            wind_speed_ms=speeds,
            cloud_cover=dict(self.cloud),
            consumer_mw=consumers,
            municipal_mw=municipal,
            battery_soc_pct=dict(self.battery_soc_pct),
            battery_available=dict(self.battery_available),
            frequency_hz=round(self.frequency_hz, 4),
            price_per_mwh=round(self._price, 4),
            carbon_price_per_tonne=self.config.market.carbon["price_per_tonne"],
            import_limit_mw=imp,
            export_limit_mw=exp,
            corridor_limit_mw=corridors,
            # Copied, not aliased: these objects mutate as the world advances, and a
            # replayed or logged interval must show the status it had at the time.
            events=tuple(replace(e) for e in self.active_events()),
            maintenance=tuple(replace(m) for m in self.maintenance),
            price_history=tuple(self.price_history),
            shift_debt_mwh=round(self.shift_debt_mwh, 4),
            dr_calls_today=self.dr_calls_today,
            missing=self.missing_channels,
        )

    # ---- advancing --------------------------------------------------------
    def apply(self, dispatch: AppliedDispatch) -> None:
        """Move physical state by what the gateway executed."""
        p = self.config.portfolio
        h = p.interval_hours
        lim = p.battery_limits
        for b in p.batteries:
            mw = dispatch.battery_mw.get(b.id, 0.0)
            if not self.battery_available[b.id]:
                continue
            if mw >= 0:   # discharging: energy out of the cell is mw*h/eff
                delta_mwh = -(mw * h) / b.one_way_efficiency
            else:         # charging: energy into the cell is |mw|*h*eff
                delta_mwh = (-mw * h) * b.one_way_efficiency
            soc = self.battery_soc_pct[b.id] + 100.0 * delta_mwh / b.energy_mwh
            self.battery_soc_pct[b.id] = float(np.clip(soc, lim.soc_min_pct, lim.soc_max_pct))

        self.shift_debt_mwh = max(0.0, self.shift_debt_mwh + dispatch.shift_mw * h)
        if dispatch.dr_mw > 0:
            self.dr_calls_today += 1

    def step(self) -> None:
        """Advance one 15-minute interval."""
        self.interval += 1
        if self.interval % INTERVALS_PER_DAY == 0:
            self.dr_calls_today = 0

        ps = self.preset
        rng = self._rng
        storm = self._storm_on_site()

        # Cloud cover per region: independent shocks, so clouds hit farms unevenly.
        cloud_target = ps.base_cloud + (0.35 if storm else 0.0)
        cparams = OUParams(reversion=0.18, volatility=0.055 * ps.cloud_volatility, lo=0.0, hi=1.0)
        for region in self.cloud:
            self.cloud[region] = float(
                ou_step(self.cloud[region], cloud_target, cparams, rng.standard_normal())
            )
        ev = self._active(EventType.CLOUD_BANK)
        if ev is not None and ev.region in self.cloud:
            self.cloud[ev.region] = float(
                max(self.cloud[ev.region], min(1.0, ev.magnitude))
            )

        # Wind speed per corridor.
        wind_target = ps.base_wind_ms * (1.9 if storm else 1.0)
        wparams = OUParams(reversion=0.14, volatility=0.75 * ps.wind_volatility, lo=0.0, hi=40.0)
        for corridor in self.wind_speed:
            self.wind_speed[corridor] = float(
                ou_step(self.wind_speed[corridor], wind_target, wparams, rng.standard_normal())
            )
        ev = self._active(EventType.WIND_SURGE)
        if ev is not None:
            target = ev.corridor or next(iter(self.wind_speed))
            if target in self.wind_speed:
                self.wind_speed[target] = float(
                    min(40.0, self.wind_speed[target] * ev.magnitude)
                )

        # Demand and price noise.
        self.demand_noise = float(
            ou_step(self.demand_noise, 0.0,
                    OUParams(0.25, 0.012 * ps.demand_volatility, -0.05, 0.05),
                    rng.standard_normal())
        )
        self.price_noise = float(
            ou_step(self.price_noise, 0.0,
                    OUParams(0.22, 0.10 * ps.price_volatility, -0.6, 2.5),
                    rng.standard_normal())
        )

        # Frequency: tight around nominal, with injected dips.
        g = self.config.portfolio.grid
        dip = self._active(EventType.FREQUENCY_DIP)
        if dip is not None:
            self.frequency_hz = g.nominal_frequency_hz - dip.magnitude
        else:
            self.frequency_hz = float(
                np.clip(g.nominal_frequency_hz + 0.012 * rng.standard_normal(), 49.5, 50.5)
            )

        # Battery trips release when the event expires; SoC is unchanged while tripped.
        for bid in self.battery_available:
            tripped = any(
                e.type is EventType.BATTERY_TRIP
                and e.active_at(self.interval)
                and (not e.affected_assets or bid in e.affected_assets)
                for e in self.events
            )
            self.battery_available[bid] = not tripped

        # Data dropout expiry.
        if self._dropout_until is not None and self.interval >= self._dropout_until:
            self.missing_channels = ()
            self._dropout_until = None

        # Maintenance progression.
        for mw in self.maintenance:
            if mw.status == "planned" and self.interval >= mw.start_interval:
                mw.status = "in_progress"
            elif mw.status == "in_progress" and self.interval >= mw.end_interval:
                mw.status = "complete"

        self._price = self._compute_price()
        self.price_history.append(self._price)
