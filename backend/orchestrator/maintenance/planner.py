"""Maintenance planner: a separate planner on the same forecast (PRD "Maintenance actions").

Delay a planned window when it would lose more than $2,500 of output compared with the
best window before its deadline; delay any window overlapping a storm (crew safety);
start and close work orders; dispatch inspection teams after a trip or cut-out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config_loader import Config
from ..domain.events import EventType, MaintenanceWindow, WorkOrder
from ..domain.state import StateSnapshot


@dataclass(frozen=True)
class WindowValuation:
    window_id: str
    asset_id: str
    asset_name: str
    current_start: int
    best_start: int
    current_loss_usd: float
    best_loss_usd: float
    saving_usd: float
    storm_overlap: bool
    recommend_delay: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id, "asset_id": self.asset_id,
            "asset_name": self.asset_name, "current_start": self.current_start,
            "best_start": self.best_start,
            "current_loss_usd": round(self.current_loss_usd, 0),
            "best_loss_usd": round(self.best_loss_usd, 0),
            "saving_usd": round(self.saving_usd, 0),
            "storm_overlap": self.storm_overlap,
            "recommend_delay": self.recommend_delay, "reason": self.reason,
        }


@dataclass
class MaintenancePlan:
    valuations: tuple[WindowValuation, ...] = ()
    work_orders: tuple[WorkOrder, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def proposed_delays(self) -> tuple[WindowValuation, ...]:
        return tuple(v for v in self.valuations if v.recommend_delay)


class MaintenancePlanner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.min_saving = float(config.objectives.maintenance["min_saving_to_delay_usd"])
        self.storm_forbidden = bool(
            config.objectives.maintenance.get("storm_overlap_forbidden", True))

    def _asset_name(self, asset_id: str) -> str:
        p = self.config.portfolio
        for f in p.solar_farms:
            if f.id == asset_id:
                return f.name
        for f in p.wind_farms:
            if f.id == asset_id:
                return f.name
        for b in p.batteries:
            if b.id == asset_id:
                return b.name
        return asset_id

    def _lost_output_usd(
        self, asset_id: str, start: int, duration: int,
        snapshot: StateSnapshot, price_path: np.ndarray, output_path: np.ndarray,
    ) -> float:
        """Value of output forgone if the window runs from `start`."""
        h = self.config.portfolio.interval_hours
        total = 0.0
        for k in range(duration):
            idx = start - snapshot.interval + k
            if 0 <= idx < len(price_path):
                total += float(output_path[idx]) * h * max(0.0, float(price_path[idx]))
        return total

    def plan(
        self,
        snapshot: StateSnapshot,
        *,
        price_path: np.ndarray,
        asset_output: dict[str, np.ndarray],
    ) -> MaintenancePlan:
        """`price_path` and `asset_output` are the median forecast over the horizon."""
        valuations: list[WindowValuation] = []
        notes: list[str] = []
        storm = snapshot.storm_alert

        for mw in snapshot.maintenance:
            if mw.status in ("complete", "in_progress"):
                continue
            output = asset_output.get(mw.asset_id)
            if output is None:
                continue
            horizon = len(price_path)
            latest = mw.deadline_interval - mw.duration_intervals
            candidates = [
                s for s in range(snapshot.interval, snapshot.interval + horizon)
                if s <= latest
            ]
            if not candidates:
                notes.append(
                    f"{self._asset_name(mw.asset_id)}: deadline interval "
                    f"{mw.deadline_interval} leaves no alternative window")
                continue

            losses = {
                s: self._lost_output_usd(
                    mw.asset_id, s, mw.duration_intervals, snapshot, price_path, output)
                for s in candidates
            }
            current = losses.get(
                mw.start_interval,
                self._lost_output_usd(mw.asset_id, mw.start_interval,
                                      mw.duration_intervals, snapshot, price_path, output))
            best_start = min(losses, key=lambda s: losses[s])
            best = losses[best_start]
            saving = current - best

            overlap = False
            if storm is not None and self.storm_forbidden:
                s_start = storm.arrival_interval or storm.start_interval
                s_end = storm.end_interval or (s_start + 1)
                overlap = mw.overlaps(s_start, s_end)

            if overlap:
                # Crew safety is not an economic trade-off; move it regardless.
                safe = [s for s in candidates
                        if not MaintenanceWindow(
                            mw.id, mw.asset_id, s, mw.duration_intervals,
                            mw.deadline_interval, mw.crew).overlaps(s_start, s_end)]
                target = min(safe, key=lambda s: losses[s]) if safe else best_start
                reason = (f"Window overlaps the forecast storm "
                          f"({s_start}-{s_end}); crew safety requires a move")
                recommend = True
                best_start, best = target, losses[target]
                saving = current - best
            elif saving > self.min_saving:
                reason = (f"Delaying to interval {best_start} saves "
                          f"${saving:,.0f} of lost output, above the "
                          f"${self.min_saving:,.0f} bar")
                recommend = True
            else:
                reason = (f"Keeping interval {mw.start_interval}: moving would save only "
                          f"${max(saving, 0):,.0f}, below the ${self.min_saving:,.0f} bar")
                recommend = False

            valuations.append(WindowValuation(
                window_id=mw.id, asset_id=mw.asset_id,
                asset_name=self._asset_name(mw.asset_id),
                current_start=mw.start_interval, best_start=best_start,
                current_loss_usd=current, best_loss_usd=best, saving_usd=saving,
                storm_overlap=overlap, recommend_delay=recommend, reason=reason))

        return MaintenancePlan(valuations=tuple(valuations), notes=tuple(notes))

    def inspections_for(self, snapshot: StateSnapshot) -> tuple[WorkOrder, ...]:
        """Dispatch inspection teams after a battery trip or a turbine cut-out."""
        orders: list[WorkOrder] = []
        curve = self.config.portfolio.wind_curve
        for e in snapshot.events:
            if e.type is EventType.BATTERY_TRIP and e.start_interval == snapshot.interval:
                for asset in (e.affected_assets or tuple(snapshot.battery_available)):
                    orders.append(WorkOrder(
                        id=f"wo-{snapshot.interval}-{asset}", asset_id=asset,
                        reason="Battery trip: inspect before returning to service",
                        raised_interval=snapshot.interval))
        for farm in self.config.portfolio.wind_farms:
            speed = snapshot.wind_speed_ms.get(farm.id, 0.0)
            if speed >= curve.cut_out_ms:
                orders.append(WorkOrder(
                    id=f"wo-{snapshot.interval}-{farm.id}", asset_id=farm.id,
                    reason=f"Turbine cut-out at {speed:.1f} m/s: inspect for damage",
                    raised_interval=snapshot.interval))
        return tuple(orders)
