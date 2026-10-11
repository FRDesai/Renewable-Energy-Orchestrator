"""Actuation gateway: bounds check, approval queue, signed set-points (PRD stage 8).

The LLM never reaches past this. Set-points arrive from the optimizer, are checked
against configured bounds, and either execute or queue for one-click approval.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from ..config_loader import Config
from ..domain.actions import CommittedActions, SetPoint

HIGH_IMPACT_VERBS = frozenset({"shed", "reduce"})


class Verdict(str, Enum):
    EXECUTED = "executed"
    QUEUED = "queued"
    REJECTED = "rejected"


@dataclass(frozen=True)
class GatedSetPoint:
    set_point: SetPoint
    verdict: Verdict
    reason: str
    signature: str

    def as_dict(self) -> dict[str, object]:
        return {**self.set_point.as_dict(), "verdict": self.verdict.value,
                "gate_reason": self.reason, "signature": self.signature}


@dataclass
class ApprovalRequest:
    id: str
    interval: int
    set_point: SetPoint
    reason: str
    status: str = "pending"      # pending | approved | rejected
    decided_by: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {"id": self.id, "interval": self.interval, "status": self.status,
                "reason": self.reason, "decided_by": self.decided_by,
                **self.set_point.as_dict()}


@dataclass
class GatewayResult:
    gated: tuple[GatedSetPoint, ...]
    queued: tuple[ApprovalRequest, ...]
    executed: tuple[SetPoint, ...]
    rejected: tuple[GatedSetPoint, ...]

    @property
    def any_queued(self) -> bool:
        return bool(self.queued)


class ActuationGateway:
    """Bounds-checks and gates every set-point. Nothing reaches the plant otherwise."""

    def __init__(self, config: Config, *, auto_approve: bool = False) -> None:
        self.config = config
        self.auto_approve = auto_approve      # batch/Monte-Carlo mode (DECISIONS.md D9)
        self.queue: list[ApprovalRequest] = []
        self.overrides: list[dict] = []
        self._counter = 0

    # ---- bounds -----------------------------------------------------------
    def _bounds_violation(self, sp: SetPoint) -> str | None:
        p = self.config.portfolio
        if sp.mw < -1e-9:
            return "negative magnitude"
        if sp.verb in ("charge", "discharge"):
            try:
                b = p.battery(sp.asset_id)
            except StopIteration:
                return f"unknown battery {sp.asset_id}"
            if sp.mw > b.power_mw + 1e-6:
                return f"{sp.mw:.1f} MW exceeds the {b.power_mw:.0f} MW rating"
        if sp.verb == "buy" and sp.mw > p.grid.import_limit_mw + 1e-6:
            return f"{sp.mw:.1f} MW exceeds the {p.grid.import_limit_mw:.0f} MW import limit"
        if sp.verb == "sell" and sp.mw > p.grid.export_limit_mw + 1e-6:
            return f"{sp.mw:.1f} MW exceeds the {p.grid.export_limit_mw:.0f} MW export limit"
        if sp.verb == "curtail":
            cap = None
            for f in p.solar_farms + p.wind_farms:
                if f.id == sp.asset_id:
                    cap = f.capacity_mw
            if cap is None:
                return f"unknown generator {sp.asset_id}"
            if sp.mw > cap + 1e-6:
                return f"{sp.mw:.1f} MW exceeds {sp.asset_name}'s {cap:.0f} MW capacity"
        if sp.verb == "reduce":
            try:
                c = p.consumer(sp.asset_id)
            except StopIteration:
                return f"unknown consumer {sp.asset_id}"
            if c.critical:
                return f"{c.name} is a critical load and is never curtailed by the agent"
        if sp.verb in ("shift", "restore"):
            try:
                c = p.consumer(sp.asset_id)
            except StopIteration:
                return f"unknown consumer {sp.asset_id}"
            if c.critical:
                return f"{c.name} is a critical load and cannot be shifted"
            if sp.mw > c.shiftable_mw + 1e-6:
                return f"{sp.mw:.1f} MW exceeds {c.name}'s {c.shiftable_mw:.0f} MW shiftable block"
        return None

    @staticmethod
    def _sign(interval: int, sp: SetPoint) -> str:
        """Signed command: a stable digest over the exact instruction."""
        payload = json.dumps(
            {"interval": interval, "verb": sp.verb, "asset": sp.asset_id,
             "mw": round(sp.mw, 6)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ---- the one entry point ---------------------------------------------
    def submit(self, actions: CommittedActions) -> GatewayResult:
        gated: list[GatedSetPoint] = []
        queued: list[ApprovalRequest] = []
        executed: list[SetPoint] = []
        rejected: list[GatedSetPoint] = []

        for sp in actions.set_points + actions.queued_for_approval:
            sig = self._sign(actions.interval, sp)
            violation = self._bounds_violation(sp)
            if violation is not None:
                g = GatedSetPoint(sp, Verdict.REJECTED,
                                  f"Out of configured bounds: {violation}", sig)
                gated.append(g)
                rejected.append(g)
                continue

            needs = sp.requires_approval or sp.verb in HIGH_IMPACT_VERBS
            if needs and not self.auto_approve:
                self._counter += 1
                req = ApprovalRequest(
                    id=f"ap-{actions.interval}-{self._counter}",
                    interval=actions.interval, set_point=sp,
                    reason=sp.reason)
                self.queue.append(req)
                queued.append(req)
                gated.append(GatedSetPoint(
                    sp, Verdict.QUEUED, "High-impact: awaiting operator approval", sig))
                continue

            note = "Routine set-point, executed automatically"
            if needs and self.auto_approve:
                note = "High-impact, auto-approved in batch mode (simulated run)"
            gated.append(GatedSetPoint(sp, Verdict.EXECUTED, note, sig))
            executed.append(sp)

        return GatewayResult(tuple(gated), tuple(queued), tuple(executed), tuple(rejected))

    # ---- operator actions -------------------------------------------------
    def decide(self, request_id: str, approve: bool, *, by: str = "operator") -> ApprovalRequest:
        req = next((r for r in self.queue if r.id == request_id), None)
        if req is None:
            raise LookupError(f"no approval request {request_id!r}")
        if req.status != "pending":
            # Deciding a request twice, or one whose plan has been superseded, would
            # dispatch a set-point no current plan called for.
            raise ValueError(
                f"approval request {request_id!r} is {req.status} and can no longer "
                f"be actioned")
        req.status = "approved" if approve else "rejected"
        req.decided_by = by
        # A rejection is preference data (PRD: Trainability).
        self.overrides.append({
            "kind": "approval", "id": req.id, "interval": req.interval,
            "verb": req.set_point.verb, "asset": req.set_point.asset_id,
            "mw": req.set_point.mw, "approved": approve, "by": by,
        })
        return req

    def withdraw_through(self, interval: int, reason: str) -> tuple[ApprovalRequest, ...]:
        """Retire pending requests belonging to interval `interval` or earlier.

        An approval request describes one interval's plan. Once that plan has been
        superseded by a re-plan, or the clock has moved past it, approving it would
        dispatch a set-point no current plan called for -- so it leaves the queue
        rather than lingering as something the operator can still click.
        """
        retired = tuple(r for r in self.queue
                        if r.status == "pending" and r.interval <= interval)
        for req in retired:
            req.status = "superseded"
            req.decided_by = reason
        return retired

    def override(self, interval: int, note: str, *, by: str = "operator") -> dict:
        """Operator override always wins and is always logged."""
        entry = {"kind": "override", "interval": interval, "note": note, "by": by}
        self.overrides.append(entry)
        return entry

    @property
    def pending(self) -> tuple[ApprovalRequest, ...]:
        return tuple(r for r in self.queue if r.status == "pending")
