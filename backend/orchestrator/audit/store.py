"""Append-only decision log with replay (PRD: Observability, Determinism and audit).

Every decision is stored with its input snapshot, seed, forecasts, regimes, weights,
guardrail removals, top clusters, actions and LLM tool calls -- enough to replay the
interval bit-for-bit.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"


def _plain(value: Any) -> Any:
    """JSON-safe conversion that never silently drops information."""
    import numpy as np
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "as_dict"):
        return _plain(value.as_dict())
    if hasattr(value, "value"):           # Enum
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    return str(value)


@dataclass
class AuditEntry:
    kind: str               # decision | event | approval | override | fallback | alarm
    interval: int
    body: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "interval": self.interval,
                           "body": _plain(self.body)}, sort_keys=True)


class AuditLog:
    """Append-only. Entries are never mutated or removed."""

    def __init__(self, path: Path | None = None, *, keep_in_memory: bool = True) -> None:
        self.path = path
        self.entries: list[AuditEntry] = [] if keep_in_memory else []
        self._keep = keep_in_memory
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, interval: int, body: dict[str, Any]) -> AuditEntry:
        entry = AuditEntry(kind, interval, body)
        if self._keep:
            self.entries.append(entry)
        if self.path is not None:
            with self.path.open("a") as fh:
                fh.write(entry.to_json() + "\n")
        return entry

    def decisions(self) -> Iterator[AuditEntry]:
        return (e for e in self.entries if e.kind == "decision")

    def at(self, interval: int) -> list[AuditEntry]:
        return [e for e in self.entries if e.interval == interval]

    def replay_key(self, interval: int) -> dict[str, Any] | None:
        """The inputs needed to reproduce an interval exactly."""
        for e in self.entries:
            if e.kind == "decision" and e.interval == interval:
                return {
                    "interval": interval,
                    "seed": e.body.get("seed"),
                    "preset": e.body.get("preset"),
                    "world_seed": e.body.get("world_seed"),
                    "chosen_index": e.body.get("chosen_index"),
                    "config_version": e.body.get("config_version"),
                }
        return None

    def __len__(self) -> int:
        return len(self.entries)


def decision_record(decision, actions=None, gateway_result=None, *,
                    preset: str | None = None, world_seed: int | None = None,
                    config_version: int | None = None) -> dict[str, Any]:
    """The full decision trace, as stored. This is what the console renders and what
    replay reads back."""
    snap = decision.snapshot
    body: dict[str, Any] = {
        "interval": decision.interval,
        "seed": decision.seed,
        "preset": preset,
        "world_seed": world_seed,
        "config_version": config_version,
        "minute_of_day": snap.minute_of_day,
        "latency_ms": round(decision.latency_ms, 3),
        "clusters_evaluated": decision.clusters_evaluated,
        "scenarios_evaluated": decision.scenarios_evaluated,
        "steps_evaluated": decision.steps_evaluated,
        "state": {
            "solar_mw": snap.solar_mw, "wind_mw": snap.wind_mw,
            "wind_speed_ms": snap.wind_speed_ms, "cloud_cover": snap.cloud_cover,
            "consumer_mw": snap.consumer_mw, "municipal_mw": snap.municipal_mw,
            "battery_soc_pct": snap.battery_soc_pct,
            "battery_available": snap.battery_available,
            "frequency_hz": snap.frequency_hz, "price_per_mwh": snap.price_per_mwh,
            "import_limit_mw": snap.import_limit_mw,
            "export_limit_mw": snap.export_limit_mw,
            "corridor_limit_mw": snap.corridor_limit_mw,
            "quality": snap.quality,
        },
        "events": [e for e in snap.events],
        "criteria": decision.criteria,
        "guardrails": [g for g in decision.guardrails],
        "surviving": list(decision.surviving),
        "chosen_index": decision.chosen_index,
        "chosen": decision.chosen,
        "held_by_hysteresis": decision.held_by_hysteresis,
        "top5": [s for s in decision.top5],
        "pareto_front": list(decision.pareto_front),
        "committed": decision.committed,
        "explanation": decision.explanation,
        "explanation_source": decision.explanation_source,
        "tool_calls": list(decision.tool_calls),
        "fallback_reason": decision.fallback_reason,
        "auto_approved": decision.auto_approved,
    }
    if actions is not None:
        body["actions"] = [sp for sp in actions.set_points]
        body["queued_for_approval"] = [sp for sp in actions.queued_for_approval]
    if gateway_result is not None:
        body["gated"] = [g for g in gateway_result.gated]
    return body
