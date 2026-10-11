"""The seven agent tools (PRD "Agent tools").

Arguments are bounded by JSON schema and the handlers only ever return optimizer
output. The orchestrator cannot write a set-point: `execute_actions` commits the
cluster the optimizer already chose, and nothing accepts an MW figure as input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

TOOL_NAMES = (
    "get_state", "run_forecast", "evaluate_options", "execute_actions",
    "plan_maintenance", "dispatch_crew", "explain",
)

# Tools with side effects, and which of those need operator approval.
SIDE_EFFECT_TOOLS = frozenset({"execute_actions", "plan_maintenance", "dispatch_crew"})
APPROVAL_TOOLS = frozenset({"plan_maintenance", "dispatch_crew"})


def tool_schemas() -> list[dict[str, Any]]:
    """JSON-schema definitions. `strict` so inputs validate exactly.

    Strict mode rejects numeric (`minimum`/`maximum`/`multipleOf`) and string
    (`minLength`/`maxLength`) constraints, so none appear below -- the API 400s on
    the whole request if one does. Bounds belong in the handlers, which clamp.
    """
    return [
        {
            "name": "get_state",
            "description": (
                "Current snapshot of all assets, limits, prices and alerts for the "
                "interval being decided. No side effects. Call this first."),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "include_forecast_quality": {
                        "type": "boolean",
                        "description": "Include staleness and substitution flags."},
                },
                "required": ["include_forecast_quality"],
                "additionalProperties": False,
            },
        },
        {
            "name": "run_forecast",
            "description": (
                "P10/P50/P90 quantile forecasts and the scenario fan for the "
                "look-ahead. No side effects."),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "horizon_hours": {
                        "type": "integer",
                        "description": "Look-ahead in hours; clamped to the configured horizon."},
                },
                "required": ["horizon_hours"],
                "additionalProperties": False,
            },
        },
        {
            "name": "evaluate_options",
            "description": (
                "Score all 80 action clusters under the current criteria. Returns the "
                "ranking, the guardrail results and the Pareto front. This is the only "
                "source of MW figures; no side effects."),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you are (re-)evaluating now."},
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "execute_actions",
            "description": (
                "Send the chosen cluster's set-points to the actuation gateway. You do "
                "not supply the numbers: the gateway commits the cluster that "
                "evaluate_options selected. Changes the grid; bounded, logged, and "
                "approval-gated when high-impact."),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to commit."},
                    "reason": {"type": "string"},
                },
                "required": ["confirm", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "plan_maintenance",
            "description": (
                "Value maintenance windows against the forecast and propose delays. "
                "Proposals need operator approval."),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "dispatch_crew",
            "description": "Raise an inspection work order. Needs operator approval.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["asset_id", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": "explain",
            "description": (
                "Build the trade-off explanation from the decision trace. Returns the "
                "regimes, the base and final weights, the guardrail removals, the top "
                "clusters and the runner-up comparison. No side effects."),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "audience": {
                        "type": "string", "enum": ["operator", "manager", "engineer"],
                        "description": "Who the explanation is for."},
                },
                "required": ["audience"],
                "additionalProperties": False,
            },
        },
    ]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result_summary: str
    ok: bool = True
    side_effect: bool = False
    needs_approval: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "arguments": self.arguments,
            "result": self.result_summary, "ok": self.ok,
            "side_effect": self.side_effect, "needs_approval": self.needs_approval,
            "error": self.error,
        }


@dataclass
class ToolBox:
    """Binds the seven tools to one interval's context."""

    handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]]
    calls: list[ToolCall] = field(default_factory=list)

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self.handlers:
            call = ToolCall(name, arguments, "", ok=False,
                            error=f"unknown tool {name!r}")
            self.calls.append(call)
            return {"error": call.error}
        try:
            result = self.handlers[name](arguments)
        except Exception as exc:
            call = ToolCall(name, arguments, "", ok=False, error=str(exc),
                            side_effect=name in SIDE_EFFECT_TOOLS)
            self.calls.append(call)
            return {"error": str(exc)}
        self.calls.append(ToolCall(
            name, arguments,
            result_summary=str(result.get("summary", ""))[:400],
            side_effect=name in SIDE_EFFECT_TOOLS,
            needs_approval=name in APPROVAL_TOOLS,
        ))
        return result

    @property
    def trace(self) -> tuple[dict[str, Any], ...]:
        return tuple(c.as_dict() for c in self.calls)

    def called(self, name: str) -> bool:
        return any(c.name == name and c.ok for c in self.calls)
