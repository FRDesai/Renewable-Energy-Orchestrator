"""Load versioned configuration. Every change is logged (PRD: Configurability)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .domain.assets import (
    Battery,
    BatteryLimits,
    GridConfig,
    IndustrialConsumer,
    MunicipalLoad,
    Portfolio,
    SolarFarm,
    SolarResponse,
    WindCurve,
    WindFarm,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass(frozen=True)
class Regime:
    key: str
    label: str
    trigger: str
    multipliers: dict[str, float]
    risk_aversion: float


@dataclass(frozen=True)
class Objectives:
    version: int
    lookahead_hours: float
    steps: int
    scenarios: int
    base_weights: dict[str, float]
    reliability: dict[str, float]
    risk: dict[str, float]
    stored_energy: dict[str, float]
    range_floor: dict[str, float]
    hysteresis_margin: float
    guardrails: dict[str, Any]
    regimes: dict[str, Regime]
    thresholds: dict[str, float]
    maintenance: dict[str, Any]
    autonomy: dict[str, Any]
    persistence_hours: dict[str, float]
    storm: dict[str, float]

    @property
    def objective_keys(self) -> tuple[str, ...]:
        return tuple(self.base_weights.keys())


@dataclass(frozen=True)
class Market:
    version: int
    price: dict[str, float]
    carbon: dict[str, Any]
    demand_response: dict[str, float]
    load_shift: dict[str, float]
    value_of_lost_load: dict[str, float]


@dataclass(frozen=True)
class Config:
    portfolio: Portfolio
    objectives: Objectives
    market: Market
    source_dir: Path
    change_log: tuple[str, ...] = ()

    @property
    def objective_keys(self) -> tuple[str, ...]:
        return self.objectives.objective_keys

    def with_base_weights(self, weights: dict[str, float], *, note: str = "") -> "Config":
        """Operator slider change. Returns a new Config and logs the change."""
        merged = dict(self.objectives.base_weights)
        for key, value in weights.items():
            if key not in merged:
                raise KeyError(f"unknown objective {key!r}")
            merged[key] = float(value)
        total = sum(merged.values())
        if total <= 0:
            raise ValueError("base weights must sum to a positive number")
        merged = {k: v / total for k, v in merged.items()}
        entry = json.dumps({"change": "base_weights", "to": merged, "note": note}, sort_keys=True)
        return replace(
            self,
            objectives=replace(self.objectives, base_weights=merged),
            change_log=self.change_log + (entry,),
        )

    def with_risk_lambda(self, lam: float, *, note: str = "") -> "Config":
        risk = dict(self.objectives.risk)
        risk["lambda_base"] = float(lam)
        entry = json.dumps({"change": "lambda_base", "to": float(lam), "note": note}, sort_keys=True)
        return replace(
            self,
            objectives=replace(self.objectives, risk=risk),
            change_log=self.change_log + (entry,),
        )

    def with_regime_multiplier(
        self, regime_key: str, objective: str, value: float, *, note: str = ""
    ) -> "Config":
        """Trainability: regime multipliers re-tuned from accepted operator overrides."""
        regimes = dict(self.objectives.regimes)
        regime = regimes[regime_key]
        mults = dict(regime.multipliers)
        mults[objective] = float(value)
        regimes[regime_key] = replace(regime, multipliers=mults)
        entry = json.dumps(
            {"change": "regime_multiplier", "regime": regime_key,
             "objective": objective, "to": float(value), "note": note},
            sort_keys=True,
        )
        return replace(
            self,
            objectives=replace(self.objectives, regimes=regimes),
            change_log=self.change_log + (entry,),
        )


def _read(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_portfolio(raw: dict[str, Any]) -> Portfolio:
    curve = raw["wind_power_curve"]
    lim = raw["battery_limits"]
    grid = raw["grid"]
    muni = raw["municipal_load"]
    return Portfolio(
        version=int(raw["version"]),
        interval_minutes=int(raw["interval_minutes"]),
        solar_farms=tuple(SolarFarm(**f) for f in raw["solar_farms"]),
        wind_farms=tuple(WindFarm(**f) for f in raw["wind_farms"]),
        solar_response=SolarResponse(**raw["solar_response"]),
        wind_curve=WindCurve(**curve),
        batteries=tuple(Battery(**b) for b in raw["batteries"]),
        battery_limits=BatteryLimits(**lim),
        consumers=tuple(IndustrialConsumer(**c) for c in raw["industrial_consumers"]),
        municipal=MunicipalLoad(**muni),
        grid=GridConfig(**grid),
    )


def load_objectives(raw: dict[str, Any]) -> Objectives:
    weights = {k: float(v) for k, v in raw["base_weights"].items()}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    regimes = {
        key: Regime(
            key=key,
            label=body["label"],
            trigger=body["trigger"],
            multipliers={k: float(v) for k, v in body["multipliers"].items()},
            risk_aversion=float(body.get("risk_aversion", 0.0) or 0.0),
        )
        for key, body in raw["regimes"].items()
    }
    unknown = {
        obj
        for r in regimes.values()
        for obj in r.multipliers
        if obj not in weights
    }
    if unknown:
        raise ValueError(f"regime multipliers reference unknown objectives: {sorted(unknown)}")
    return Objectives(
        version=int(raw["version"]),
        lookahead_hours=float(raw["horizon"]["lookahead_hours"]),
        steps=int(raw["horizon"]["steps"]),
        scenarios=int(raw["horizon"]["scenarios"]),
        base_weights=weights,
        reliability={k: float(v) for k, v in raw["reliability"].items()},
        risk={k: float(v) for k, v in raw["risk"].items()},
        stored_energy={k: float(v) for k, v in raw["stored_energy"].items()},
        range_floor={k: float(v) for k, v in raw["scoring"]["range_floor"].items()},
        hysteresis_margin=float(raw["scoring"]["hysteresis_margin"]),
        guardrails=dict(raw["guardrails"]),
        regimes=regimes,
        thresholds={k: float(v) for k, v in raw["thresholds"].items()},
        maintenance=dict(raw["maintenance"]),
        autonomy=dict(raw["autonomy"]),
        persistence_hours={k: float(v) for k, v in raw["persistence_hours"].items()},
        storm={k: float(v) for k, v in raw["storm"].items()},
    )


def load_market(raw: dict[str, Any]) -> Market:
    return Market(
        version=int(raw["version"]),
        price={k: float(v) for k, v in raw["price"].items()},
        carbon=dict(raw["carbon"]),
        demand_response={k: float(v) for k, v in raw["demand_response"].items()},
        load_shift={k: float(v) for k, v in raw["load_shift"].items()},
        value_of_lost_load={k: float(v) for k, v in raw["value_of_lost_load"].items()},
    )


def load_config(config_dir: str | os.PathLike[str] | None = None) -> Config:
    base = Path(config_dir) if config_dir else CONFIG_DIR
    objectives = load_objectives(_read(base / "objectives.yaml"))
    portfolio = load_portfolio(_read(base / "portfolio.yaml"))
    market = load_market(_read(base / "market.yaml"))

    # Cross-file consistency: the range floor and base weights must cover the same
    # objective set, or scoring would silently skip a dimension.
    if set(objectives.range_floor) != set(objectives.base_weights):
        raise ValueError("scoring.range_floor must cover exactly the base_weights keys")
    if objectives.steps * portfolio.interval_minutes / 60.0 != objectives.lookahead_hours:
        raise ValueError(
            f"horizon mismatch: {objectives.steps} steps x {portfolio.interval_minutes} min "
            f"!= {objectives.lookahead_hours} h"
        )
    return Config(portfolio=portfolio, objectives=objectives, market=market, source_dir=base)
