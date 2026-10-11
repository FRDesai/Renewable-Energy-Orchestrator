"""The HTTP and WebSocket surface the operator console runs on.

A thin shell: every endpoint is a call into `LiveSession`, which holds all the state and
all the physics. The only logic here is request validation, serialisation, the run-loop
that ticks the clock, and the broadcast hub that pushes each new frame to connected
consoles.

The session is not thread-safe and a decision costs tens of milliseconds, so every
mutation runs on a worker thread under one lock. That keeps the event loop responsive
while guaranteeing the session only ever sees one caller at a time.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..audit.store import ARTIFACTS
from ..config_loader import Config, load_config
from ..domain.actions import CLUSTERS
from ..domain.events import EventType
from ..forecast.quantile import QuantileForecaster
from ..sim.episode import STRATEGIES
from ..sim.monte_carlo import PATH_CHOICES, MonteCarlo
from ..world.presets import DEMO_OPENING_INTERVAL, PRESETS
from .session import (
    DEMO_WARMUP_INTERVALS,
    EVENT_LABELS,
    INJECTION_DEFAULTS,
    LiveSession,
    jsonable,
)

MAX_SIM_PATHS = max(PATH_CHOICES)


# ---- request bodies -------------------------------------------------------
class StepBody(BaseModel):
    intervals: int = Field(default=1, ge=1, le=96)


class RunBody(BaseModel):
    speed_x: float = Field(default=1.0, gt=0.0, le=600.0)


class InjectBody(BaseModel):
    type: str
    magnitude: float | None = Field(default=None, gt=0.0, le=50.0)
    duration: int | None = Field(default=None, ge=1, le=96)
    lead: int | None = Field(default=None, ge=0, le=96)
    region: str | None = None
    corridor: str | None = None
    assets: list[str] | None = None
    channels: list[str] | None = None

    def spec(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in ("magnitude", "duration", "lead", "region", "corridor"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.assets:
            out["assets"] = tuple(self.assets)
        if self.channels:
            out["channels"] = tuple(self.channels)
        return out


class BulletinBody(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class ConfirmBody(BaseModel):
    accept: bool


class WeightsBody(BaseModel):
    weights: dict[str, float] | None = None
    risk_lambda: float | None = Field(default=None, ge=0.0, le=5.0)
    note: str = Field(default="operator slider", max_length=200)


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class ApprovalBody(BaseModel):
    approve: bool
    by: str = Field(default="operator", max_length=80)


class OverrideBody(BaseModel):
    note: str = Field(min_length=1, max_length=500)
    by: str = Field(default="operator", max_length=80)


class ResetBody(BaseModel):
    preset: str | None = None
    seed: int | None = Field(default=None, ge=0, le=10**9)
    open_at: int | None = Field(default=None, ge=0, le=95)
    warmup_intervals: int | None = Field(default=None, ge=0, le=96)


class SimulateBody(BaseModel):
    preset: str = "typical_day"
    paths: int = Field(default=6, ge=1, le=MAX_SIM_PATHS)
    intervals: int = Field(default=96, ge=1, le=96)
    strategies: list[Literal["agent", "rule_based", "no_storage"]] | None = None


# ---- the broadcast hub ---------------------------------------------------
class Hub:
    """One queue per connected console. A slow client is dropped, never blocked:
    the control loop must not wait on a browser."""

    def __init__(self, maxsize: int = 8) -> None:
        self.queues: set[asyncio.Queue] = set()
        self.maxsize = maxsize

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.maxsize)
        self.queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.queues.discard(q)

    def publish(self, message: dict[str, Any]) -> None:
        for q in list(self.queues):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                self.queues.discard(q)


def create_app(
    *,
    config: Config | None = None,
    preset: str = "typical_day",
    seed: int = 7,
    open_at: int = DEMO_OPENING_INTERVAL,
    warmup_intervals: int = DEMO_WARMUP_INTERVALS,
    audit_path: Path | None = None,
) -> FastAPI:
    cfg = config or load_config()
    forecaster = QuantileForecaster.load()
    hub = Hub()
    lock = asyncio.Lock()
    state: dict[str, Any] = {"session": None, "runner": None, "jobs": {}}

    def build_session(**overrides: Any) -> LiveSession:
        kwargs = dict(preset=preset, seed=seed, open_at=open_at,
                      warmup_intervals=warmup_intervals, forecaster=forecaster,
                      audit_path=audit_path)
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return LiveSession(cfg, **kwargs)

    def session() -> LiveSession:
        s = state["session"]
        if s is None:                       # pragma: no cover - lifespan guarantees it
            raise HTTPException(503, "session not started")
        return s

    async def run_mutation(fn, *args, **kwargs) -> Any:
        """Serialise every session mutation and broadcast the resulting frame."""
        async with lock:
            result = await asyncio.to_thread(fn, *args, **kwargs)
            frame = await asyncio.to_thread(lambda: session().frame)
        hub.publish({"type": "frame", "frame": frame})
        return result, frame

    async def tick_loop() -> None:
        """The simulated clock. One decision per interval at the chosen speed."""
        try:
            while True:
                s = state["session"]
                if s is None or not s.running:
                    await asyncio.sleep(0.05)
                    continue
                delay = s.tick_seconds
                await asyncio.sleep(delay)
                s = state["session"]
                if s is None or not s.running:
                    continue
                try:
                    await run_mutation(s.step)
                except Exception as exc:    # pragma: no cover - defensive
                    s.running = False
                    hub.publish({"type": "error", "message": str(exc)})
        except asyncio.CancelledError:
            raise

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["session"] = await asyncio.to_thread(build_session)
        state["runner"] = asyncio.create_task(tick_loop())
        try:
            yield
        finally:
            state["runner"].cancel()
            try:
                await state["runner"]
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="Renewable Energy Orchestrator", version="0.1.0",
                  lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
        allow_headers=["*"])

    # ---- read-only ------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        s = state["session"]
        return {
            "status": "ok",
            "clusters": len(CLUSTERS),
            "forecaster_trained": bool(forecaster is not None and forecaster.is_trained),
            "llm_available": s.llm.available if s else False,
            "config_version": {
                "portfolio": cfg.portfolio.version,
                "objectives": cfg.objectives.version,
                "market": cfg.market.version,
            },
            "interval": s.world.interval if s else None,
        }

    @app.get("/api/config")
    def configuration() -> dict[str, Any]:
        p, o, m = cfg.portfolio, cfg.objectives, cfg.market
        return {
            "portfolio": {
                "version": p.version,
                "interval_minutes": p.interval_minutes,
                "solar": [{"id": f.id, "name": f.name, "capacity_mw": f.capacity_mw,
                           "cloud_region": f.cloud_region} for f in p.solar_farms],
                "wind": [{"id": f.id, "name": f.name, "capacity_mw": f.capacity_mw,
                          "corridor": f.corridor} for f in p.wind_farms],
                "batteries": [{"id": b.id, "name": b.name, "power_mw": b.power_mw,
                               "energy_mwh": b.energy_mwh,
                               "one_way_efficiency": b.one_way_efficiency,
                               "wear_cost_per_mwh": b.wear_cost_per_mwh}
                              for b in p.batteries],
                "consumers": [{"id": c.id, "name": c.name, "base_mw": c.base_mw,
                               "shiftable_mw": c.shiftable_mw, "dr_mw": c.dr_mw,
                               "critical": c.critical} for c in p.consumers],
                "battery_limits": jsonable(p.battery_limits),
                "grid": jsonable(p.grid),
                "municipal": jsonable(p.municipal),
                "cloud_regions": list(p.cloud_regions),
                "corridors": list(p.corridors),
            },
            "objectives": {
                "version": o.version,
                "lookahead_hours": o.lookahead_hours,
                "steps": o.steps,
                "scenarios": o.scenarios,
                "base_weights": dict(o.base_weights),
                "risk": dict(o.risk),
                "hysteresis_margin": o.hysteresis_margin,
                "guardrails": jsonable(o.guardrails),
                "regimes": [{"key": r.key, "label": r.label, "trigger": r.trigger,
                             "multipliers": dict(r.multipliers),
                             "risk_aversion": r.risk_aversion}
                            for r in o.regimes.values()],
                "autonomy": jsonable(o.autonomy),
                "maintenance": jsonable(o.maintenance),
            },
            "market": {"version": m.version, "price": dict(m.price),
                       "carbon": jsonable(m.carbon),
                       "demand_response": dict(m.demand_response),
                       "load_shift": dict(m.load_shift),
                       "value_of_lost_load": dict(m.value_of_lost_load)},
            "presets": {k: {"label": v.label, "description": v.description}
                        for k, v in PRESETS.items()},
            "injections": {k.value: {"label": EVENT_LABELS[k],
                                     "defaults": jsonable(v)}
                           for k, v in INJECTION_DEFAULTS.items()},
            "path_choices": list(PATH_CHOICES),
            "strategies": list(STRATEGIES),
            "change_log": list(session().config.change_log) if state["session"] else [],
        }

    @app.get("/api/session")
    def get_session() -> dict[str, Any]:
        return session().frame

    @app.get("/api/session/log")
    def get_log(limit: int = 200) -> dict[str, Any]:
        return {"entries": session().log[:max(1, min(limit, 400))]}

    @app.get("/api/session/approvals")
    def get_approvals() -> dict[str, Any]:
        s = session()
        return {"pending": s.pending_proposals(), "all": s.proposals(),
                "overrides": list(s.gateway.overrides)}

    @app.get("/api/session/replay/{interval}")
    def get_replay(interval: int) -> dict[str, Any]:
        stored = session().replay(interval)
        if stored is None:
            raise HTTPException(404, f"interval {interval} is not in the audit log")
        return stored

    @app.get("/api/reliability")
    def reliability() -> dict[str, Any]:
        return _artifact("reliability.json",
                         hint="run `make episodes` to produce it")

    @app.get("/api/simulation")
    def simulation() -> dict[str, Any]:
        # simulation.json is keyed by preset at the top level, so it is wrapped rather
        # than merged: merging would put `available` beside the preset names.
        return _artifact("simulation.json", hint="run `make simulate` to produce it",
                         under="presets")

    def _artifact(name: str, *, hint: str, under: str | None = None) -> dict[str, Any]:
        path = ARTIFACTS / name
        if not path.exists():
            return {"available": False, "reason": f"{name} not found; {hint}"}
        try:
            body = json.loads(path.read_text())
        except Exception as exc:            # pragma: no cover - corrupt artifact
            return {"available": False, "reason": f"{name} unreadable: {exc}"}
        if under is not None:
            return {"available": True, "generated_from": name, under: body}
        body["available"] = True
        return body

    # ---- controls -------------------------------------------------------
    @app.post("/api/session/step")
    async def post_step(body: StepBody | None = None) -> dict[str, Any]:
        intervals = (body or StepBody()).intervals
        _, frame = await run_mutation(session().step, intervals)
        return frame

    @app.post("/api/session/run")
    async def post_run(body: RunBody | None = None) -> dict[str, Any]:
        _, frame = await run_mutation(
            session().set_running, True, speed_x=(body or RunBody()).speed_x)
        return frame

    @app.post("/api/session/pause")
    async def post_pause() -> dict[str, Any]:
        _, frame = await run_mutation(session().set_running, False)
        return frame

    @app.post("/api/session/reset")
    async def post_reset(body: ResetBody | None = None) -> dict[str, Any]:
        b = body or ResetBody()
        if b.preset is not None and b.preset not in PRESETS:
            raise HTTPException(422, f"unknown preset {b.preset!r}")
        async with lock:
            fresh = await asyncio.to_thread(
                build_session, preset=b.preset, seed=b.seed, open_at=b.open_at,
                warmup_intervals=b.warmup_intervals)
            state["session"] = fresh
            frame = await asyncio.to_thread(lambda: fresh.frame)
        hub.publish({"type": "frame", "frame": frame})
        return frame

    @app.post("/api/session/inject")
    async def post_inject(body: InjectBody) -> dict[str, Any]:
        try:
            kind = EventType(body.type)
        except ValueError:
            raise HTTPException(
                422, f"unknown situation {body.type!r}; options are "
                     f"{sorted(e.value for e in EventType)}")
        _, frame = await run_mutation(session().inject, kind, **body.spec())
        return frame

    @app.post("/api/session/bulletin")
    async def post_bulletin(body: BulletinBody) -> dict[str, Any]:
        result, frame = await run_mutation(session().submit_bulletin, body.text)
        return {**result, "frame": frame}

    @app.post("/api/session/bulletin/confirm")
    async def post_bulletin_confirm(body: ConfirmBody) -> dict[str, Any]:
        try:
            result, frame = await run_mutation(session().confirm_bulletin, body.accept)
        except LookupError as exc:
            raise HTTPException(409, str(exc))
        return {**result, "frame": frame}

    @app.post("/api/session/weights")
    async def post_weights(body: WeightsBody) -> dict[str, Any]:
        unknown = set(body.weights or {}) - set(cfg.objectives.base_weights)
        if unknown:
            raise HTTPException(422, f"unknown objective(s): {sorted(unknown)}")
        if not body.weights and body.risk_lambda is None:
            raise HTTPException(422, "nothing to change")
        try:
            result, frame = await run_mutation(
                session().set_weights, body.weights,
                risk_lambda=body.risk_lambda, note=body.note)
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc))
        return {**result, "frame": frame}

    @app.post("/api/session/ask")
    async def post_ask(body: AskBody) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(session().ask, body.question)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except LookupError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/session/approvals/{proposal_id}")
    async def post_approval(proposal_id: str, body: ApprovalBody) -> dict[str, Any]:
        try:
            result, frame = await run_mutation(
                session().decide_approval, proposal_id, body.approve, by=body.by)
        except LookupError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        return {**result, "frame": frame}

    @app.post("/api/session/override")
    async def post_override(body: OverrideBody) -> dict[str, Any]:
        result, frame = await run_mutation(
            session().override, body.note, by=body.by)
        return {**result, "frame": frame}

    # ---- F3: scenario simulation ---------------------------------------
    @app.post("/api/simulate")
    async def post_simulate(body: SimulateBody) -> dict[str, Any]:
        if body.preset not in PRESETS:
            raise HTTPException(422, f"unknown preset {body.preset!r}")
        job_id = f"sim-{uuid.uuid4().hex[:10]}"
        job: dict[str, Any] = {
            "id": job_id, "status": "running", "preset": body.preset,
            "paths": body.paths, "intervals": body.intervals,
            "done": 0, "total": body.paths * len(body.strategies or STRATEGIES),
            "result": None, "error": None,
        }
        state["jobs"][job_id] = job

        def progress(done: int, total: int, strategy: str, path: int) -> None:
            job["done"], job["total"] = done, total
            job["stage"] = f"{strategy} path {path + 1}"

        def work() -> None:
            try:
                mc = MonteCarlo(cfg, forecaster)
                result = mc.run(
                    preset=body.preset, paths=body.paths, intervals=body.intervals,
                    strategies=tuple(body.strategies or STRATEGIES),
                    progress=progress)
                job["result"] = jsonable(result.as_dict())
                job["result"]["strategy_mix_by_hour"] = jsonable(
                    result.by_strategy["agent"].strategy_mix_by_hour(cfg)
                    if "agent" in result.by_strategy else [])
                job["status"] = "done"
            except Exception as exc:
                job["status"], job["error"] = "error", str(exc)

        asyncio.create_task(asyncio.to_thread(work))
        return job

    @app.get("/api/simulate/{job_id}")
    def get_simulate(job_id: str) -> dict[str, Any]:
        job = state["jobs"].get(job_id)
        if job is None:
            raise HTTPException(404, f"no simulation job {job_id!r}")
        return job

    # ---- the tick stream -------------------------------------------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        queue = hub.subscribe()
        # The console must never render an empty screen, so the current frame goes
        # out before anything else.
        s = state["session"]
        if s is not None:
            await ws.send_json({"type": "frame",
                                "frame": await asyncio.to_thread(lambda: s.frame)})
        try:
            while True:
                message = await queue.get()
                await ws.send_json(message)
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            hub.unsubscribe(queue)

    return app


app = create_app()
