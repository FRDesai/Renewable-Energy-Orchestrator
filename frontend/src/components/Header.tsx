import { api } from "../api/client";
import type { ConfigResponse, Frame } from "../api/types";
import { Chip, type Tone } from "./ui";

export const TABS = [
  { key: "live", label: "Live operations" },
  { key: "simulation", label: "Scenario simulation" },
  { key: "reliability", label: "Reliability tests" },
  { key: "how", label: "How it decides" },
] as const;

export type TabKey = (typeof TABS)[number]["key"];

const SPEEDS = [1, 4, 15, 60, 240, 600];

const MODE_CHIP: Record<string, { tone: Tone; label: string }> = {
  agent: { tone: "good", label: "Agent in control" },
  rule_based_fallback: { tone: "critical", label: "Rule-based fallback" },
  degraded: { tone: "critical", label: "Degraded — holding" },
};

export function Header({
  frame,
  config,
  tab,
  onTab,
  connected,
  busy,
  act,
}: {
  frame: Frame | null;
  config: ConfigResponse | null;
  tab: TabKey;
  onTab: (tab: TabKey) => void;
  connected: boolean;
  busy: boolean;
  act: <T>(fn: () => Promise<T>, pick?: (r: T) => Frame | undefined) => Promise<T | null>;
}) {
  const running = frame?.clock.running ?? false;
  const mode = frame ? (MODE_CHIP[frame.status.mode] ?? MODE_CHIP["agent"]!) : null;

  return (
    <header className="topbar">
      <div className="brand">
        <strong>Renewable Energy Orchestrator</strong>
        <small>
          {frame ? `${frame.scenario.label} · seed ${frame.scenario.seed}` : "connecting…"}
        </small>
      </div>

      <div className="clock">
        <span className="time">{frame?.clock.time ?? "--:--"}</span>
        <span className="meta">
          interval {frame?.clock.interval ?? "—"}
          <br />
          {frame?.clock.interval_minutes ?? 15} min steps
        </span>
      </div>

      <div className="controls">
        <button
          className={`btn ${running ? "" : "primary"}`}
          disabled={busy || !frame}
          onClick={() =>
            void act(() => (running ? api.pause() : api.run(frame?.clock.speed_x ?? 60)))
          }
        >
          {running ? "Pause" : "Run"}
        </button>
        <button
          className="btn"
          disabled={busy || !frame || running}
          onClick={() => void act(() => api.step(1))}
          title="Advance one 15-minute interval"
        >
          Step 15 min
        </button>
        <label className="tiny muted" style={{ display: "flex", gap: 4, alignItems: "center" }}>
          Speed
          <select
            value={frame?.clock.speed_x ?? 60}
            disabled={busy || !frame}
            onChange={(event) => void act(() => api.run(Number(event.target.value)))}
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </label>
        <label className="tiny muted" style={{ display: "flex", gap: 4, alignItems: "center" }}>
          Scenario
          <select
            value={frame?.scenario.preset ?? "typical_day"}
            disabled={busy || !config}
            onChange={(event) =>
              void act(() => api.reset({ preset: event.target.value }), (f) => f)
            }
          >
            {Object.entries(config?.presets ?? {}).map(([key, meta]) => (
              <option key={key} value={key}>
                {meta.label}
              </option>
            ))}
          </select>
        </label>
        <button
          className="btn"
          disabled={busy || !frame}
          onClick={() =>
            void act(() => api.reset({ preset: frame?.scenario.preset }), (f) => f)
          }
          title="Restart this scenario at 09:00"
        >
          Reset
        </button>
      </div>

      <div className="controls">
        {mode && <Chip tone={mode.tone}>{mode.label}</Chip>}
        <Chip tone={connected ? "good" : "critical"}>
          {connected ? "Live" : "Reconnecting"}
        </Chip>
        {frame && !frame.agent.llm_available && <Chip tone="neutral">LLM off</Chip>}
      </div>

      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.key}
            className="tab"
            role="tab"
            aria-selected={tab === t.key}
            onClick={() => onTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
