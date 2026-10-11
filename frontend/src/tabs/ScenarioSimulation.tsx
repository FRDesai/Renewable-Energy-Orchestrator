import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { ConfigResponse, SimulationJob, SimulationResult } from "../api/types";
import { FanChart, STRATEGY_COLOR, STRATEGY_LABEL } from "../components/charts/FanChart";
import { StrategyMixBars } from "../components/charts/Bars";
import { deltaPct, mwh, pct, tonnes, usd } from "../lib/units";
import { Chip, Empty, Panel, Stat } from "../components/ui";

const METRICS = [
  { key: "energy_cost_usd", label: "Energy cost", format: usd },
  { key: "net_cost_usd", label: "Net cost (incl. storage)", format: usd },
  { key: "carbon_tonnes", label: "Carbon", format: tonnes },
  { key: "unserved_mwh", label: "Load shed", format: (v: number) => mwh(v, 2) },
  { key: "curtailed_mwh", label: "Curtailed", format: (v: number) => mwh(v, 1) },
  { key: "throughput_mwh", label: "Battery throughput", format: (v: number) => mwh(v, 0) },
  { key: "wear_usd", label: "Battery wear", format: usd },
] as const;

function band(row: Record<string, unknown>, key: string) {
  const value = row[key];
  if (value && typeof value === "object" && "mean" in value) {
    return value as { mean: number; p10?: number; p90?: number };
  }
  return null;
}

/** PRD F3: "preset, paths, run with progress, KPI comparison table, cumulative-cost and
 *  SoC fan charts, agent strategy mix by hour." The stored artifact from `make simulate`
 *  is shown immediately, so the tab is never empty; a fresh run replaces it. */
export function ScenarioSimulation({ config }: { config: ConfigResponse | null }) {
  const [preset, setPreset] = useState("typical_day");
  const [paths, setPaths] = useState(6);
  const [job, setJob] = useState<SimulationJob | null>(null);
  const [stored, setStored] = useState<Record<string, SimulationResult> | null>(null);
  const [storedReason, setStoredReason] = useState<string | null>(null);
  const poll = useRef<number | undefined>(undefined);

  useEffect(() => {
    void (async () => {
      const artifact = await api.simulation();
      if (artifact.available && artifact.presets) setStored(artifact.presets);
      else setStoredReason(artifact.reason ?? "not available");
    })();
  }, []);

  useEffect(
    () => () => {
      if (poll.current !== undefined) window.clearInterval(poll.current);
    },
    [],
  );

  const start = async () => {
    const started = await api.simulate({ preset, paths, intervals: 96 });
    setJob(started);
    if (poll.current !== undefined) window.clearInterval(poll.current);
    poll.current = window.setInterval(() => {
      void (async () => {
        const status = await api.simulateStatus(started.id);
        setJob(status);
        if (status.status !== "running" && poll.current !== undefined) {
          window.clearInterval(poll.current);
          poll.current = undefined;
        }
      })();
    }, 700);
  };

  const result: SimulationResult | null =
    job?.status === "done" && job.result ? job.result : (stored?.[preset] ?? null);
  const live = job?.status === "running";
  const fromStore = !(job?.status === "done" && job.result);

  return (
    <div className="stack">
      <Panel title="Run a scenario" hint="the same engine the console runs, in batch mode">
        <div className="row" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
          <label className="small">
            Preset{" "}
            <select value={preset} onChange={(e) => setPreset(e.target.value)} disabled={live}>
              {Object.entries(config?.presets ?? {}).map(([key, meta]) => (
                <option key={key} value={key}>
                  {meta.label}
                </option>
              ))}
            </select>
          </label>
          <label className="small">
            Paths{" "}
            <select
              value={paths}
              onChange={(e) => setPaths(Number(e.target.value))}
              disabled={live}
            >
              {(config?.path_choices ?? [6, 12, 24]).map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <button className="btn primary" disabled={live} onClick={() => void start()}>
            {live ? "Running…" : "Run 24 h Monte Carlo"}
          </button>
          <span className="tiny muted">
            {config?.presets[preset]?.description ?? ""}
          </span>
        </div>

        {job && (
          <div style={{ marginTop: 10 }}>
            <div className="progress">
              <span style={{ width: `${(100 * job.done) / Math.max(1, job.total)}%` }} />
            </div>
            <p className="tiny muted" style={{ marginTop: 4 }}>
              {job.status === "running"
                ? `${job.done} of ${job.total} episodes · ${job.stage ?? ""}`
                : job.status === "error"
                  ? `failed: ${job.error}`
                  : `done — ${job.paths} paths × 96 intervals × 3 strategies`}
            </p>
          </div>
        )}
        {!job && storedReason && (
          <p className="tiny muted" style={{ marginTop: 8 }}>
            No stored benchmark: {storedReason}
          </p>
        )}
      </Panel>

      {!result ? (
        <Panel title="Results">
          <Empty>Run a scenario, or run `make simulate` to populate the stored benchmark.</Empty>
        </Panel>
      ) : (
        <>
          <Panel
            title="KPI comparison"
            hint={
              fromStore
                ? `stored benchmark · ${result.paths} paths`
                : `this run · ${result.paths} paths`
            }
          >
            <div className="statrow" style={{ marginBottom: 12 }}>
              {Object.entries(result.comparison).map(([other, c]) => (
                <Stat
                  key={other}
                  label={`vs ${STRATEGY_LABEL[other] ?? other}`}
                  value={deltaPct(c.cost_delta_pct)}
                  sub={`cost · shed ${deltaPct(c.shed_delta_pct)} · CO₂ ${deltaPct(
                    c.co2_delta_pct,
                  )} · wins ${pct(c.win_rate_cost * 100)}`}
                  tone={c.cost_delta_pct <= 0 ? "good" : "warning"}
                />
              ))}
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    {result.kpis.map((row) => (
                      <th className="right" key={row.strategy}>
                        <span style={{ color: STRATEGY_COLOR[row.strategy] }}>●</span>{" "}
                        {STRATEGY_LABEL[row.strategy] ?? row.strategy}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {METRICS.map((metric) => (
                    <tr key={metric.key}>
                      <td>{metric.label}</td>
                      {result.kpis.map((row) => {
                        const b = band(row as Record<string, unknown>, metric.key);
                        return (
                          <td className="right" key={row.strategy}>
                            {b ? (
                              <>
                                {metric.format(b.mean)}
                                {b.p10 !== undefined && b.p90 !== undefined && (
                                  <div className="tiny muted">
                                    {metric.format(b.p10)} – {metric.format(b.p90)}
                                  </div>
                                )}
                              </>
                            ) : (
                              "—"
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                  <tr>
                    <td>Renewable utilisation</td>
                    {result.kpis.map((row) => {
                      const b = band(row as Record<string, unknown>, "renewable_utilisation");
                      return (
                        <td className="right" key={row.strategy}>
                          {b ? pct(b.mean * 100, 1) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                  <tr>
                    <td>Reserve compliance</td>
                    {result.kpis.map((row) => {
                      const b = band(row as Record<string, unknown>, "reserve_compliance");
                      return (
                        <td className="right" key={row.strategy}>
                          {b ? pct(b.mean * 100, 1) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="tiny muted" style={{ marginTop: 8 }}>
              Mean with the P10–P90 range across paths. All three strategies run on common
              random numbers — the same weather and the same event timing per path — so the
              difference is attributable to the strategy alone.
            </p>
          </Panel>

          <div className="cols two">
            <Panel title="Cumulative cost" hint="P10–P90 across paths">
              <FanChart
                series={Object.entries(result.fans).map(([strategy, f]) => ({
                  strategy,
                  fan: f.cumulative_cost,
                }))}
                unitLabel="$"
                tickFormat={(v) => (Math.abs(v) >= 1000 ? `${Math.round(v / 1000)}k` : String(Math.round(v)))}
                intervalMinutes={config?.portfolio.interval_minutes ?? 15}
              />
            </Panel>
            <Panel title="Fleet state of charge" hint="P10–P90 across paths">
              <FanChart
                series={Object.entries(result.fans).map(([strategy, f]) => ({
                  strategy,
                  fan: f.fleet_soc_pct,
                }))}
                unitLabel="%"
                intervalMinutes={config?.portfolio.interval_minutes ?? 15}
              />
            </Panel>
          </div>

          <Panel title="Agent strategy mix by hour" hint="share of intervals per battery strategy">
            {result.strategy_mix_by_hour && result.strategy_mix_by_hour.length > 0 ? (
              <StrategyMixBars rows={result.strategy_mix_by_hour} />
            ) : (
              <Empty>
                Strategy mix is recorded for runs started here; the stored benchmark keeps
                KPIs and fans only.
              </Empty>
            )}
          </Panel>

          <Panel title="Replay a path in Live operations" hint="F3.5">
            <p className="small secondary" style={{ margin: 0 }}>
              Each path is a preset and a seed, and Live operations opens any preset and
              seed — so a path can be stepped through interval by interval on the same
              engine that produced it. Seeds in this run:{" "}
              <span className="mono">{result.seeds.join(", ")}</span>.
            </p>
            <div className="legend" style={{ marginTop: 8 }}>
              <Chip tone="neutral">preset {result.preset}</Chip>
              <Chip tone="neutral">{result.intervals} intervals</Chip>
              <Chip tone="neutral">{result.paths} paths</Chip>
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}
