import { useState } from "react";
import { api } from "../api/client";
import type { ConfigResponse, Frame, LogEntry, ReplayResponse } from "../api/types";
import { ActionLog, Alarms, CommittedActions, ResponseBanner } from "../components/ActionsAndLog";
import { AgentPanel } from "../components/AgentPanel";
import { SupplyDemandChart } from "../components/charts/SupplyDemandChart";
import { PriceSocChart } from "../components/charts/PriceSocChart";
import { DecisionTrace } from "../components/DecisionTrace";
import { EventInjector } from "../components/EventInjector";
import { PortfolioPanel } from "../components/PortfolioPanel";
import { OperatorPriorities } from "./OperatorPriorities";
import { Chip, Panel } from "../components/ui";

export function LiveOperations({
  frame,
  config,
  log,
  busy,
  act,
}: {
  frame: Frame;
  config: ConfigResponse | null;
  log: LogEntry[];
  busy: boolean;
  act: <T>(fn: () => Promise<T>, pick?: (r: T) => Frame | undefined) => Promise<T | null>;
}) {
  const [replay, setReplay] = useState<ReplayResponse | null>(null);
  const socMax = config?.portfolio.battery_limits["soc_max_pct"] ?? 95;
  const reserveFloor = config?.portfolio.battery_limits["reserve_floor_pct"] ?? 10;

  return (
    <div className="stack">
      <Alarms frame={frame} />
      {frame.response_to_change && <ResponseBanner change={frame.response_to_change} />}

      <div className="cols live">
        <div className="stack">
          <Panel
            title="Supply and demand"
            hint={`last ${frame.history.length} intervals, next ${frame.forecast.horizon_hours} h`}
          >
            <SupplyDemandChart
              history={frame.history}
              forecast={frame.forecast}
              intervalMinutes={frame.clock.interval_minutes}
              nowInterval={frame.clock.interval}
            />
          </Panel>

          <Panel title="Price and state of charge" hint="two scales, two charts">
            <PriceSocChart
              history={frame.history}
              forecast={frame.forecast}
              intervalMinutes={frame.clock.interval_minutes}
              socMaxPct={socMax}
              reserveFloorPct={reserveFloor}
            />
          </Panel>

          <CommittedActions frame={frame} />
          <DecisionTrace frame={frame} />
        </div>

        <div className="stack">
          <PortfolioPanel frame={frame} config={config} />
          <EventInjector config={config} frame={frame} act={act} busy={busy} />
          <AgentPanel frame={frame} act={act} busy={busy} />
          <OperatorPriorities frame={frame} config={config} act={act} busy={busy} />
          <ActionLog
            entries={log}
            onReplay={(interval) => {
              void (async () => {
                const result = await act(() => api.replay(interval));
                if (result) setReplay(result);
              })();
            }}
          />
          {replay && (
            <Panel
              title={`Replay of interval ${replay.replay_key.interval}`}
              hint="from the append-only audit log"
              actions={
                <button className="btn small" onClick={() => setReplay(null)}>
                  Close
                </button>
              }
            >
              <dl className="kv">
                <dt>Decision seed</dt>
                <dd>{replay.replay_key.seed ?? "—"}</dd>
                <dt>Preset / world seed</dt>
                <dd>
                  {replay.replay_key.preset} / {replay.replay_key.world_seed}
                </dd>
                <dt>Cluster committed</dt>
                <dd>#{replay.replay_key.chosen_index}</dd>
                <dt>Config version</dt>
                <dd>{replay.replay_key.config_version}</dd>
              </dl>
              {replay.superseded > 0 && (
                <p className="tiny muted" style={{ marginTop: 6 }}>
                  <Chip tone="neutral">{replay.superseded} superseded plan(s)</Chip> kept in
                  the log for this interval — a re-plan never overwrites what it replaced.
                </p>
              )}
              <details className="plain" style={{ marginTop: 8 }}>
                <summary>Stored decision record</summary>
                <pre className="pre mono" style={{ maxHeight: 220, overflow: "auto" }}>
                  {JSON.stringify(replay.decision, null, 1)}
                </pre>
              </details>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
