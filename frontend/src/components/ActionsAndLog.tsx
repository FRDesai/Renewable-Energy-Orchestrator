import type { Frame, LogEntry, ResponseToChange } from "../api/types";
import { ms, mw } from "../lib/units";
import { Chip, Empty, Panel, severityTone, type Tone } from "./ui";

const VERDICT: Record<string, { tone: Tone; label: string }> = {
  executed: { tone: "good", label: "Executed" },
  queued: { tone: "warning", label: "Awaiting approval" },
  rejected: { tone: "critical", label: "Rejected" },
};

/** PRD F1.3: "Action list shows verb, asset, MW and reason for every set-point." */
export function CommittedActions({ frame }: { frame: Frame }) {
  if (frame.actions.length === 0) return <Empty>No set-points this interval.</Empty>;
  return (
    <Panel
      title="Committed actions"
      hint={`${frame.actions.filter((a) => a.verdict === "executed").length} executed`}
      flush
    >
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Instruction</th>
              <th>Reason</th>
              <th>State</th>
            </tr>
          </thead>
          <tbody>
            {frame.actions.map((a, i) => {
              const verdict = VERDICT[a.verdict] ?? VERDICT["executed"]!;
              return (
                <tr key={`${a.verb}-${a.asset_id}-${i}`}>
                  <td style={{ whiteSpace: "nowrap" }}>
                    <strong>
                      {a.verb.charAt(0).toUpperCase() + a.verb.slice(1)} {a.asset_name}
                    </strong>
                    {a.verb !== "hold" && <> {mw(a.mw)}</>}
                  </td>
                  <td className="small secondary">
                    {a.reason}
                    {a.verdict !== "executed" && (
                      <div className="tiny muted">{a.gate_reason}</div>
                    )}
                  </td>
                  <td>
                    <Chip tone={verdict.tone} title={`signed ${a.signature}`}>
                      {verdict.label}
                    </Chip>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="tiny muted" style={{ padding: "8px 12px", margin: 0 }}>
        Every instruction is signed over its exact contents, so re-wording a reason
        cannot change a signed command.
      </p>
    </Panel>
  );
}

/** PRD: ""Response to change" banner after an injected event: plan before → plan after,
 *  with the changed actions." */
export function ResponseBanner({ change }: { change: ResponseToChange }) {
  return (
    <div className="banner info">
      <div className="grow">
        <div className="row" style={{ justifyContent: "flex-start" }}>
          <strong>Re-planned: {change.trigger}</strong>
          <Chip tone="good">new plan in {ms(change.evaluation_ms)}</Chip>
        </div>
        <div className="small secondary" style={{ marginTop: 4 }}>
          {change.before.cluster ?? "—"} → <strong>{change.after.cluster ?? "—"}</strong>
          {change.after.regimes.length > 0 && ` · now ${change.after.regimes.join("; ")}`}
        </div>
        {change.changed_actions.length === 0 ? (
          <div className="tiny muted" style={{ marginTop: 4 }}>
            The event did not change any set-point.
          </div>
        ) : (
          <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
            {change.changed_actions.map((a, i) => (
              <li key={i}>
                {a.verb} {a.asset_name}:{" "}
                <span className="num">
                  {a.before_mw === null ? "—" : mw(a.before_mw)} →{" "}
                  <strong>{a.after_mw === null ? "withdrawn" : mw(a.after_mw)}</strong>
                </span>
                <span className="muted"> · {a.reason}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

/** PRD: "Action log, newest first, with every decision and event." */
export function ActionLog({
  entries,
  onReplay,
}: {
  entries: LogEntry[];
  onReplay?: (interval: number) => void;
}) {
  return (
    <Panel title="Action log" hint="newest first" flush scroll>
      {entries.length === 0 ? (
        <Empty>Nothing logged yet.</Empty>
      ) : (
        entries.map((entry, i) => (
          <div className={`log-entry ${entry.severity}`} key={`${entry.interval}-${i}`}>
            <span className="when">{entry.time}</span>
            <span className="what">
              {entry.severity !== "info" && (
                <Chip tone={severityTone(entry.severity)}>{entry.kind.replace(/_/g, " ")}</Chip>
              )}{" "}
              {entry.message}
              {onReplay && entry.kind === "decision" && (
                <button
                  className="btn small"
                  style={{ marginLeft: 6, padding: "0 6px" }}
                  onClick={() => onReplay(entry.interval)}
                  title="Replay this interval from the audit log"
                >
                  replay
                </button>
              )}
            </span>
          </div>
        ))
      )}
    </Panel>
  );
}

export function Alarms({ frame }: { frame: Frame }) {
  if (frame.alarms.length === 0) return null;
  const order = { critical: 0, warning: 1, info: 2 } as const;
  const sorted = [...frame.alarms].sort((a, b) => order[a.severity] - order[b.severity]);
  return (
    <div className="stack" style={{ gap: 6 }}>
      {sorted.map((alarm, i) => (
        <div className={`banner ${alarm.severity}`} key={i}>
          <div className="grow">
            <div className="row" style={{ justifyContent: "flex-start" }}>
              <Chip tone={severityTone(alarm.severity)}>{alarm.kind.replace(/_/g, " ")}</Chip>
              <span className="small">{alarm.message}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
