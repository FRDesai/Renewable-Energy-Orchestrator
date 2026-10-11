import { useState } from "react";
import { api } from "../api/client";
import type { AskAnswer, Frame } from "../api/types";
import { mw, ms } from "../lib/units";
import { Chip, Empty, Panel } from "./ui";

const SUGGESTIONS = [
  "What did you give up for this?",
  "Why are you holding the batteries where they are?",
  "What was curtailed, and why?",
  "Which regime is driving the weights?",
];

/** PRD "Agent panel": tool-call trace, Ask the agent, approval queue, data-dropout
 *  injection. */
export function AgentPanel({
  frame,
  act,
  busy,
}: {
  frame: Frame;
  act: <T>(fn: () => Promise<T>, pick?: (r: T) => Frame | undefined) => Promise<T | null>;
  busy: boolean;
}) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<AskAnswer | null>(null);
  const [overrideNote, setOverrideNote] = useState("");

  const ask = async (text: string) => {
    if (!text.trim()) return;
    setQuestion(text);
    const result = await act(() => api.ask(text));
    if (result) setAnswer(result);
  };

  const pending = frame.approvals.filter((p) => p.status === "pending");

  return (
    <div className="stack">
      <Panel
        title="Agent"
        hint={
          frame.agent.llm_available
            ? `${frame.agent.model} · ${frame.agent.iterations} iteration(s)`
            : "optimizer running the loop alone"
        }
      >
        {!frame.agent.llm_available && (
          <div className="banner info" style={{ marginBottom: 10 }}>
            <div className="grow">
              <strong>Orchestrator narration unavailable</strong>
              <div className="small secondary">
                {frame.agent.unavailable_reason}. The optimizer is running the loop on
                schedule and explanations are template-generated — the PRD's required
                degraded mode, not an outage.
              </div>
            </div>
          </div>
        )}

        <h4 style={{ marginBottom: 6 }}>Tool calls this interval</h4>
        {frame.agent.tool_calls.length === 0 ? (
          <Empty>No tools were called.</Empty>
        ) : (
          <ol className="rows" style={{ margin: 0, paddingLeft: 18 }}>
            {frame.agent.tool_calls.map((call, i) => (
              <li key={i} style={{ listStyle: "decimal" }}>
                <div className="row" style={{ justifyContent: "flex-start" }}>
                  <code className="mono">{call.name}</code>
                  {call.side_effect && <Chip tone="warning">side effect</Chip>}
                  {!call.ok && <Chip tone="critical">error</Chip>}
                </div>
                <div className="tiny muted">
                  {call.error ?? call.result}
                  {Object.keys(call.arguments).length > 0 && (
                    <>
                      {" "}
                      <span className="mono">{JSON.stringify(call.arguments)}</span>
                    </>
                  )}
                </div>
              </li>
            ))}
          </ol>
        )}
        <p className="tiny muted" style={{ marginTop: 8 }}>
          No tool accepts a power figure, state of charge or set-point from the model.
          Every MW below came from the optimizer.
        </p>
      </Panel>

      <Panel title="Ask the agent" hint="answered from this interval's decision trace">
        <div className="row" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
          {SUGGESTIONS.map((s) => (
            <button key={s} className="btn small" disabled={busy} onClick={() => void ask(s)}>
              {s}
            </button>
          ))}
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <input
            type="text"
            style={{ flex: 1 }}
            value={question}
            placeholder="Ask about this decision…"
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void ask(question);
            }}
          />
          <button className="btn" disabled={busy} onClick={() => void ask(question)}>
            Ask
          </button>
        </div>
        {answer && (
          <div className="banner info" style={{ marginTop: 10 }}>
            <div className="grow">
              <pre className="pre small">{answer.text}</pre>
              <div className="tiny muted" style={{ marginTop: 6 }}>
                {answer.source === "llm" ? "Written by the orchestrator" : "Template answer"}
                {answer.verification &&
                  (answer.verification.ok
                    ? " · every figure checked against the trace"
                    : ` · failed verification (${answer.verification.reason}), template shown`)}
                {answer.citations.length > 0 && ` · drawn from ${answer.citations.join(", ")}`}
              </div>
            </div>
          </div>
        )}
      </Panel>

      <Panel
        title="Approval queue"
        hint={pending.length === 0 ? "nothing waiting" : `${pending.length} waiting`}
      >
        {pending.length === 0 ? (
          <Empty>
            Routine set-points execute automatically. High-impact proposals — load
            reduction beyond contracted DR, maintenance moves, crew dispatch — wait here.
          </Empty>
        ) : (
          <div className="rows">
            {pending.map((p) => (
              <div
                className="banner warning"
                key={p.id}
                style={{ alignItems: "center", flexWrap: "wrap" }}
              >
                <div className="grow">
                  <strong>
                    {p.kind === "set_point"
                      ? `${p.verb} ${p.asset_name} ${mw(p.mw ?? 0)}`
                      : p.summary}
                  </strong>
                  <div className="small secondary">
                    {p.kind === "set_point" ? p.reason : ""}
                    {p.kind === "maintenance" &&
                      p.saving_usd !== undefined &&
                      ` · interval ${p.current_start} → ${p.proposed_start}`}
                  </div>
                  <div className="tiny muted">
                    {p.kind.replace(/_/g, " ")} · raised at interval {p.interval}
                  </div>
                </div>
                <button
                  className="btn primary small"
                  disabled={busy}
                  onClick={() => void act(() => api.decideApproval(p.id, true), (r) => r.frame)}
                >
                  Approve
                </button>
                <button
                  className="btn small"
                  disabled={busy}
                  onClick={() => void act(() => api.decideApproval(p.id, false), (r) => r.frame)}
                >
                  Reject
                </button>
              </div>
            ))}
          </div>
        )}
        <p className="tiny muted" style={{ marginTop: 8 }}>
          Rejections are logged as preference data.
        </p>
      </Panel>

      <Panel title="Operator override" hint="always wins, always logged">
        <div className="row">
          <input
            type="text"
            style={{ flex: 1 }}
            value={overrideNote}
            placeholder="e.g. holding Harbor BESS for the evening peak"
            onChange={(event) => setOverrideNote(event.target.value)}
          />
          <button
            className="btn danger"
            disabled={busy || overrideNote.trim().length === 0}
            onClick={async () => {
              await act(() => api.override(overrideNote), (r) => r.frame);
              setOverrideNote("");
            }}
          >
            Log override
          </button>
        </div>
        <div className="row" style={{ marginTop: 10, justifyContent: "flex-start" }}>
          <button
            className="btn small"
            disabled={busy}
            onClick={() =>
              void act(() => api.inject({ type: "data_dropout", channels: ["solar", "price"] }))
            }
            title="Cut the solar and price feeds to show forecast substitution"
          >
            Cut the solar + price feed
          </button>
          <span className="tiny muted">
            Shows forecast substitution and, past {frame.data_quality.limit_intervals}{" "}
            intervals, the rule-based fallback.
          </span>
        </div>
        <p className="tiny muted" style={{ marginTop: 8 }}>
          Decision evaluation took {ms(frame.status.evaluation_ms)}; the full cycle took{" "}
          {ms(frame.status.cycle_ms)}.
        </p>
      </Panel>
    </div>
  );
}
