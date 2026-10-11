import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ConfigResponse, Frame } from "../api/types";
import { ms, OBJECTIVE_LABEL } from "../lib/units";
import { Chip, Panel } from "../components/ui";

/** PRD F2.6: "Let operators change base weights and risk appetite and see the decision
 *  change — re-evaluation within 1 s of a slider change." The measured time comes back
 *  from the backend and is shown, because the acceptance criterion is a number. */
export function OperatorPriorities({
  frame,
  config,
  act,
  busy,
}: {
  frame: Frame;
  config: ConfigResponse | null;
  act: <T>(fn: () => Promise<T>, pick?: (r: T) => Frame | undefined) => Promise<T | null>;
  busy: boolean;
}) {
  const base = frame.trace.criteria.base_weights;
  const [draft, setDraft] = useState<Record<string, number>>(base);
  const [lambda, setLambda] = useState(frame.trace.criteria.base_risk_lambda);
  const [result, setResult] = useState<{ elapsed_ms: number; changed: boolean } | null>(null);

  // Follow the session when it is reset or changed elsewhere, but do not fight the
  // operator mid-drag.
  const stamp = Object.values(base).join(",");
  useEffect(() => {
    setDraft(base);
    setLambda(frame.trace.criteria.base_risk_lambda);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stamp]);

  const keys = Object.keys(base);
  const total = keys.reduce((sum, key) => sum + (draft[key] ?? 0), 0) || 1;
  const dirty =
    keys.some((key) => Math.abs((draft[key] ?? 0) - (base[key] ?? 0)) > 1e-4) ||
    Math.abs(lambda - frame.trace.criteria.base_risk_lambda) > 1e-4;

  const apply = async () => {
    const response = await act(
      () => api.weights({ weights: draft, risk_lambda: lambda, note: "operator slider" }),
      (r) => r.frame,
    );
    if (response) setResult({ elapsed_ms: response.elapsed_ms, changed: response.changed });
  };

  return (
    <Panel
      title="Operator priorities"
      hint="the agent shifts these; it never rewrites them"
      actions={
        <button className="btn primary small" disabled={busy || !dirty} onClick={() => void apply()}>
          Re-evaluate
        </button>
      }
    >
      <div className="weights">
        {keys.map((key) => (
          <label key={key} style={{ display: "contents" }}>
            <span className="small" style={{ display: "flex", flexDirection: "column" }}>
              {OBJECTIVE_LABEL[key] ?? key}
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={draft[key] ?? 0}
                disabled={busy}
                onChange={(event) =>
                  setDraft((d) => ({ ...d, [key]: Number(event.target.value) }))
                }
              />
            </span>
            <span className="num small" style={{ textAlign: "right", whiteSpace: "nowrap" }}>
              {((draft[key] ?? 0) / total).toFixed(3)}
              <span className="muted">
                {" "}
                → {(frame.trace.criteria.final_weights[key] ?? 0).toFixed(3)}
              </span>
            </span>
          </label>
        ))}
        <label style={{ display: "contents" }}>
          <span className="small" style={{ display: "flex", flexDirection: "column" }}>
            Risk aversion λ
            <input
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={lambda}
              disabled={busy}
              onChange={(event) => setLambda(Number(event.target.value))}
            />
          </span>
          <span className="num small" style={{ textAlign: "right" }}>
            {lambda.toFixed(2)}
            <span className="muted"> → {frame.trace.criteria.risk_lambda.toFixed(2)}</span>
          </span>
        </label>
      </div>

      <p className="tiny muted" style={{ marginTop: 8 }}>
        Weights are renormalised to sum to 1. The right-hand figure is what the active
        regimes made of them.
      </p>

      {result && (
        <p className="small" style={{ marginTop: 6 }}>
          <Chip tone={result.elapsed_ms < 1000 ? "good" : "warning"}>
            re-evaluated in {ms(result.elapsed_ms)}
          </Chip>{" "}
          {result.changed
            ? "the committed cluster changed"
            : "the same cluster still wins"}
        </p>
      )}

      {(config?.change_log.length ?? 0) > 0 && (
        <details className="plain" style={{ marginTop: 8 }}>
          <summary>Configuration change log ({config?.change_log.length})</summary>
          <pre className="pre mono" style={{ maxHeight: 140, overflow: "auto" }}>
            {config?.change_log.join("\n")}
          </pre>
        </details>
      )}
    </Panel>
  );
}
