import type { Frame } from "../api/types";
import { mw, objective, OBJECTIVE_LABEL, usd } from "../lib/units";
import { ParetoScatter } from "./charts/ParetoScatter";
import { Chip, Empty, Panel } from "./ui";

/** PRD: "Decision trace, in loop order: situation (regimes and why) → criteria (base vs
 *  final weights, risk) → options evaluated (80 clusters × scenarios, guardrail
 *  removals, top 5 with P10–P90 cost) → Pareto scatter → committed actions → why this
 *  over the runner-up." The numbering is the loop order, made visible. */
export function DecisionTrace({ frame }: { frame: Frame }) {
  const { trace } = frame;
  const r = trace.committed.interval;

  return (
    <Panel
      title="How this interval was decided"
      hint={`${trace.options.clusters_evaluated} clusters × ${trace.options.scenarios} scenarios × ${trace.options.steps} steps`}
    >
      <div className="loop">
        {/* 1. situation */}
        <div className="loop-step">
          <span className="idx">1</span>
          <div>
            <h4>Situation</h4>
            {trace.situation.regimes.length === 0 ? (
              <p className="small secondary" style={{ margin: "2px 0 0" }}>
                Normal conditions: no regime triggered, so the operator's base weights
                stand unchanged.
              </p>
            ) : (
              <div className="rows" style={{ marginTop: 4 }}>
                {trace.situation.regimes.map((regime) => (
                  <div className="row" key={regime.key} style={{ justifyContent: "flex-start" }}>
                    <Chip tone="warning">{regime.label}</Chip>
                    <span className="small secondary">{regime.trigger}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* 2. criteria */}
        <div className="loop-step">
          <span className="idx">2</span>
          <div>
            <h4>Criteria</h4>
            <p className="tiny muted" style={{ margin: "2px 0 6px" }}>
              The operator sets the base weights; the agent only shifts them by the
              multipliers the active regimes carry.
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Objective</th>
                    <th className="right">Base</th>
                    <th className="right">Final</th>
                    <th className="right">Shift</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(trace.criteria.base_weights).map((key) => {
                    const base = trace.criteria.base_weights[key] ?? 0;
                    const final = trace.criteria.final_weights[key] ?? 0;
                    const moved = Math.abs(final - base) > 1e-9;
                    return (
                      <tr key={key}>
                        <td>{OBJECTIVE_LABEL[key] ?? key}</td>
                        <td className="right">{base.toFixed(3)}</td>
                        <td className="right">
                          <strong>{final.toFixed(3)}</strong>
                        </td>
                        <td className="right">
                          {moved ? (
                            <span
                              style={{
                                color:
                                  final > base ? "var(--success-text)" : "var(--text-muted)",
                              }}
                            >
                              {final > base ? "▲" : "▼"} ×
                              {(trace.criteria.multiplier_applied[key] ?? 1).toFixed(2)}
                            </span>
                          ) : (
                            <span className="muted">—</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                  <tr>
                    <td>Risk aversion λ</td>
                    <td className="right">{trace.criteria.base_risk_lambda.toFixed(2)}</td>
                    <td className="right">
                      <strong>{trace.criteria.risk_lambda.toFixed(2)}</strong>
                    </td>
                    <td className="right muted">CVaR₉₀ weight</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* 3. options and guardrails */}
        <div className="loop-step">
          <span className="idx">3</span>
          <div>
            <h4>Options evaluated</h4>
            <p className="small secondary" style={{ margin: "2px 0 6px" }}>
              {trace.options.text}
            </p>
            <div className="rows">
              {trace.options.guardrails.map((g) => (
                <div className="row" key={g.key} style={{ alignItems: "flex-start" }}>
                  <span style={{ flex: "1 1 auto", minWidth: 0 }}>
                    <span className="small">{g.rule}</span>
                    <br />
                    <span className="tiny muted">{g.note}</span>
                  </span>
                  <Chip
                    tone={!g.applied ? "neutral" : g.removed_count > 0 ? "critical" : "good"}
                  >
                    {!g.applied
                      ? "not applicable"
                      : g.removed_count > 0
                        ? `removed ${g.removed_count}`
                        : "removed none"}
                  </Chip>
                </div>
              ))}
            </div>

            <h4 style={{ margin: "12px 0 4px" }}>Top 5 surviving clusters</h4>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Cluster</th>
                    <th className="right">Score</th>
                    <th className="right">Cost P10–P90</th>
                    <th className="right">Unserved</th>
                  </tr>
                </thead>
                <tbody>
                  {trace.options.top5.map((row) => (
                    <tr
                      key={row.index}
                      className={row.index === trace.committed.index ? "chosen" : undefined}
                    >
                      <td className="num">{row.index}</td>
                      <td>
                        {row.cluster.label}
                        {row.index === trace.committed.index && (
                          <>
                            {" "}
                            <Chip tone="good">committed</Chip>
                          </>
                        )}
                      </td>
                      <td className="right">{row.score.toFixed(4)}</td>
                      <td className="right">
                        {usd(row.cost_p10)} – {usd(row.cost_p90)}
                      </td>
                      <td className="right">
                        {row.unserved_mwh < 1e-4 ? "0 MWh" : `${row.unserved_mwh.toFixed(3)} MWh`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* 4. pareto */}
        <div className="loop-step">
          <span className="idx">4</span>
          <div>
            <h4>Pareto front</h4>
            <p className="tiny muted" style={{ margin: "2px 0 6px" }}>
              {trace.options.pareto_front.length} of {trace.options.clusters_evaluated}{" "}
              clusters are non-dominated. The committed cluster is always on the front — if
              it were not, something strictly better would exist.
            </p>
            <ParetoScatter
              points={trace.options.scatter}
              chosenIndex={trace.committed.index}
            />
          </div>
        </div>

        {/* 5. committed */}
        <div className="loop-step">
          <span className="idx">5</span>
          <div>
            <h4>Committed</h4>
            <p className="row" style={{ margin: "2px 0 6px", justifyContent: "flex-start" }}>
              <strong>{trace.committed.cluster.label}</strong>
              {trace.committed.held_by_hysteresis && (
                <Chip tone="neutral" title="Within the hysteresis margin of the winner">
                  held from the previous interval
                </Chip>
              )}
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Objective</th>
                    <th className="right">This plan</th>
                    <th className="right">Weighted contribution</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(trace.committed.objectives).map(([key, value]) => (
                    <tr key={key}>
                      <td>{OBJECTIVE_LABEL[key] ?? key}</td>
                      <td className="right">{objective(key, value)}</td>
                      <td className="right muted">
                        {(trace.committed.contributions[key] ?? 0).toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <dl className="kv" style={{ marginTop: 8 }}>
              <dt>Cost outlook</dt>
              <dd>
                P10 {usd(trace.committed.cost_p10_usd)} · P50{" "}
                {usd(trace.committed.cost_p50_usd)} · P90 {usd(trace.committed.cost_p90_usd)}
              </dd>
              <dt>This interval</dt>
              <dd>
                import {mw(r.import_mw)} · export {mw(r.export_mw)} · curtailed{" "}
                {mw(r.curtailed_mw)} · DR {mw(r.dr_mw)} · shift {mw(r.shift_mw)} · unserved{" "}
                {mw(r.unserved_mw)}
              </dd>
              {r.forced_injection_mw > 0.05 && (
                <>
                  <dt>Frequency reflex</dt>
                  <dd>forcing {mw(r.forced_injection_mw)} of injection</dd>
                </>
              )}
            </dl>
          </div>
        </div>

        {/* 6. trade-off */}
        <div className="loop-step">
          <span className="idx">6</span>
          <div>
            <h4>Why this, and not the alternative</h4>
            <p className="small" style={{ margin: "2px 0 6px" }}>
              {trace.tradeoff.text}
            </p>
            {trace.tradeoff.alternative ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Objective</th>
                      <th className="right">Chosen</th>
                      <th className="right">
                        {trace.tradeoff.alternative.label}
                        {!trace.tradeoff.alternative.survived_guardrails && " (removed)"}
                      </th>
                      <th className="right">Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(trace.tradeoff.objective_deltas).map(([key, delta]) => (
                      <tr
                        key={key}
                        className={
                          key === trace.tradeoff.deciding_objective ? "chosen" : undefined
                        }
                      >
                        <td>
                          {OBJECTIVE_LABEL[key] ?? key}
                          {key === trace.tradeoff.deciding_objective && (
                            <>
                              {" "}
                              <Chip tone="good">decides</Chip>
                            </>
                          )}
                        </td>
                        <td className="right">
                          {objective(key, trace.committed.objectives[key] ?? 0)}
                        </td>
                        <td className="right">
                          {objective(
                            key,
                            trace.tradeoff.alternative?.objectives[key] ?? 0,
                          )}
                        </td>
                        <td className="right">
                          <span
                            style={{
                              color: delta > 0 ? "var(--critical)" : "var(--success-text)",
                            }}
                          >
                            {delta > 0 ? "+" : "−"}
                            {objective(key, Math.abs(delta)).replace(/^[−+]/, "")}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty>Only one cluster survived the guardrails, so there was no trade-off.</Empty>
            )}
          </div>
        </div>

        {/* 7. explanation */}
        <div className="loop-step">
          <span className="idx">7</span>
          <div>
            <h4>
              Operator explanation{" "}
              <Chip tone={trace.explanation_source === "llm" ? "good" : "neutral"}>
                {trace.explanation_source === "llm"
                  ? "written by the orchestrator"
                  : "template"}
              </Chip>
            </h4>
            <pre className="pre small secondary" style={{ marginTop: 4 }}>
              {trace.explanation}
            </pre>
            {frame.status.verification && (
              <p className="tiny muted" style={{ marginTop: 6 }}>
                Number verification:{" "}
                {frame.status.verification.ok
                  ? `every figure matched the decision trace (${frame.status.verification.checked} checked)`
                  : `failed — ${frame.status.verification.reason}; the template was shown instead`}
              </p>
            )}
          </div>
        </div>
      </div>
    </Panel>
  );
}
