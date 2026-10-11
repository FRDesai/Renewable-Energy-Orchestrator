import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ReliabilityArtifact } from "../api/types";
import { Histogram } from "../components/charts/Bars";
import { deltaPct, mwh, pct, tonnes, usd } from "../lib/units";
import { Chip, Empty, Panel, Stat } from "../components/ui";

/** PRD tab 3: "each test with pass/fail counts and failing examples, win rate vs
 *  rule-based, latency distribution." Reads the artifact `make episodes` writes, so the
 *  tab shows a measured run rather than a claim. */
export function ReliabilityTests() {
  const [data, setData] = useState<ReliabilityArtifact | null>(null);

  useEffect(() => {
    void (async () => setData(await api.reliability()))();
  }, []);

  if (!data) return <Panel title="Reliability tests"><Empty>Loading…</Empty></Panel>;
  if (!data.available) {
    return (
      <Panel title="Reliability tests">
        <Empty>
          No measured run available: {data.reason}. Run <code>make episodes</code> to
          produce it.
        </Empty>
      </Panel>
    );
  }

  const invariants = data.invariants ?? [];
  const bench = data.benchmark ?? {};
  const rows = data.rows ?? [];
  const latencies = rows
    .map((r) => r.agent["latency_p95_ms"])
    .filter((v): v is number => typeof v === "number");
  const failing = invariants.filter((i) => i.failed > 0);
  const notApplicable = invariants.filter((i) => i.applicable === 0);

  const winRate = (bench["win_rate_vs_rule_based"] ?? 0) * 100;
  const netWinRate = (bench["win_rate_net_cost"] ?? 0) * 100;
  const costDelta =
    bench["agent_cost_usd"] && bench["rule_based_cost_usd"]
      ? (100 * (bench["agent_cost_usd"] - bench["rule_based_cost_usd"])) /
        bench["rule_based_cost_usd"]
      : 0;
  const co2Delta =
    bench["agent_co2_t"] && bench["rule_based_co2_t"]
      ? (100 * (bench["agent_co2_t"] - bench["rule_based_co2_t"])) / bench["rule_based_co2_t"]
      : 0;

  return (
    <div className="stack">
      <Panel
        title="Invariant suite"
        hint={`${data.episodes} randomised episodes · ${data.decisions} decisions · ${data.elapsed_s}s`}
      >
        <div className="statrow" style={{ marginBottom: 12 }}>
          <Stat
            label="Invariants passing"
            value={`${invariants.length - failing.length} of ${invariants.length}`}
            tone={failing.length === 0 ? "good" : "critical"}
            sub={failing.length === 0 ? "every check green" : `${failing.length} failing`}
          />
          <Stat
            label="Load shed"
            value={mwh(bench["agent_unserved_mwh"] ?? 0, 2)}
            sub={`rule-based ${mwh(bench["rule_based_unserved_mwh"] ?? 0, 1)}`}
            tone={(bench["agent_unserved_mwh"] ?? 0) <= 1e-6 ? "good" : "warning"}
          />
          <Stat
            label="Avoidable shedding"
            value={mwh(bench["avoidable_unserved_mwh"] ?? 0, 3)}
            sub="target 0"
            tone={(bench["avoidable_unserved_mwh"] ?? 0) <= 1e-6 ? "good" : "critical"}
          />
          <Stat
            label="Cost win rate"
            value={pct(winRate)}
            sub="target ≥ 75%"
            tone={winRate >= 75 ? "good" : "critical"}
          />
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Test</th>
                <th>Pass condition</th>
                <th className="right">Passed</th>
                <th className="right">Failed</th>
                <th className="right">Rate</th>
              </tr>
            </thead>
            <tbody>
              {invariants.map((row) => (
                <tr key={row.key}>
                  <td>
                    <strong>{row.label}</strong>
                  </td>
                  <td className="small secondary">{row.pass_condition}</td>
                  <td className="right num">{row.passed}</td>
                  <td className="right num">{row.failed}</td>
                  <td className="right">
                    {row.applicable === 0 ? (
                      <Chip tone="neutral">never arose</Chip>
                    ) : (
                      <Chip tone={row.failed === 0 ? "good" : "critical"}>
                        {pct(row.rate * 100, 1)}
                      </Chip>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {failing.length > 0 && (
          <div className="stack" style={{ marginTop: 12, gap: 8 }}>
            {failing.map((row) => (
              <div className="banner critical" key={row.key}>
                <div className="grow">
                  <strong>
                    {row.label} — {row.failed} failure(s) of {row.applicable}
                  </strong>
                  <ul className="small secondary" style={{ margin: "4px 0 0", paddingLeft: 18 }}>
                    {row.failures.map((f, i) => (
                      <li key={i} className="mono">
                        {f}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        )}
        {notApplicable.length > 0 && (
          <p className="tiny muted" style={{ marginTop: 8 }}>
            {notApplicable.map((i) => i.label).join(", ")} never had a case to judge in
            this run — reported as not applicable rather than as a pass.
          </p>
        )}
      </Panel>

      <div className="cols two">
        <Panel title="Outcome benchmark" hint="against both baselines">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th className="right">Agent</th>
                  <th className="right">Rule-based</th>
                  <th className="right">No storage</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Energy cost (cash)</td>
                  <td className="right">{usd(bench["agent_cost_usd"] ?? 0)}</td>
                  <td className="right">{usd(bench["rule_based_cost_usd"] ?? 0)}</td>
                  <td className="right">{usd(bench["no_storage_cost_usd"] ?? 0)}</td>
                </tr>
                <tr>
                  <td>Net cost (incl. storage)</td>
                  <td className="right">{usd(bench["agent_net_cost_usd"] ?? 0)}</td>
                  <td className="right">{usd(bench["rule_based_net_cost_usd"] ?? 0)}</td>
                  <td className="right muted">—</td>
                </tr>
                <tr>
                  <td>Carbon</td>
                  <td className="right">{tonnes(bench["agent_co2_t"] ?? 0)}</td>
                  <td className="right">{tonnes(bench["rule_based_co2_t"] ?? 0)}</td>
                  <td className="right">{tonnes(bench["no_storage_co2_t"] ?? 0)}</td>
                </tr>
                <tr>
                  <td>Load shed</td>
                  <td className="right">{mwh(bench["agent_unserved_mwh"] ?? 0, 2)}</td>
                  <td className="right">{mwh(bench["rule_based_unserved_mwh"] ?? 0, 1)}</td>
                  <td className="right muted">—</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="statrow" style={{ marginTop: 12 }}>
            <Stat
              label="Cash cost vs rule-based"
              value={deltaPct(costDelta)}
              tone={costDelta <= 0 ? "good" : "critical"}
              sub="PRD target: lower"
            />
            <Stat
              label="Net-cost win rate"
              value={pct(netWinRate)}
              tone={netWinRate >= 75 ? "good" : "critical"}
              sub="PRD target: ≥ 75% of episodes"
            />
            <Stat
              label="Carbon vs rule-based"
              value={deltaPct(co2Delta)}
              tone={co2Delta <= 0 ? "good" : "critical"}
              sub="PRD target: no worse"
            />
          </div>
          <p className="tiny muted" style={{ marginTop: 8 }}>
            Net cost credits the change in the value of stored energy, which the PRD's
            objective includes and its KPI table omits. At an episode boundary the agent is
            holding charge it has paid for, so both are shown. On this run the cost and
            carbon targets are <strong>not met</strong>; the shedding targets are.
          </p>
        </Panel>

        <Panel title="Decision latency" hint="per-episode P95 against the 1 s budget">
          <Histogram values={latencies} unit="ms (P95 per episode)" budget={1000} />
          <div className="statrow" style={{ marginTop: 10 }}>
            <Stat
              label="Median episode P95"
              value={`${median(latencies).toFixed(0)} ms`}
              tone="good"
            />
            <Stat
              label="Worst episode P95"
              value={`${Math.max(...latencies, 0).toFixed(0)} ms`}
              tone={Math.max(...latencies, 0) < 1000 ? "good" : "critical"}
              sub="budget 1,000 ms"
            />
          </div>
        </Panel>
      </div>

      <Panel title="Per-episode detail" hint={`${rows.length} episodes`} flush scroll>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th className="right">#</th>
                <th>Preset</th>
                <th className="right">Seed</th>
                <th className="right">Start</th>
                <th className="right">Agent cost</th>
                <th className="right">Rule-based</th>
                <th className="right">Agent shed</th>
                <th className="right">Rule shed</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const agentCost = r.agent["energy_cost_usd"] ?? 0;
                const ruleCost = r.rule_based["energy_cost_usd"] ?? 0;
                return (
                  <tr key={r.episode}>
                    <td className="right num">{r.episode}</td>
                    <td>{r.preset.replace(/_/g, " ")}</td>
                    <td className="right num">{r.world_seed}</td>
                    <td className="right num">{r.start}</td>
                    <td className="right">
                      <span
                        style={{
                          color:
                            agentCost <= ruleCost ? "var(--success-text)" : "var(--critical)",
                        }}
                      >
                        {usd(agentCost)}
                      </span>
                    </td>
                    <td className="right">{usd(ruleCost)}</td>
                    <td className="right">{mwh(r.agent["unserved_mwh"] ?? 0, 2)}</td>
                    <td className="right">{mwh(r.rule_based["unserved_mwh"] ?? 0, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2
    : (sorted[mid] ?? 0);
}
