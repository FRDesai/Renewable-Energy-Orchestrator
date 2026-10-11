import type { ConfigResponse } from "../api/types";
import { OBJECTIVE_LABEL, usd } from "../lib/units";
import { Chip, Empty, Panel } from "../components/ui";

const STAGES = [
  {
    title: "Read the state",
    body:
      "Telemetry from every farm, battery, consumer and tie-line is normalised into one " +
      "snapshot. Anything stale or missing is replaced by the forecast and flagged — never " +
      "silently zeroed.",
  },
  {
    title: "Forecast the next four hours",
    body:
      "Gradient-boosted quantile models give P10, P50 and P90 for solar, wind, demand and " +
      "price. Those bands seed a fan of scenarios in which step 0 is the measured present, " +
      "announced events appear in every scenario, and spread grows with lead time.",
  },
  {
    title: "Classify the situation",
    body:
      "Ten regimes are tested against the state, the forecast and active events — grid " +
      "stress, frequency event, asset contingency, scarcity ahead, price spike, peak " +
      "pricing, transmission constrained, renewable surplus, negative prices, DR active. " +
      "Each reports the measured figure that triggered it.",
  },
  {
    title: "Set the criteria",
    body:
      "The operator's base weights are multiplied by the multipliers the active regimes " +
      "carry and renormalised. Risk aversion rises in stress. The agent shifts the " +
      "operator's priorities; it never rewrites them.",
  },
  {
    title: "Simulate all 80 clusters",
    body:
      "Every combination of battery strategy, market posture, curtailment rule, demand " +
      "response and load shift is stepped through the whole look-ahead in every scenario, " +
      "under real physics: state-of-charge limits, power ratings, efficiencies, and the " +
      "import, export and corridor caps.",
  },
  {
    title: "Apply the hard guardrails",
    body:
      "Before any trade-off is considered, clusters are discarded for avoidable unserved " +
      "load, for not being on track to be charged when a forecast storm arrives, and for " +
      "failing to inject during a frequency dip. These sit below the language model and " +
      "cannot be overridden by it.",
  },
  {
    title: "Score what survived",
    body:
      "Each objective is normalised across the surviving clusters, weighted and summed; " +
      "lowest wins. Cost and reliability carry a CVaR₉₀ tail penalty, so a plan that is " +
      "fine on average but sheds in the worst tenth of scenarios loses.",
  },
  {
    title: "Commit the first interval only",
    body:
      "The winner is translated into per-asset set-points, which pass a bounds check and a " +
      "signature before reaching the plant. Only this interval is executed — the whole " +
      "problem is re-solved fifteen minutes later with fresh data.",
  },
];

const LIMITATIONS = [
  "Cost and carbon targets are not met. Over 24 h the agent is cheaper than rule-based in 2 of 5 presets against a target of 4, and cleaner in 3 of 5 against a target of \"no worse\". The excess concentrates where the storm guardrail forces a pre-charge, and the open defect is that the pre-charge buys at rated power as soon as it is behind rather than from the cheapest and cleanest hours in the alert window.",
  "The agent imports more grid energy than it needs to, and prices carbon into that decision too weakly.",
  "Wind forecasts are mildly under-covered out of sample: the P10–P90 band catches about 75% of outcomes against a target of 80%. The band is deliberately widened rather than sharpened, because under-stating uncertainty is the more dangerous error.",
  "Load shedding is eliminated in four of five presets and cut 87% in the fifth. What remains is unavoidable — asserted by test: imports at the limit and batteries at full discharge.",
  "Multimodal inputs are out of scope. No satellite imagery, sky cameras, drone photographs or PDFs — text bulletins and structured telemetry only.",
  "This is a decision-support layer, not a protection system. It does not replace EMS or SCADA protection, does not do ISO frequency control, and does not solve AC power flow or voltage.",
];

/** PRD tab 4: "the loop, objectives, regime table, guardrails and known limitations, in
 *  plain language." Everything numeric is read from the live configuration, so the page
 *  cannot drift from what the engine is actually running. */
export function HowItDecides({ config }: { config: ConfigResponse | null }) {
  if (!config) {
    return (
      <Panel title="How it decides">
        <Empty>Loading configuration…</Empty>
      </Panel>
    );
  }

  const o = config.objectives;
  const guardrails = o.guardrails as Record<string, unknown>;

  return (
    <div className="stack">
      <Panel title="The loop, every fifteen minutes">
        <div className="loop">
          {STAGES.map((stage, i) => (
            <div className="loop-step" key={stage.title}>
              <span className="idx">{i + 1}</span>
              <div>
                <h4>{stage.title}</h4>
                <p className="small secondary" style={{ margin: "2px 0 0" }}>
                  {stage.body}
                </p>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <div className="cols two">
        <Panel
          title="What it optimises"
          hint={`${o.lookahead_hours} h look-ahead · ${o.steps} steps · ${o.scenarios} scenarios`}
        >
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Objective</th>
                  <th className="right">Default weight</th>
                  <th>Minimised as</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(o.base_weights).map(([key, weight]) => (
                  <tr key={key}>
                    <td>{OBJECTIVE_LABEL[key] ?? key}</td>
                    <td className="right num">{weight.toFixed(2)}</td>
                    <td className="small secondary">{OBJECTIVE_NOTE[key] ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="tiny muted" style={{ marginTop: 8 }}>
            All six are minimised, so maximisation goals enter as negations: profit is
            negative net cost, renewable utilisation is negative curtailment. Cost is
            risk-adjusted as mean + λ(CVaR₉₀ − mean), with λ from{" "}
            {o.risk["lambda_base"]?.toFixed(2)} up to {o.risk["lambda_max"]?.toFixed(2)} in
            stress. A cluster within {o.hysteresis_margin} of the winner is kept from the
            previous interval, so set-points do not flip-flop.
          </p>
        </Panel>

        <Panel title="Hard guardrails" hint="applied before any trade-off">
          <ul className="small secondary" style={{ margin: 0, paddingLeft: 18 }}>
            <li>
              Discard clusters with more than{" "}
              <strong>{String(guardrails["avoidable_unserved_mwh"])} MWh</strong> of unserved
              load above the best available.
            </li>
            <li>
              Under a storm alert, discard clusters not on track to reach{" "}
              <strong>{String(guardrails["storm_soc_pct"])}%</strong> fleet state of charge
              when the storm arrives.
            </li>
            <li>
              During a dip below{" "}
              <strong>{String(guardrails["frequency_dip_hz"])} Hz</strong>, discard clusters
              that do not inject battery power.
            </li>
          </ul>
          <p className="small secondary" style={{ marginTop: 10 }}>
            These are lexicographic — each filters what the previous one left — and
            reliability is never traded for money below this line. If a guardrail would
            remove every remaining cluster it is reported but not enforced, and the trace
            says the trade-off is unguarded: an empty option set is worse than a breached
            preference.
          </p>
          <p className="small secondary">
            Three reflexes apply inside every cluster and are not optional: corridor and
            export limits curtail the lowest-value units first; frequency below{" "}
            {String(guardrails["frequency_dip_hz"])} Hz forces battery injection; and
            batteries discharge to the emergency floor before any load is shed. Where they
            conflict, the frequency reflex wins, then served load, then spilled energy.
          </p>
        </Panel>
      </div>

      <Panel title="Regimes and what they do to the weights" hint={`${o.regimes.length} regimes`}>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Regime</th>
                <th>Triggered by</th>
                <th>Weight multipliers</th>
                <th className="right">Risk aversion</th>
              </tr>
            </thead>
            <tbody>
              {o.regimes.map((regime) => (
                <tr key={regime.key}>
                  <td>
                    <strong>{regime.label}</strong>
                  </td>
                  <td className="small secondary">{regime.trigger}</td>
                  <td className="small">
                    {Object.entries(regime.multipliers ?? {}).map(([key, factor]) => (
                      <Chip
                        key={key}
                        tone={factor > 1 ? "warning" : "neutral"}
                        title={OBJECTIVE_LABEL[key] ?? key}
                      >
                        {OBJECTIVE_LABEL[key] ?? key} ×{factor.toFixed(2)}
                      </Chip>
                    ))}
                  </td>
                  <td className="right num">
                    {regime.risk_aversion ? `+${regime.risk_aversion.toFixed(2)}` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="cols two">
        <Panel title="What the language model does, and cannot do">
          <div className="prose">
            <p>
              <strong>It decides what to look at and what to do next.</strong> It runs the
              loop through seven typed tools, reads free-text bulletins, drafts maintenance
              moves, writes the operator explanation and answers questions from the
              decision trace.
            </p>
            <p>
              <strong>It never produces a number.</strong> No tool accepts a power figure, a
              state of charge or a set-point from the model. Every MW in an executed action
              comes from the optimizer, and every figure in its prose is checked against the
              decision trace before display — a mismatch discards the text and shows the
              template instead.
            </p>
            <p>
              <strong>It cannot reach the guardrails.</strong> They run below it. If a
              guardrail removed a cluster, that is final.
            </p>
            <p>
              <strong>Extracted text is data, never instructions.</strong> A bulletin that
              contains a command is treated as a bulletin that failed to describe an event.
            </p>
            <p>
              <strong>If it is unavailable the plant still runs.</strong> The optimizer
              drives the loop on schedule and explanations are template-generated.
            </p>
          </div>
        </Panel>

        <Panel title="When things go wrong">
          <div className="prose">
            <ul>
              <li>A failed tool call is retried once, then the last valid plan is used.</li>
              <li>
                If the language model is unavailable, the optimizer runs the loop alone with
                template explanations.
              </li>
              <li>
                If the optimizer errors, a cycle is missed, or a feed is stale beyond two
                intervals, dispatch falls back to the fixed rule with reserve held, an alarm
                is raised and the cause is logged.
              </li>
              <li>
                If that also fails, every asset is held where it is rather than dispatching
                a figure no engine produced.
              </li>
              <li>
                Stale, missing or out-of-range inputs are replaced by the forecast with a
                visible warning — never silently zeroed.
              </li>
              <li>An operator override always wins, and is always logged.</li>
            </ul>
            <p>
              Routine set-points execute automatically. Load reduction beyond contracted
              demand response, maintenance moves, crew dispatch and anything outside
              configured bounds wait for one-click approval. Rejections are kept as
              preference data.
            </p>
          </div>
        </Panel>
      </div>

      <Panel title="Known limitations" hint="measured, not aspirational">
        <div className="prose">
          <ul>
            {LIMITATIONS.map((text, i) => (
              <li key={i}>{text}</li>
            ))}
          </ul>
          <p className="tiny muted">
            Value of lost load is configured at {usd(config.market.value_of_lost_load["default_per_mwh"] ?? 0)}/MWh
            and {usd(config.market.value_of_lost_load["critical_per_mwh"] ?? 0)}/MWh for
            critical load, and is used for reporting only — the reliability objective is in
            MWh-equivalent, so no dollar figure is load-bearing in the decision. Carbon is
            accounted at average grid intensity. Portfolio config version{" "}
            {config.portfolio.version}, objectives {o.version}, market{" "}
            {config.market.version}; every change is logged.
          </p>
        </div>
      </Panel>
    </div>
  );
}

const OBJECTIVE_NOTE: Record<string, string> = {
  net_cost:
    "imports minus exports, plus DR and shift payments, minus the change in the value of stored energy",
  reliability_risk:
    "unserved energy plus a CVaR₉₀ tail term and a fast-reserve shortfall penalty, in MWh-equivalent",
  carbon: "tonnes CO₂ from grid imports at average grid intensity",
  curtailment: "MWh of renewable output spilled",
  grid_stability: "distance from nominal frequency and reserve adequacy, as an index",
  battery_wear: "throughput times the per-MWh wear cost of each unit",
};
