# Renewable Energy Orchestrator — Product Requirements



## Overview

The Renewable Energy Orchestrator is an AI agent that re-plans a utility's renewable portfolio every 15 minutes and acts on it: charging or discharging batteries, buying or selling power, curtailing generation, calling demand response, shifting industrial load and rescheduling maintenance. It optimizes for cost, carbon, curtailment, battery wear, reliability and grid stability at once, and states which goal won each trade-off and why.

**Problem.** Solar and wind output swings within minutes, prices spike without warning, and assets fail. Operators today juggle separate tools for forecasting, battery scheduling and market bidding, and fall back on fixed rules ("charge on surplus, discharge on deficit"). Fixed rules cannot look ahead to a storm, hold energy for a higher evening price, or weigh a $40k saving against a reliability risk.

**Product.** Three AI layers work as one agent:

- **LLM orchestrator.** Reasons about the situation, plans, and acts through tools: read portfolio state, run forecasts, evaluate action clusters, execute set-points, reschedule maintenance, dispatch crews. It also reads free-text storm bulletins and ISO notices, and answers operator questions such as "why did you curtail Ridgeback?" from the decision trace.
- **ML forecasters.** Quantile models for solar, wind, demand and price produce P10/P50/P90 forecasts that seed a fan of scenarios.
- **Stochastic optimizer.** The trusted numeric core. It scores 80 action clusters across every scenario under situation-adaptive criteria and hard safety guardrails.

The LLM decides what to look at and what to do next; the optimizer decides the numbers; guardrails decide what is never allowed. The same engine runs a 24-hour Monte Carlo simulator, so operators can test the strategy before a volatile day.

**The novel idea.** The agent changes its own optimality criteria as the situation changes, within fixed safety guardrails, and says what it gave up. Examples: shifting weight to reliability when a storm is forecast, or to cost during a price spike. Most dispatch tools optimize one fixed objective and cannot explain a trade-off.

**Why now.** Renewable share is rising sharply, batteries are cheap enough to deploy at scale, and markets increasingly pay for flexibility (demand response, fast frequency response). The value now sits in coordination, not in any single asset.

## 9-blocker position

**Claimed position: F3 × D2.** The solution covers all three feature levels and handles mostly structured plus textual inputs with demonstrated reliability. It does not claim D3.

| Axis | Rubric definition | Where this solution sits | Why |
| --- | --- | --- | --- |
| Features | Distinct, significant features + supporting features (observability, trainability, fault tolerance) | F3 | Optimal action clusters (F1), situation-specific criteria under uncertainty (F2), 24 h scenario simulation with Monte Carlo variation (F3); all four supporting features present |
| Depth | Input uncertainty / lack of structure handled + output reliability | D2 | Structured telemetry, forecasts and prices plus free-text bulletins; reliability shown by 11 automated invariants, randomized episodes and baseline benchmarks |

**Why not D3.** D3 needs highly heterogeneous multimodal input. Satellite cloud imagery, sky-camera nowcasts, drone inspection photos and scanned PDFs are designed for but not built. Claiming D3 would be overestimation. If an image-based cloud nowcast ships before submission, the claim moves to F3 × D3 only for that input path.

**Evidence matrix.** Each claimed cell is shown in the demo, delivered by a named component, and proved by a test.

| Claim | Shown in the demo | Delivered by | Proved by |
| --- | --- | --- | --- |
| F1 Optimal cluster of actions | Live operations: committed actions with MW per asset after every tick and every injected event | Optimizer (dispatch simulation over 80 clusters); actuation tools | Energy-balance, limits, rating, curtailment and shedding invariants; 960 decisions |
| F2 Criteria set per situation, adjusted for uncertainty | Decision trace: regimes and triggers, base vs final weights, guardrail removals, P10–P90 cost, Pareto chart, runner-up trade-off | Situation classifier, weight adapter, ML quantile forecasters, scenario fan, CVaR scoring, guardrails | Storm-readiness and frequency tests; same-seed determinism; slider changes re-rank within 1 s |
| F3 Simulation over time with varying inputs | Scenario simulation tab: 5 presets, 6–24 paths, fan charts, KPI table vs two baselines | Simulator (same engine in batch mode) | Benchmark: shedding cut 80–100% vs rule-based on all 5 presets |
| D1 Acceptable output on structured + text input | Any tick in normal conditions | Full pipeline | Lower total cost than rule-based on 23 of 30 random episodes |
| D2 Demonstrable reliability | Reliability tests tab: pass/fail counts, failing examples, latency | Guardrails, invariant suite, fallback controller, audit log | 9 of 11 invariants at 100%; two defects listed with fixes (see Validation) |
| Text inputs | Paste a storm bulletin → extracted fields with source phrases → operator confirms → agent re-plans | LLM extractor with JSON-schema validation; rules parser as fallback | Extraction test set of labelled bulletins (target ≥ 95% field accuracy) |

## Evaluation criteria coverage

Each judging criterion has a specific answer and a place in the submission where a judge can verify it.

| Criterion | Our answer | Where to verify |
| --- | --- | --- |
| 1. Significance and relevance | Variable renewables plus price volatility make coordination the bottleneck; every input and action in the brief is covered | Overview, Inputs, Action space |
| 2. Innovation and originality | Agent adapts its own optimality criteria to the situation within fixed guardrails and explains what it gave up | Optimality criteria; decision trace in the demo |
| 3. Effective use of AI | LLM orchestrator with tools, ML quantile forecasters, stochastic optimizer; each does what it is best at | AI architecture |
| 4. Technical complexity and execution | Receding-horizon stochastic control, 80 clusters × 24 scenarios × 16 steps in about 100 ms, CVaR risk, Pareto analysis, Monte Carlo simulator | Decision loop, Uncertainty handling, Validation |
| 5. Agentic capability | Perceives, plans, calls tools, acts every 15 min, re-plans on events, falls back on errors, asks approval for high-impact actions | AI architecture, Supporting features |
| 6. Business and user impact | Shedding cut 80–100%, energy cost down 9–11% in 4 of 5 scenarios, carbon trade-off stated openly | Business impact |
| 7. Prototype quality and usability | Working console that opens mid-morning, one-click events, plain-language explanations | Demo and UX |
| 8. Scalability, Responsible AI, robustness | Grounded LLM, guardrails, human approval gates, audit replay, linear scaling in assets | Responsible AI, Non-functional requirements, Reliability |

## Goals and success metrics

Success means zero avoidable load shedding first, then lower total system cost than rule-based dispatch on at least 75% of days.

**Goals**

1. Decide and act every 15 minutes without an operator in the loop for routine actions.
2. Make the optimality criteria explicit, adapt them to the situation, and show how conflicts were resolved.
3. Plan against uncertainty, not a single forecast.
4. Prove reliability with automated invariant tests and scenario benchmarks.

**Non-goals (v1)**

- Replacing the EMS/SCADA protection layer or the ISO's own frequency control.
- Unit commitment of thermal plants, network power-flow (AC OPF) or voltage control.
- Bidding strategy into day-ahead auctions (v1 acts on real-time prices; day-ahead bidding is phase 3).
- Fully autonomous high-impact actions (load shedding, maintenance cancellation) without operator approval.

| Metric | Definition | Target | Prototype result (6 paths × 5 presets) |
| --- | --- | --- | --- |
| Avoidable load shedding | MWh shed while import or battery headroom existed | 0 | 0 across 960 tested intervals |
| Load shed vs rule-based | Total unserved MWh per day | −75% | −80% to −100% |
| Energy cost vs rule-based | Imports minus exports, DR and shift payments | Lower in ≥ 4 of 5 presets | −9% to −11% in 4 presets; +8% in contingency cascade |
| Carbon vs rule-based | t CO₂ from imports | No worse than rule-based | +10% to +27% — not met; agent charges reserve from the grid |
| Carbon vs no storage | t CO₂ from imports | −15% | −16% to −20% |
| Renewable utilization | Used ÷ available renewable MWh | ≥ 97% on typical days | 100% typical; 93% volatile; 97% storm |
| Reserve compliance | Intervals meeting the fast-reserve requirement | ≥ 98% | 85–99% (rule-based 70–77%) — not met under contingencies |
| Storm readiness | Battery fleet ≥ 50% when a forecast storm hits | 100% | 2 of 9 — open defect |
| Decision latency | Time to evaluate all options, P95 | < 1 s (budget 900 s) | 12 ms (optimizer only; LLM step budgeted at < 10 s) |
| Explainability | Decisions with regime, weights, runner-up and trade-off | 100% | 100% |

## Users

The primary user is the control-room operator who supervises the agent; everyone else consumes its plans and reports.

| User | What they need from the agent | Key screens |
| --- | --- | --- |
| Control-room operator | See what the agent is doing and why; approve high-impact actions; override in one click | Live operations, decision trace, action log |
| Portfolio / trading manager | Set priorities (cost vs carbon vs reliability), risk appetite, and see P&L impact | Operator priorities, scenario simulator |
| Asset manager | Maintenance windows that cost the least lost output; inspection dispatch after faults | Maintenance plan, action log |
| Grid reliability engineer | Evidence that hard limits are never breached | Reliability test suite, audit log |
| Sustainability lead | Carbon avoided, renewable utilization, curtailment | Daily KPI report |

## Reference scenario

The demo and acceptance tests use one synthetic utility: 400 MW of solar, 380 MW of wind, 200 MW / 800 MWh of storage and 220–330 MW of load, connected through a 220 MW import / 200 MW export tie-line. Capacities are illustrative and configurable.

| Asset | Units | Key parameters |
| --- | --- | --- |
| Solar farms (5) | Mesa Verde 120 MW, Dry Creek 90, Redrock 80, Alder Flats 60, Pinecrest 50 | Three cloud regions, so clouds hit farms unevenly |
| Wind farms (3) | Ridgeback 160 MW, Gale Point 120 (north corridor), Coastal Bluff 100 (south) | Cut-in 3 m/s, rated 12.5 m/s, cut-out 25 m/s; north corridor limit 220 MW |
| Batteries (2) | Harbor BESS 120 MW / 480 MWh; Valley BESS 80 MW / 320 MWh | 94–95% one-way efficiency; SoC 8–95%; wear $14 and $18 per MWh throughput |
| Industrial consumers | Northgate Steel 60 MW (20 MW shiftable), Cobalt Data Center 40 MW (critical), Sinclair Chemicals 35 MW (12 MW DR), Polar Cold Storage 15 MW (8 MW shiftable) | Plus municipal and commercial load of 95–180 MW |
| Grid interconnection | Import 220 MW, export 200 MW | Congestion derates export to 80 MW and the north corridor to 150 MW |
| Market | Real-time price, $6/MWh buy spread, carbon $85/t at 0.42 t/MWh grid intensity | DR pays customers $120/MWh; ISO DR calls pay $300/MWh |

**Situations the agent must handle** (each can be injected live in the demo): cloud bank, wind surge, price spike, battery trip, line congestion, industrial demand surge, storm alert with uncertain arrival time, frequency dip, ISO demand-response call, and a weather-forecast revision.

## Inputs

Every input is normalized into one state snapshot per 15-minute interval; anything stale or missing is flagged and replaced by the forecast, never silently zeroed.

| Domain | Signal | Cadence | Format (v1) | Used for |
| --- | --- | --- | --- | --- |
| Renewables | Solar output per farm | 1–5 min | SCADA telemetry, MW | Current balance |
| Renewables | Solar forecast with uncertainty | 15 min, 24–48 h ahead | P10/P50/P90 MW | Scenario fan |
| Renewables | Wind output and speed per farm | 1–5 min | MW, m/s | Balance, cut-out risk |
| Renewables | Wind forecast | 15 min | P10/P50/P90 m/s | Scenario fan |
| Grid | Current demand per consumer | 1–5 min | MW | Balance |
| Grid | Demand forecast | 15 min | P10/P50/P90 MW | Scenario fan |
| Grid | Frequency | 1 s (aggregated) | Hz | Frequency-response trigger |
| Grid | Transmission limits | On change | MW per line/corridor | Hard constraints |
| Batteries | State of charge, availability | 1 min | %, on/off | Dispatch limits |
| Batteries | Charge/discharge efficiency, max rate | Static + health updates | %, MW | Dispatch physics |
| Batteries | Degradation cost | Daily | $/MWh throughput | Wear objective |
| Market | Electricity prices (real-time, day-ahead) | 5–15 min | $/MWh | Cost objective, thresholds |
| Market | Carbon price | Daily | $/t | Carbon objective |
| Market | Demand-response incentives and calls | On event | $/MWh, window | DR decision |
| External | Weather forecast and revisions | Hourly | Gridded or point forecast | Driver of scenarios |
| External | Storm alerts | On event | Text bulletin + structured fields | Reserve and readiness |
| External | Maintenance schedules | On change | Asset, window, deadline, crew | Maintenance planning |

**Text inputs.** Storm bulletins, ISO notices and maintenance notes arrive as free text. In v1 an LLM extracts them into a fixed JSON schema: event type, start, end, region, affected assets, severity, and the source phrase for each field. The output is validated against the schema and checked for consistency (dates in the future, known asset names). A rules parser is the fallback when the LLM is unavailable or the output fails validation. Low-confidence or high-impact extractions, such as a storm alert that would raise reserve, need operator confirmation before they reach the optimizer.

## Action space

The agent chooses one **action cluster** per interval: a battery strategy, a market posture, a curtailment rule, and on/off for demand response and load shifting. That gives 5 × 2 × 2 × 2 × 2 = **80 clusters**, each simulated over the look-ahead. The chosen cluster is then turned into concrete MW set-points per asset.

| Dimension | Options | Concrete actions it produces |
| --- | --- | --- |
| Battery strategy | Soak surplus · Price arbitrage · Hold for higher price · Reserve for emergencies · Discharge now | Charge / discharge each BESS (MW), reserve capacity for emergencies, delay selling until prices rise |
| Market posture | Spot (import/export freely) · Batteries before imports | Buy electricity, sell electricity |
| Curtailment rule | Curtail at negative prices · Curtail only when forced | Curtail wind, curtail solar, prioritize clean generation |
| Demand response | Off · On (at peak price or ISO call) | Trigger DR at Sinclair Chemicals, reduce noncritical loads |
| Load shifting | Off · On (at peak, repaid off-peak) | Shift Northgate Steel and Polar Cold Storage load; restore it later |

**Always-on reflexes** (applied inside every cluster, not optional):

- Corridor and export limits curtail the lowest-value units first.
- A frequency dip below 49.9 Hz forces battery injection.
- If load would exceed import limit plus battery output, batteries discharge to the emergency floor before any load is shed.

**Maintenance actions** run as a separate planner on the same forecast: delay a planned window when it would lose more than $2,500 of output compared with the best window before its deadline, delay any window that overlaps a storm (crew safety), start and close work orders, and dispatch inspection teams after a battery trip or turbine cut-out.

**Autonomy levels.** Routine set-points execute automatically. Load reduction beyond contracted DR, maintenance moves and any action outside configured bounds require one-click operator approval in v1.

## Optimality criteria

Conflicts are resolved in a fixed order: hard safety guardrails first, then a weighted score whose weights adapt to the situation, then a stability tie-break. Reliability is never traded for money below the guardrail line.

**Six objectives, all minimized over a 4-hour look-ahead:**

| Objective | Measure | Unit | Default weight |
| --- | --- | --- | --- |
| Net cost | Imports × (price + spread) − exports × price + DR and shift payments − change in value of stored energy, risk-adjusted | $ | 0.30 |
| Reliability risk | 5 × unserved MWh + 0.4 × reserve-shortfall MWh, plus a tail (CVaR90) term | MWh-equivalent | 0.20 |
| Carbon | Imports × 0.42 t/MWh | t CO₂ | 0.18 |
| Curtailment | Renewable energy spilled | MWh | 0.14 |
| Grid stability | Ramping of grid exchange + missing frequency headroom | index | 0.10 |
| Battery wear | Throughput × $/MWh degradation | $ | 0.08 |

Maximization goals map onto these: profit = −net cost, renewable utilization = −curtailment, grid stability and reliability as above.

**Step 1 — Guardrails (lexicographic, applied before any trade-off).**

1. Discard clusters with avoidable unserved load (more than 0.25 MWh above the best cluster).
2. Under a storm alert, discard clusters whose batteries would be below 55% at forecast arrival.
3. During a frequency dip, discard clusters that do not inject battery power.

**Step 2 — Situation-adaptive weights.** The agent classifies the interval into one or more regimes and multiplies the operator's base weights, then renormalizes. The operator sets the base weights and risk appetite; the agent only shifts them.

| Regime | Trigger | Weight shift | Risk aversion |
| --- | --- | --- | --- |
| Grid stress | Storm alert or storm on site | Reliability ×2.4, stability ×1.6, cost ×0.7, curtailment ×0.7 | +0.6 |
| Frequency event | Frequency < 49.95 Hz | Stability ×2.5, reliability ×1.6 | +0.3 |
| Asset contingency | A battery offline | Reliability ×1.7, wear ×0.8 | +0.3 |
| Scarcity ahead | Deficit within 3 h > 75% of import limit | Reliability ×1.8, cost ×1.1 | +0.3 |
| Price spike | Price ≥ max(1.6 × 70th percentile, $150) | Cost ×1.9, reliability ×1.2, carbon ×0.8 | — |
| Peak pricing | Price ≥ 70th percentile of the day | Cost ×1.35 | — |
| Transmission constrained | Export derated or corridor full | Curtailment ×1.4, stability ×1.3 | — |
| Renewable surplus | Renewables exceed load by > 25 MW | Curtailment ×2, carbon ×1.3 | — |
| Negative prices | Price < $0 | Cost ×1.4, curtailment ×0.6 | — |
| DR program active | ISO DR call | Cost ×1.3 | — |

**Step 3 — Score.** Each objective is min–max normalized across the surviving clusters (with a floor on the range so trivial differences are not amplified), multiplied by its weight and summed. The lowest score wins.

**Step 4 — Stability tie-break.** If the previous interval's cluster scores within 0.015 of the winner, the agent keeps it, avoiding set-point flip-flopping.

**Explaining the trade-off.** Every decision shows the active regimes and why, the final weights against the operator's base, the top five clusters, the Pareto front, and the runner-up with the objectives where they differ, e.g. "chosen plan costs $1.2k more but removes 3.4 MWh of shedding risk; reliability weight 0.35 decides".

## Decision loop

The agent is a receding-horizon planner wrapped in a perceive–decide–act loop; each stage is a separate, testable service.

&#91;embedded content: decision loop · 8 stages per 15-minute interval\]

Stages 1–4 build the picture and the criteria, 5–7 search and filter the options, and 8 executes only the first interval before the loop starts again with fresh data.

| Component | Responsibility | Stage | v1 technology |
| --- | --- | --- | --- |
| Ingest and state store | Normalize telemetry, prices, forecasts, alerts into one snapshot; flag stale data | 1 | Stream ingest + time-series store |
| LLM text extractor | Turn bulletins and notices into schema-validated events | 1 | LLM with JSON schema; rules fallback |
| ML forecasters | P10/P50/P90 for solar, wind, demand, price per 15-min step, 4–24 h ahead | 2 | Gradient-boosted quantile regression |
| Scenario generator | Correlated scenario paths consistent with the quantiles and known events | 2 | Stochastic driver model |
| Situation classifier | Regimes and their measured triggers | 3 | Rules over state and forecasts (v1) |
| Weight adapter | Operator base weights × regime multipliers; risk level | 4 | Config + multipliers tuned from overrides |
| Optimizer | Dispatch simulation, CVaR scoring, Pareto front, hysteresis | 5, 7 | Engine module (< 1 s) |
| Guardrails | Hard safety filters before any trade-off | 6 | Engine module, not overridable by the LLM |
| LLM orchestrator | Runs the loop through tools, decides when to re-plan, drafts maintenance moves, writes the explanation | 1–8 | LLM with tool calling |
| Actuation gateway | Bounds check, approval queue, signed set-points to EMS | 8 | Service |
| Explainer and audit log | Decision trace, runner-up text, replay | 8 | Append-only store |
| Simulator | 24 h Monte Carlo with baselines | — | Same engine, batch mode |

## AI architecture

Each AI technique does the job it is best at: the LLM reasons, reads text and explains; ML forecasts; the optimizer computes; guardrails enforce.

&#91;embedded content: AI architecture · models, agent tools and the action path\]

The orchestrator never writes a set-point itself: it calls evaluate\_options, and the numbers it executes are the optimizer's, filtered by guardrails.

| Model or technique | Used for | Why this choice |
| --- | --- | --- |
| LLM with tool calling (orchestrator) | Runs the loop, decides when to re-plan, sequences tools, drafts maintenance moves, writes explanations, answers operator questions | Handles open-ended situations and language; cannot be trusted with arithmetic, so it delegates numbers |
| LLM with JSON schema (extractor) | Turns bulletins and notices into structured events with source phrases | Text is varied and unstructured; schema validation catches bad output |
| Gradient-boosted quantile regression | P10/P50/P90 forecasts for solar, wind, demand, price | Accurate on tabular weather data, fast to retrain nightly, gives calibrated uncertainty |
| Stochastic receding-horizon optimizer | Scores 80 action clusters across 24 scenarios with CVaR risk | Exact, deterministic and testable; decides in about 100 ms |
| Preference tuning from overrides | Adjusts regime multipliers from accepted operator overrides | Lets the agent learn local priorities without retraining a model |

**Agent tools.**

| Tool | What it does | Side effects |
| --- | --- | --- |
| get\_state | Current snapshot of all assets, limits, prices, alerts | None |
| run\_forecast | Quantile forecasts and the scenario fan for a horizon | None |
| evaluate\_options | Scores all clusters under current criteria; returns ranking, guardrail results, Pareto front | None |
| execute\_actions | Sends the chosen cluster's set-points to the actuation gateway | Changes the grid; bounded, logged, approval-gated when high-impact |
| plan\_maintenance | Values maintenance windows; proposes delays | Needs approval |
| dispatch\_crew | Raises an inspection work order | Needs approval |
| explain | Builds the trade-off explanation from the decision trace | None |

**Error recovery.** A failed tool call is retried once, then the orchestrator falls back to the last valid plan. If the LLM is unavailable, the optimizer runs the loop alone on schedule with template explanations. If the optimizer fails, the gateway applies rule-based dispatch with reserve held.

## Uncertainty handling

The agent never plans against a single forecast: it scores every cluster across a fan of 20–24 scenarios and penalizes bad tails, not just bad averages.

- **Scenario generation.** The ML quantile forecasts set the P10–P90 envelope per step. Within it, each scenario evolves the underlying drivers (cloud cover per region, wind speed per corridor, demand noise, price noise) as mean-reverting stochastic processes from the current measured state. Scenario 0 is the expected path. Spread grows with lead time and with a volatility factor that rises in unsettled weather.
- **Known events.** Announced events enter all scenarios: a congestion derate or ISO DR window with known end, a storm with its forecast arrival jittered by about ±0.6 h per scenario. Unannounced surprises (a cloud bank, a spike) are seen once they start and assumed to decay with a typical persistence (cloud 1.5 h, price spike 45 min, demand surge 3 h).
- **Risk-adjusted cost.** Cost score = mean + λ × (CVaR90 − mean). λ starts at the operator's setting (default 0.4) and rises in stress regimes, up to 1.5.
- **Tail reliability.** The reliability objective adds the CVaR90 of unserved energy, so a plan that is fine on average but sheds load in the worst 10% of scenarios loses.
- **Value of stored energy.** Energy left in batteries at the end of the look-ahead is valued at the upper-quartile price of the next 24 hours × 0.93, so a short horizon does not drain batteries cheaply before the evening peak.
- **Receding horizon.** Only the first interval of the winning plan is executed; the whole problem is re-solved 15 minutes later with fresh data, so forecast errors are corrected continuously.

```latex
\text{score}(a) = \sum_{o} w_o(\text{regime}) \cdot \frac{J_o(a) - \min_{a'} J_o(a')}{\max\big(\text{range}_o,\ \text{floor}_o\big)}, \qquad J_{\text{cost}}(a) = \mathbb{E}[C] + \lambda\big(\text{CVaR}_{0.9}[C] - \mathbb{E}[C]\big)
```

## Functional requirements

All three feature levels are in scope for v1; F3 reuses the F1/F2 engine unchanged.

**F1 — Determine the optimal cluster of actions**

| ID | Requirement | Acceptance |
| --- | --- | --- |
| F1.1 | Enumerate all 80 action clusters every interval | All 80 scored; count shown in the decision trace |
| F1.2 | Simulate each cluster with a dispatch model that enforces SoC, power ratings, efficiencies, import/export and corridor limits | Zero limit violations in the test suite |
| F1.3 | Select one cluster and translate it into per-asset set-points (MW per battery, MW bought/sold, MW curtailed by farm, DR and shift MW) | Action list shows verb, asset, MW and reason for every set-point |
| F1.4 | Re-plan immediately when an event arrives, not only on the next tick | New plan within one evaluation (< 1 s) of the event |
| F1.5 | Plan maintenance: delay, start, complete, dispatch inspection | Delay only when savings > $2,500 or a storm overlaps the window |

**F2 — Determine optimality criteria for the situation, then the optimal cluster, adjusting for uncertainty**

| ID | Requirement | Acceptance |
| --- | --- | --- |
| F2.1 | Detect active regimes from state, forecasts and events | Each regime shown with the measured trigger, e.g. "Spot $310 vs typical $52" |
| F2.2 | Derive weights and risk aversion from operator base × regime multipliers | Base and final weights shown side by side |
| F2.3 | Apply guardrails before trade-offs and report how many clusters each removed | Guardrail list with counts |
| F2.4 | Score on a scenario fan with CVaR tail penalties | P10–P90 cost range shown for the top clusters |
| F2.5 | Show the Pareto front and the runner-up with per-objective deltas | Every decision carries a runner-up explanation |
| F2.6 | Let operators change base weights and risk appetite and see the decision change | Re-evaluation within 1 s of a slider change |

**F3 — Simulate a scenario over time with varying conditions**

| ID | Requirement | Acceptance |
| --- | --- | --- |
| F3.1 | Run 24 h (96 intervals) under preset or custom scenarios | Presets: typical day, volatile weather, storm passage, evening price crisis, contingency cascade |
| F3.2 | Monte Carlo over N paths with randomized weather noise and event timing | N selectable (6/12/24); same random numbers across strategies for a fair comparison |
| F3.3 | Benchmark the agent against rule-based dispatch and a no-storage baseline | KPI table with mean and P10–P90 per strategy |
| F3.4 | Visualize cumulative cost and battery SoC as fan charts, and the agent's strategy mix by hour | Charts render for every preset |
| F3.5 | Let operators replay a path step by step in the live view | Any simulated path can be opened in Live operations |

## Reliability and solution depth

v1 targets D2: structured and textual inputs with demonstrable reliability. D3 (multimodal) is phase 3.

| Level | Inputs | Reliability bar | How this product meets it |
| --- | --- | --- | --- |
| D1 | Mostly structured + text | Acceptable output in most situations | Deterministic dispatch physics, guardrails, regime-adaptive scoring |
| D2 (v1 target) | Mostly structured + text | High, demonstrable | Automated invariant suite on every build, randomized episodes, baseline benchmarks, reproducible decisions, full audit log, LLM extraction checked against a schema and a labelled test set |
| D3 (phase 3) | Highly heterogeneous multimodal | High, demonstrable | Adds satellite cloud imagery, sky-camera nowcasts, drone/thermal inspection images, ISO PDFs and voice radio logs via extractors that emit the same structured events, each with confidence and a fallback |

**Reliability test suite (runs on every build and nightly on 30+ randomized episodes):**

| Test | Pass condition |
| --- | --- |
| Energy balance | Generation − curtailment + imports + discharge = load + charge + exports (+ unserved), every interval |
| SoC limits | 8% ≤ SoC ≤ 95% |
| Power ratings | No unit above its MW rating; tripped units at 0 MW |
| Transmission limits | Imports, exports and corridor flow within current (derated) limits |
| No wash trading | Never import and export in the same interval |
| Justified curtailment | Spill only if corridor full, export line full or price negative, with batteries absorbing what they can |
| Unavoidable shedding only | Unserved load only with imports at the limit and batteries at full discharge |
| Frequency response | Batteries inject during every dip below 49.9 Hz |
| Storm readiness | Fleet ≥ 50% charged when a forecast storm arrives |
| Latency | Every decision < 1 s |
| Determinism | Same inputs and seed produce the same cluster |
| Outcome benchmark | Total cost ≤ rule-based on ≥ 75% of episodes |

**Fail-safe behavior.** If the agent misses a cycle, a data feed is stale beyond 2 intervals, or the solver errors, the system falls back to the rule-based dispatch with reserve held, raises an alarm, and logs the cause. Operator override always wins and is logged.

## Supporting features

The rubric counts supporting features on the features axis; all four are in v1.

| Feature | What it does | How it is shown |
| --- | --- | --- |
| Observability | Every decision stored with input snapshot, seed, forecasts, regimes, weights, guardrail removals, top 5 clusters, actions and LLM tool calls; replayable | Decision trace and action log in the demo; replay any past interval |
| Trainability | Forecasters retrain nightly on new actuals. Operator overrides are logged as preference data; regime multipliers are re-tuned weekly by replaying history to minimize disagreement with accepted overrides, within bounds | Override button in the demo; "weights learned from overrides" panel |
| Fault tolerance | Stale or missing feed → forecast substitution with a warning. Optimizer error or missed cycle → rule-based dispatch with reserve held. LLM timeout or invalid tool call → optimizer runs on its own and the explanation is template-generated. Battery trip → immediate re-plan and inspection dispatch | Battery-trip and data-dropout injections in the demo; fallback events in the log |
| Configurability | Assets, limits, prices, weights, multipliers and guardrails in versioned config; every change logged | Operator priorities sliders in the demo |

## Responsible AI

The LLM can propose and explain, but it can never invent a number or bypass a guardrail: every MW figure comes from the optimizer, and every action passes the actuation gateway.

| Risk | Control |
| --- | --- |
| Hallucinated numbers or actions | The LLM acts only through typed tools with bounded arguments. Set-points come from optimizer output, never from LLM text. Every number in an explanation is checked against the decision trace before display; a mismatch falls back to the template explanation. |
| Prompt injection via bulletins or notices | Extracted text is treated as data, never as instructions. Extraction output must match the event schema. Events that raise reserve or move maintenance need operator confirmation. |
| Unsafe autonomy | Guardrails sit below the LLM and cannot be overridden by it. High-impact actions (load reduction beyond contracted DR, maintenance moves, anything outside configured bounds) wait in an approval queue. One-click operator override, always logged. |
| Unfair burden on customers | DR calls and load shifts rotate across eligible consumers and respect contract caps per day and month. Critical loads (data center, municipal) are never curtailed by the agent. Curtailment burden per farm is reported weekly. |
| Security | OT network segmentation, read-only telemetry, signed set-point commands, role-based access, LLM calls without credentials or control-network access. |
| Privacy | Industrial consumer data is aggregated in explanations shown outside the control room; no personal data is sent to the LLM. |
| Robustness and drift | Forecast calibration (P10–P90 coverage) monitored weekly; the invariant suite runs on every build and nightly; fallback to rule-based dispatch on any failure. |
| Accountability | Each decision is replayable with its inputs, model versions, prompts and tool calls. |

## Demo and UX

The working demo is a single web console with four tabs. It must open in a realistic running state (a morning already in progress), never an empty screen.

**1. Live operations**

- Header: simulated clock, Run / Pause, Step 15 min, speed, Reset.
- Portfolio panel: every solar and wind farm (output vs capacity, curtailed or under maintenance), both batteries (SoC, MW, availability), each industrial consumer, tie-line and corridor flow vs limits, price, carbon price, DR status, storm alerts, maintenance plan.
- Supply-and-demand chart: last 12 h actuals stacked by source, next 4 h as a P10–P90 band, a "now" line. A second chart for price and battery SoC with the plan's projected SoC.
- Event injector: one button per situation (cloud bank, wind surge, price spike, battery trip, congestion, demand surge, storm alert, frequency dip, ISO DR call, forecast revision), plus a text box that parses a pasted bulletin.
- Decision trace, in loop order: situation (regimes and why) → criteria (base vs final weights, risk) → options evaluated (80 clusters × scenarios, guardrail removals, top 5 with P10–P90 cost) → Pareto scatter → committed actions → why this over the runner-up.
- "Response to change" banner after an injected event: plan before → plan after, with the changed actions.
- Action log, newest first, with every decision and event.

**Agent panel** (beside the decision trace):

- Tool-call trace for the current interval: which tools the orchestrator called, in order, with arguments and a one-line result.
- Ask the agent: free-text questions ("why are you holding Harbor BESS at 70%?") answered from the decision trace, with the cited figures linked to their source.
- Approval queue: high-impact proposals with Approve / Reject; rejections are logged as preference data.
- Data-dropout injection: cut a feed to show forecast substitution and fallback behavior.

**2. Scenario simulation (F3):** preset, paths, run with progress, KPI comparison table, cumulative-cost and SoC fan charts, agent strategy mix by hour.

**3. Reliability tests (D2):** each test with pass/fail counts and failing examples, win rate vs rule-based, latency distribution.

**4. How it decides:** the loop, objectives, regime table, guardrails and known limitations, in plain language.

**Demo script (7 minutes), each step tied to a judging criterion:**

1. 09:00, solar surplus: the agent charges batteries and explains why it is not selling yet. *(F1, criterion 2)*
2. Inject a cloud bank: the agent re-plans within one interval; show the tool-call trace. *(F1, criterion 5)*
3. Paste a storm bulletin: the LLM extracts arrival time and region, the operator confirms, weights shift to reliability, batteries pre-charge, the Gale Point maintenance move goes to the approval queue. *(F2, D2 text input, criteria 3 and 8)*
4. Ask the agent "what did you give up for this?" and get the runner-up trade-off with numbers. *(F2, criterion 3)*
5. 18:30 price spike: the agent discharges into the spike and triggers DR. *(F1, criterion 6)*
6. Battery trip plus data dropout: immediate re-plan, inspection dispatched, forecast substitution flagged. *(criteria 5 and 8)*
7. Scenario simulation and Reliability tests tabs: the benchmark and invariant evidence. *(F3, D2, criteria 4 and 6)*

**UX rules:** state shown in form as well as numbers (status chips, severity colors); every number carries its unit; the agent's text uses operator vocabulary ("Discharge Harbor BESS 80 MW @ $142"), never internal names; works on a laptop and degrades to one column on a tablet.

## Non-functional requirements

| Area | Requirement |
| --- | --- |
| Latency | Decision P95 < 1 s for 80 clusters × 24 scenarios × 16 steps; full cycle including data ingest < 60 s |
| Availability | 99.9% for the decision service; automatic fallback to rule-based dispatch within one interval |
| Determinism and audit | Every decision stored with its inputs snapshot, seed, weights, guardrail results, top clusters and actions; replayable bit-for-bit |
| Safety | Agent issues set-points only within configured bounds; protection and EMS limits remain authoritative; operator override in one click |
| Security | OT network segmentation; read-only telemetry ingest; signed set-point commands; role-based access (operator, manager, engineer, viewer) |
| Data quality | Stale (> 2 intervals), missing or out-of-range inputs flagged and replaced by forecast with a visible warning |
| Scalability | Linear in assets; 50 assets and 200 clusters within the latency budget on one server |
| Integrations | SCADA/EMS via IEC 60870-5-104 or OPC UA; ISO market APIs; weather provider APIs; CMMS for work orders |
| Configurability | Assets, limits, prices, weights, regime multipliers and guardrails in versioned configuration, changes logged |
| Explainability | Every action traceable to a decision with runner-up and trade-off text |

## Validation evidence

The prototype engine already passes 9 of 11 invariants on every interval tested and never shed load; storm readiness and one curtailment edge case are open defects. Results below are from 30 randomized 8-hour episodes (960 decisions, 2–4 random events each) on synthetic data.

| Test | Passed | Failed | Note |
| --- | --- | --- | --- |
| Energy balance | 960 | 0 |  |
| SoC limits | 960 | 0 |  |
| Power ratings | 960 | 0 |  |
| Transmission limits | 960 | 0 |  |
| No wash trading | 960 | 0 |  |
| Unavoidable shedding only | 960 | 0 |  |
| Frequency response | 20 | 0 | Only intervals with a dip are tested |
| Latency | 960 | 0 | P95 12 ms |
| Determinism | pass |  | Same seed, same 24 decisions |
| Justified curtailment | 958 | 2 | Battery discharging at full power while solar was curtailed at 09:00 ("Discharge now" cluster) |
| Storm readiness | 2 | 7 | Fleet at 14–34% when the storm hit; guardrail only checks 4 h ahead |

Against rule-based dispatch, the agent had lower total cost in 23 of 30 episodes (77%).

Scenario benchmarks against the two baselines are under Business impact below.

**Fixes required before v1:**

- [ ] Storm readiness: extend the guardrail to the full alert lead time (up to 12 h) and add a pre-charge target to the "Reserve" strategy.
- [ ] Curtailment edge case: forbid battery discharge in any interval where renewables are curtailed for export limits.
- [ ] Carbon: fill reserve from midday surplus before grid imports, and price carbon into the reserve-charging decision.
- [ ] Re-run the suite on 200 episodes and on 90 days of historical data.

## Business impact

The main value is reliability: the agent avoids 80–100% of the load shedding that rule-based dispatch causes, and lowers energy cost by 9–11% in four of five scenarios. It currently emits more carbon than the rule-based baseline, which is a fix before submission.

&#91;embedded content: prototype simulator · 5 scenarios × 6 paths × 96 intervals, synthetic portfolio\]

Remaining agent shedding happens only when imports are at the limit and batteries are empty; no avoidable shedding occurred.

| Scenario | Energy cost, agent | Energy cost, rule-based | Energy cost, no storage | CO₂ agent (t) | CO₂ rule-based (t) | CO₂ no storage (t) |
| --- | --- | --- | --- | --- | --- | --- |
| Typical day | $187k | $207k | $259k | 831 | 753 | 1,028 |
| Volatile weather | $98k | $107k | $112k | 757 | 596 | 922 |
| Storm passage | $108k | $121k | $138k | 709 | 627 | 886 |
| Price crisis | $306k | $337k | $403k | 817 | 729 | 1,010 |
| Contingency cascade | $172k | $160k | $195k | 799 | 682 | 951 |

**Annualized estimate.** Scaling the typical day to 365 days, versus rule-based dispatch the agent saves about $7.1M of energy cost and avoids about 47,700 MWh of load shedding a year, but adds about 28,700 t CO₂. Versus a portfolio with no storage control it avoids about 71,800 t CO₂ a year. These are synthetic-portfolio figures to be re-run on historical data; real years mix calm and stressed days.

**Who benefits.** Industrial consumers get fewer interruptions and paid DR; the utility gets lower procurement cost and higher renewable utilization (93–100%); operators get a plan with its reasoning instead of a black box.

## Risks, assumptions and open questions

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Forecast models under-state uncertainty | Plans too aggressive, reserve shortfalls | Calibrate scenario spread against historical forecast errors; monitor P10–P90 coverage weekly |
| Simplified network model (no power flow) | Set-points that violate local limits | Keep EMS limits authoritative; add corridor constraints per substation in phase 2 |
| Operator distrust of autonomous actions | Low adoption | Shadow mode first; approval gates for high-impact actions; clear explanations |
| Regime multipliers are hand-tuned | Mis-weighted trade-offs | Tune on historical replays; expose multipliers in versioned config; review quarterly |
| Market rules differ by ISO | Wrong prices or penalties modeled | Market adapter per ISO; validate against settlement data |
| Battery degradation cost is approximate | Over- or under-cycling | Update $/MWh from vendor warranty curves and measured capacity fade |

**Assumptions**

- Telemetry at 1–5 minute resolution and price feeds are available through existing SCADA/EMS and ISO interfaces.
- The utility can send battery, curtailment and DR set-points through existing control channels.
- DR and load-shift contracts with the named industrial consumers allow automated calls within agreed limits.

**Open questions**

- Which ISO and market product (real-time only, or day-ahead too) is the first deployment target?
- What is the utility's value of lost load, and should it differ by consumer class (the data center is critical)?
- Which actions must stay operator-approved permanently, and which can graduate to autonomous after shadow mode?
- Is carbon accounted at average or marginal grid intensity for reporting?

## Build plan to submission

The engine is built and tested; the AI layer, the console and the evidence pack remain, each with a gate that must pass before the next step.

&#91;embedded content: build plan · 4 steps, 3 gates, order of work\]

**Beyond the hackathon:** shadow mode on live utility data, then supervised control of batteries and market actions, then multimodal inputs (D3) and day-ahead bidding.
