# Decisions

Every ambiguity in `Renewable Energy Orchestrator PRD.md` that I resolved myself, with the
PRD text it resolves and the choice made. Nothing here contradicts the PRD; these are gaps
it deliberately leaves to implementation.

## D1 — Tech stack (PRD gives none)

The PRD's "v1 technology" column is deliberately generic ("stream ingest + time-series
store", "engine module"). Chosen with the user: **Python 3.13 + NumPy + scikit-learn
backend, React + TypeScript + Vite frontend.**

- NumPy because the dispatch simulation vectorises over 80 clusters × 24 scenarios = 1920
  parallel states stepped 16 times — a handful of array operations, comfortably inside the
  1 s budget.
- scikit-learn's `GradientBoostingRegressor(loss="quantile")` rather than LightGBM: the PRD
  specifies gradient-boosted quantile regression, sklearn implements it natively, and it
  installs from wheels with no OpenMP/Homebrew dependency.

## D2 — The synthetic world is the data source (PRD "Assumptions")

The PRD assumes SCADA/EMS and ISO interfaces exist. None do here, so a synthetic `World`
model sits behind the ingest interface. Real adapters (IEC 60870-5-104, OPC UA, ISO market
APIs) replace it without touching the engine. This is why the reference scenario is both the
demo *and* the acceptance-test fixture — exactly as the PRD intends.

## D3 — Wind corridor assignment (PRD "Reference scenario")

The PRD annotates "Gale Point 120 (north corridor), Coastal Bluff 100 (south)" but leaves
Ridgeback's corridor unstated. **Ridgeback and Gale Point are north; Coastal Bluff is south.**
Reason: that puts 280 MW behind a 220 MW corridor limit (derating to 150 MW), which is the
only assignment that makes the corridor constraint bind and the "line congestion" event
meaningful. With Ridgeback in the south the corridor would never be tight and the constraint
would be dead code. `test_wind_fleet_and_corridor_matches_prd` asserts the binding property
rather than my specific assignment, so a different split that still binds would pass.

## D4 — Battery efficiency and wear pairing (PRD "Reference scenario")

The PRD gives "94–95% one-way efficiency" and "wear $14 and $18 per MWh" for the two
batteries without saying which is which. **Harbor BESS (120 MW / 480 MWh) gets 95% and $14;
Valley BESS (80 MW / 320 MWh) gets 94% and $18** — the larger unit is treated as the newer,
better one. The test asserts the *set* of values, not the assignment, so the pairing is a
free parameter.

## D5 — Cloud region assignment (PRD "Three cloud regions, so clouds hit farms unevenly")

Mesa Verde + Dry Creek → region_a (210 MW), Redrock + Alder Flats → region_b (140 MW),
Pinecrest → region_c (50 MW). The test asserts the regions are unevenly sized, which is the
property the PRD actually states.

## D6 — Carbon accounting (PRD open question)

The PRD asks "Is carbon accounted at average or marginal grid intensity for reporting?"
**Average**, at the stated 0.42 t/MWh, which is what the reference scenario's figures imply.
It is `carbon.accounting` in config so marginal can be added without an engine change.

## D7 — Value of lost load (PRD open question)

The PRD asks for the utility's VOLL and whether it should differ by consumer class. The
reliability objective is in MWh-equivalent and does not need a dollar VOLL, so no number is
load-bearing; `value_of_lost_load` is in config (default $9,000/MWh, critical $30,000/MWh)
for reporting only, and the question stays open.

## D8 — Simulated clock, not wall-clock (PRD "Determinism and audit")

Events and maintenance windows are indexed by interval number rather than timestamped. The
PRD requires decisions be "replayable bit-for-bit"; anchoring to wall-clock time would make
replay depend on when it was run.

## D9 — Approval queue in batch mode (PRD "Autonomy levels" vs F3.2)

The PRD requires one-click operator approval for high-impact actions, and separately requires
Monte Carlo over 96 intervals × up to 24 paths. Those are incompatible with a human in the
loop, so **batch mode auto-approves and records that fact in the trace**, keeping simulated
and live runs distinguishable in the audit log.

## D10 — Storm guardrail beyond the look-ahead (PRD defect fix)

The PRD's fix is to "extend the guardrail to the full alert lead time (up to 12 h)", but the
optimizer's horizon is only 4 h, so there is no simulated SoC at a 12-hour arrival to test.
The guardrail instead requires a **feasible pre-charge ramp**: given the cluster's trajectory
over the 4 h horizon and the remaining time to arrival, could the fleet still reach 55% by
charging at its rated power? Clusters that cannot are discarded. This is implemented in M6.

## D11 — Load-shift payment rate (PRD gives DR rates but not shift)

The PRD's net-cost definition includes "DR and shift payments", and it gives both DR
rates ($120/MWh to customers, $300/MWh from ISO calls), but no rate for load shifting.
Set to **$35/MWh** in `market.yaml` — materially cheaper than DR, which is the right
ordering: shifting defers consumption within the day, whereas DR destroys it. The number
is config, not code, so it is a single-line change if the real contract differs.

## D12 — "Lowest-value units first" for curtailment (PRD "Always-on reflexes")

The PRD says corridor and export limits "curtail the lowest-value units first" without
defining value. Renewables all have ~zero marginal cost, so there is no price-based
ordering. Implemented as: **corridor spill necessarily falls on units behind the
constrained corridor** (physics, not choice), and **limit-driven spill follows a configured
merit order, solar before wind**, attributed pro-rata within each group. The engine records
curtailment per farm so the per-farm burden report the PRD asks for weekly is possible.

## D13 — The two state-of-charge floors (PRD "Reference scenario" vs "Always-on reflexes")

The PRD gives "SoC 8-95%" as the operating range and separately says batteries "discharge
to the emergency floor before any load is shed". Read literally those conflict: if 8% is
the floor, there is nothing below it to discharge into. Resolved as **two floors**:

- `soc_min_pct` = 8% is the **hard floor**, reached only by the emergency cascade;
- `reserve_floor_pct` = 10% is the **reserve floor** that normal dispatch holds above.

So "discharge to the emergency floor" means dipping from 10% down to 8% to avoid shedding.
The invariant suite checks both: nothing ever breaches 8%, and discharging below 10% is
only legal with the tie-line maxed out.

## D14 — Precedence when the PRD's reflexes conflict (PRD "Always-on reflexes")

Three reflexes can demand opposite things in one interval: curtail for a transmission
limit, inject for a frequency dip, and discharge rather than shed load. The PRD lists all
three as non-optional but does not order them. Implemented precedence, highest first:

1. **Frequency reflex** (inject below 49.9 Hz) — a safety reflex; it survives both the
   curtailment rule and the export cap.
2. **Discharge before shedding** — served load beats spilled energy.
3. **No discharge while curtailing for a limit** — the PRD's open curtailment defect.

Justification: grid stability and served load are the two things the PRD refuses to trade
for money, and it lifts the stability weight 2.5x in a frequency event for exactly this
reason. The invariant tests encode the exceptions explicitly rather than weakening the
rule, so a discharge during limit-driven curtailment still fails unless it is the forced
reflex amount or the tie-line is maxed.

## D15 — Calibration widens only (PRD risk: "forecast models under-state uncertainty")

The PRD's first listed risk is forecasts under-stating uncertainty, mitigated by
"calibrate scenario spread against historical forecast errors; monitor P10-P90 coverage
weekly". Implemented as a per-target, per-lead band-width multiplier fitted on held-out
episodes. **The multiplier is floored at 1.0**, so calibration can widen a band but never
sharpen one. Letting it sharpen solar (90% -> 81% coverage) looked like a free win but was
fitted in-sample: on seeds used for neither training nor calibration the sharpened band
covered only 67%. An over-wide band costs sharpness; an under-wide one causes the reserve
shortfalls the PRD is worried about.

## D16 — `max_cloud_attenuation` is a shared physical parameter

The solar P10 model predicted 0 MW at midday, which cannot happen — cloud cuts direct
irradiance but diffuse light remains. Rather than special-casing the forecaster, the
attenuation limit (0.85) is now `solar_response.max_cloud_attenuation` in
`portfolio.yaml`, read by both the world model and the forecaster's per-lead physical
bounds. A forecast therefore cannot predict a physically unreachable collapse, and the
two components cannot drift apart.

## D17 — A price spike raises the price rather than scaling it

The injected `PRICE_SPIKE` event originally multiplied the current price by its
magnitude. During a midday surplus the price is negative, so a "spike" made it *more*
negative — the opposite of a scarcity event. A spike now lifts the price to
`typical x magnitude` and relaxes back toward the modelled price with the configured
45-minute persistence, so it is monotonically upward by construction.

## D18 — Corridor spill is not absorbable by storage (PRD "Justified curtailment")

The invariant reads "spill only if the corridor is full, the export line is full or the
price is negative, **with batteries absorbing what they can**". Applied literally to
corridor spill it fails constantly: energy stranded behind a congested corridor cannot
reach storage on the far side of it, so the batteries demonstrably *cannot* absorb it.
The absorption clause is therefore checked against export-limit and negative-price spill
only. Corridor spill is still required to be genuinely corridor-driven.

## D19 — Storm readiness is measured over *available* capacity

Under the contingency-cascade preset, Harbor BESS (480 of 800 MWh) is tripped when the
storm arrives, making a 50% *fleet* charge level physically unreachable. Measuring against
total nameplate would make the invariant untrue rather than strict, so it is measured over
online capacity and the report states what fraction was online.

## D20 — Both of the PRD's net-cost definitions are reported

The PRD's net-cost *objective* includes "- change in value of stored energy"; its KPI
table defines energy cost as "imports minus exports, DR and shift payments" and omits it.
These disagree at an episode boundary where the agent is holding charge it has paid for
but not yet used. Rather than pick the flattering one, the episode report prints both:
`energy_cost_usd` (the KPI definition) and `net_cost_usd` (the objective definition).

## D21 — Episode start intervals are randomised

Fixing the 8-hour episodes to start at 09:00 meant none of them reached the evening peak,
so load shedding never arose for either strategy and the agent paid for reliability it
never needed. Starts are randomised across the day, which is what revealed the 541 MWh of
shedding the rule-based baseline incurs and the agent avoids.

## D22 — "Bit-for-bit" has to mean across processes, not just twice in one

The world model derived each wind farm's small per-farm speed offset from
`hash(farm_id)`. Python salts string hashing per process, so every wind figure — and
therefore every price, cost, carbon and shedding number downstream — depended on which
interpreter produced it. The determinism invariant compares two runs inside one process
and could not see it, which is why `make test-m7` passed or failed depending on what
else was in the same pytest session.

Replaced with a `blake2s` digest, which keeps the property the PRD actually asks for
(farms in one corridor are not identical) while being stable. Added
`test_replay_is_reproducible_across_processes`, which runs a short episode in three
subprocesses under different `PYTHONHASHSEED` values and requires identical clusters,
price path and cost. **All measured figures in PLAN.md were re-recorded after this fix;
the pre-fix ones were not reproducible.**

## D23 — When a live plan is dispatched, and what an event does to it

The PRD describes a 15-minute receding-horizon loop and separately requires a re-plan
"within one evaluation (< 1 s)" of an event arriving, without saying how the two meet in
a live session. Resolved as:

- **A plan is dispatched when it is made.** `step()` applies the standing plan to the
  plant, advances the clock 15 minutes, then decides and dispatches the new interval. So
  the console always shows set-points that are already out to the assets and the physical
  response appears on the next tick — which is what a control room looks like.
- **An event re-plans its own interval.** Injecting an event, confirming a bulletin or
  moving a weight slider re-decides the *current* interval without moving the clock, and
  reports the plan before against the plan after. The superseded plan stays in the
  append-only audit log and `replay()` returns the latest one for that interval.
- **A superseded plan's approvals are withdrawn.** An approval request describes one
  interval's plan; once that plan is replaced, or the clock has passed it, approving it
  would dispatch a set-point no current plan called for. Pending requests are retired as
  `superseded` and the gateway refuses to action anything that is not `pending`.
- **The warm-up history is generated in batch mode and says so.** The console must open
  mid-morning with real history behind it, and there is no operator to approve anything
  before the session exists (same reasoning as D9). Routing 36 warm-up intervals through
  the LLM would also cost 36 API calls to produce prose nobody reads.

## D24 — Which price window each economic signal is read from

Three signals are day-scale by definition — what a MWh left in storage is worth, and the
prices below and above which charging and discharging make sense — and all three were
being read off the 4-hour look-ahead. A 4-hour window is not a day and is systematically
unrepresentative of one: midday is the cheapest stretch there is and the evening peak the
most expensive. The consequences were both directions of the same error:

- At 09:00 the stored-energy value came out at **$16.48/MWh against a $31.16/MWh
  break-even**, so the midday surplus was exported instead of saved — the exact failure
  the PRD's clause exists to prevent ("without this the short horizon drains batteries
  before the evening peak").
- At 18:00 under a storm the charge band reached **$69/MWh while a stored MWh was worth
  $48**, so grid energy was bought above what storing it was worth, paying the spread and
  the wear and emitting 0.42 t/MWh on top.

Resolved by splitting the two by the kind of question each asks:

- **Buying is a day-scale decision.** The charge band and the stored-energy value both
  come off a 24-hour price view, assembled at one value per interval: each scenario's
  expected path for the intervals the look-ahead covers, the rolling price window for the
  rest, and the configured typical price for intervals with neither (the stated prior for
  a day nobody has observed yet, rather than a fill-in from the cheap hours). The charge
  band is then capped at the stored-energy value, because on a flat price view the day's
  30th percentile still sits a round-trip loss above it.
- **Selling is a receding-horizon decision.** "Discharge at the best moment inside the
  window I can actually see" is what the look-ahead's upper band expresses, so the sell
  band stays window-local. Making it day-scale as well was tried and measured: it
  suppressed discharging across three strategies and the no-storage baseline and cost
  **54% more over 30 episodes**, so it was reverted.

## D25 — Console rendering choices the PRD leaves open

**No charting library.** The six chart forms the console needs — a stacked area with a
forecast band, two line panels, a scatter, a fan and two bar forms — are a few dozen lines
of SVG each, and all of them need exact control over details a library abstracts away: a
2px surface gap between stacked segments, a crosshair that snaps to an interval, a reserve
floor drawn as a rule rather than a series. The geometry is unit-tested instead
(`src/lib/scale.test.ts`), which is cheaper than configuring a library to do the same.

**Price and state of charge are two charts, not one with two axes.** The PRD asks for "a
second chart for price and battery SoC with the plan's projected SoC". Putting $/MWh and
% on one pair of axes would be a dual-axis chart, which is never correct — the two scales
are arbitrary against each other, so any crossing the reader sees is an artefact of the
scaling. They render as two panels sharing one x-axis, which gives the same read honestly.

**The colour orderings were validated, not chosen.** Each series ordering was run through
the palette validator's six checks in both light and dark before anything was drawn, on
the pairlist appropriate to the form: adjacent pairs for stacks and bars, all pairs for
the scatter. The first supply-stack ordering put violet beside blue and failed dark mode
at CVD ΔE 1.9 against an 8.0 target; it was re-ordered rather than shipped.

**Verification without a browser.** Browser tooling was unavailable, so the PRD's "browser
walkthrough" was replaced with two repeatable things: the console mounted in jsdom and
asserted against API responses recorded from the running backend (`make fixtures`), and
`make demo`, which drives the PRD's seven demo steps through real HTTP. Those cover the
data contract and the behaviour. **They do not cover appearance** — layout, spacing, label
collisions, the responsive collapse and dark mode are unverified by eye, and that is
recorded as a gap rather than assumed away.
