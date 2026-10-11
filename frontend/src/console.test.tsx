/** The console, rendered against real API responses.
 *
 * The fixtures in `src/__fixtures__` were captured from a running backend by
 * `scripts/capture-fixtures.mjs`, so these tests exercise the shapes the engine actually
 * produces. That is the point: TypeScript checks that the console reads the contract it
 * was told about, and this checks the contract was described correctly — a renamed key
 * or a null where an object was expected shows up here as a thrown render, not as a
 * blank panel in front of an operator.
 *
 * Each assertion is tied to something the PRD's "Demo and UX" section requires on screen.
 */
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "./api/client";

import configFixture from "./__fixtures__/config.json";
import openingFixture from "./__fixtures__/frame-opening.json";
import eventFixture from "./__fixtures__/frame-event.json";
import fallbackFixture from "./__fixtures__/frame-fallback.json";
import logFixture from "./__fixtures__/log.json";
import reliabilityFixture from "./__fixtures__/reliability.json";
import simulationFixture from "./__fixtures__/simulation.json";
import stormBulletin from "./__fixtures__/bulletin-storm.json";
import injectionBulletin from "./__fixtures__/bulletin-injection.json";

import type { ConfigResponse, EventSummary, Frame, LogEntry } from "./api/types";
import { EventInjector } from "./components/EventInjector";
import { LiveOperations } from "./tabs/LiveOperations";
import { HowItDecides } from "./tabs/HowItDecides";
import { ReliabilityTests } from "./tabs/ReliabilityTests";
import { ScenarioSimulation } from "./tabs/ScenarioSimulation";

const config = configFixture as unknown as ConfigResponse;
const opening = openingFixture as unknown as Frame;
const withEvent = eventFixture as unknown as Frame;
const fallback = fallbackFixture as unknown as Frame;
const log = (logFixture as { entries: LogEntry[] }).entries;

const noop = async () => null;
// `act` hands the caller whatever the request returned, exactly as the session hook does.
const passthrough = async <T,>(fn: () => Promise<T>) => fn();

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const live = (frame: Frame) =>
  render(
    <LiveOperations frame={frame} config={config} log={log} busy={false} act={noop} />,
  );

describe("Live operations", () => {
  it("opens mid-morning in a running state, never an empty screen", () => {
    live(opening);
    expect(opening.clock.time).toBe("09:00");
    // Portfolio: every asset from the reference scenario is on screen by name.
    for (const farm of config.portfolio.solar) {
      expect(screen.getByText(farm.name)).toBeTruthy();
    }
    for (const farm of config.portfolio.wind) {
      expect(screen.getAllByText(farm.name).length).toBeGreaterThan(0);
    }
    for (const battery of config.portfolio.batteries) {
      expect(screen.getAllByText(battery.name).length).toBeGreaterThan(0);
    }
    for (const consumer of config.portfolio.consumers) {
      expect(screen.getAllByText(consumer.name).length).toBeGreaterThan(0);
    }
  });

  it("shows the committed set-points in operator vocabulary with a unit", () => {
    live(opening);
    const executed = opening.actions.filter((a) => a.verdict === "executed");
    expect(executed.length).toBeGreaterThan(0);
    for (const action of executed) {
      // "Charge Harbor BESS 44.8 MW", never "charge harbor 44.78"
      const verb = action.verb.charAt(0).toUpperCase() + action.verb.slice(1);
      expect(
        screen.getAllByText((_, node) => {
          const text = node?.textContent ?? "";
          return text.startsWith(`${verb} ${action.asset_name}`);
        }).length,
      ).toBeGreaterThan(0);
    }
  });

  it("renders the decision trace in loop order with every guardrail reported", () => {
    live(opening);
    for (const heading of [
      "Situation",
      "Criteria",
      "Options evaluated",
      "Pareto front",
      "Committed",
      "Why this, and not the alternative",
    ]) {
      expect(screen.getAllByText(heading).length).toBeGreaterThan(0);
    }
    // Every guardrail is listed whether or not it fired.
    for (const guardrail of opening.trace.options.guardrails) {
      expect(screen.getAllByText(guardrail.rule).length).toBeGreaterThan(0);
    }
    // Base and final weights side by side, per F2.2.
    expect(screen.getByText("Base")).toBeTruthy();
    expect(screen.getByText("Final")).toBeTruthy();
  });

  it("plots the whole Pareto scatter, one mark per cluster", () => {
    const { container } = live(opening);
    const scatter = container.querySelector(
      'svg[aria-label^="Clusters plotted"]',
    ) as SVGElement | null;
    expect(scatter).not.toBeNull();
    const marks = scatter!.querySelectorAll("circle, path[stroke]");
    // 80 clusters, some drawn as crosses; the committed one is drawn twice (mark + ring
    // is one circle, so at least one mark per cluster must exist).
    expect(marks.length).toBeGreaterThanOrEqual(
      opening.trace.options.clusters_evaluated,
    );
  });

  it("stacks the supply chart by source and bands the forecast", () => {
    const { container } = live(opening);
    const chart = container.querySelector(
      'svg[aria-label^="Supply by source"]',
    ) as SVGElement | null;
    expect(chart).not.toBeNull();
    // four stacked layers + two forecast bands + two demand lines, all drawn
    expect(chart!.querySelectorAll("path").length).toBeGreaterThanOrEqual(6);
    expect(within(container).getByText("now")).toBeTruthy();
    for (const label of ["Solar", "Wind", "Battery discharge", "Import"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
  });

  it("keeps price and state of charge on separate axes, never a dual axis", () => {
    const { container } = live(opening);
    const price = container.querySelector('svg[aria-label^="Spot price"]');
    const soc = container.querySelector('svg[aria-label^="Fleet state of charge"]');
    expect(price).not.toBeNull();
    expect(soc).not.toBeNull();
    expect(price).not.toBe(soc);
  });

  it("offers one injector button per situation, plus the bulletin box", () => {
    live(opening);
    for (const meta of Object.values(config.injections)) {
      expect(screen.getAllByText(meta.label).length).toBeGreaterThan(0);
    }
    expect(screen.getByText("Extract event")).toBeTruthy();
  });

  it("shows the agent's tool-call trace and says the model cannot write a number", () => {
    live(opening);
    for (const call of opening.agent.tool_calls) {
      expect(screen.getAllByText(call.name).length).toBeGreaterThan(0);
    }
    expect(
      screen.getByText(/No tool accepts a power figure, state of charge or set-point/),
    ).toBeTruthy();
  });

  it("explains the degraded mode rather than looking broken when the LLM is off", () => {
    expect(opening.agent.llm_available).toBe(false);
    live(opening);
    expect(screen.getByText("Orchestrator narration unavailable")).toBeTruthy();
    expect(screen.getAllByText(/template/i).length).toBeGreaterThan(0);
  });

  it("shows the response-to-change banner after an injected event", () => {
    expect(withEvent.response_to_change).not.toBeNull();
    live(withEvent);
    const change = withEvent.response_to_change!;
    expect(screen.getByText(`Re-planned: ${change.trigger}`)).toBeTruthy();
    // The measured evaluation time is on screen, because it is the acceptance criterion.
    expect(screen.getByText(/new plan in \d+ ms/)).toBeTruthy();
  });

  it("surfaces the frequency reflex as an alarm and in the set-point reason", () => {
    live(withEvent);
    const r = withEvent.trace.committed.interval;
    expect(r.frequency_hz).toBeLessThan(49.9);
    expect(r.forced_injection_mw).toBeGreaterThan(0);
    expect(screen.getByText(/Frequency reflex/)).toBeTruthy();
  });

  it("shows the rule-based fallback and the stale feed that caused it", () => {
    expect(fallback.status.mode).toBe("rule_based_fallback");
    live(fallback);
    expect(screen.getAllByText(/stale/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText("fallback").length).toBeGreaterThan(0);
    // Substitution is stated, not hidden: the PRD forbids silent zeroing.
    for (const channel of fallback.data_quality.substituted) {
      expect(screen.getAllByText(new RegExp(channel)).length).toBeGreaterThan(0);
    }
  });

  it("lists the action log newest first", () => {
    live(opening);
    const intervals = log.map((e) => e.interval);
    expect(intervals).toEqual([...intervals].sort((a, b) => b - a));
    expect(screen.getByText("Action log")).toBeTruthy();
  });

  it("renders every frame fixture without throwing", () => {
    for (const frame of [opening, withEvent, fallback]) {
      cleanup();
      expect(() => live(frame)).not.toThrow();
    }
  });
});

describe("Scenario simulation (F3)", () => {
  it("shows the stored benchmark immediately, with the KPI table and both fan charts", async () => {
    vi.spyOn(api, "simulation").mockResolvedValue(simulationFixture as never);

    const { container } = render(<ScenarioSimulation config={config} />);
    expect(await screen.findByText("KPI comparison")).toBeTruthy();

    for (const label of ["Energy cost", "Carbon", "Load shed", "Battery throughput"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
    for (const strategy of ["Agent", "Rule-based", "No storage"]) {
      expect(screen.getAllByText(new RegExp(strategy)).length).toBeGreaterThan(0);
    }
    expect(container.querySelector('svg[aria-label^="$ across paths"]')).not.toBeNull();
    expect(container.querySelector('svg[aria-label^="% across paths"]')).not.toBeNull();
    // F3.5: a path is a preset and a seed, so it can be reopened in Live operations.
    expect(screen.getByText("Replay a path in Live operations")).toBeTruthy();
  });
});

describe("Reliability tests (D2)", () => {
  it("shows every invariant with its counts, and the failing examples when there are any", async () => {
    vi.spyOn(api, "reliability").mockResolvedValue(reliabilityFixture as never);

    render(<ReliabilityTests />);
    expect(await screen.findByText("Invariant suite")).toBeTruthy();

    const rows = reliabilityFixture.invariants ?? [];
    expect(rows.length).toBeGreaterThanOrEqual(12);
    for (const row of rows) {
      expect(screen.getAllByText(row.label).length).toBeGreaterThan(0);
      expect(screen.getAllByText(row.pass_condition).length).toBeGreaterThan(0);
    }
    // The benchmark is shown against its PRD target, pass or fail.
    expect(screen.getByText("Cost win rate")).toBeTruthy();
    expect(screen.getAllByText(/target/i).length).toBeGreaterThan(0);
    // A missed target is stated plainly rather than omitted.
    expect(screen.getAllByText(/not met/).length).toBeGreaterThan(0);
  });

  it("plots the latency distribution against the 1 s budget", async () => {
    vi.spyOn(api, "reliability").mockResolvedValue(reliabilityFixture as never);

    const { container } = render(<ReliabilityTests />);
    await screen.findByText("Decision latency");
    expect(
      container.querySelector('svg[aria-label^="Distribution of decision latency"]'),
    ).not.toBeNull();
    expect(screen.getByText("budget")).toBeTruthy();
  });
});

describe("How it decides", () => {
  it("explains the loop, the objectives, the regimes and the guardrails from live config", () => {
    render(<HowItDecides config={config} />);
    expect(screen.getByText("The loop, every fifteen minutes")).toBeTruthy();

    // Every regime the engine knows about is described with its trigger.
    for (const regime of config.objectives.regimes) {
      expect(screen.getAllByText(regime.label).length).toBeGreaterThan(0);
      expect(screen.getAllByText(regime.trigger).length).toBeGreaterThan(0);
    }
    // The weights shown are the configured ones, so the page cannot drift from the engine.
    for (const [, weight] of Object.entries(config.objectives.base_weights)) {
      expect(screen.getAllByText(weight.toFixed(2)).length).toBeGreaterThan(0);
    }
    expect(screen.getByText("Known limitations")).toBeTruthy();
    expect(
      screen.getByText(/Cost and carbon targets are not met/),
    ).toBeTruthy();
    expect(screen.getByText(/Multimodal inputs are out of scope/)).toBeTruthy();
  });
});

describe("Bulletin extraction (D2 text input)", () => {
  const typeAndExtract = async (response: unknown) => {
    vi.spyOn(api, "bulletin").mockResolvedValue(response as never);
    render(
      <EventInjector config={config} frame={opening} busy={false} act={passthrough} />,
    );
    const box = screen.getByRole("textbox");
    fireEvent.change(box, { target: { value: "a bulletin" } });
    fireEvent.click(screen.getByText("Extract event"));
    await screen.findByText(/extracted by|No event extracted/);
  };

  it("shows the extracted event with its label, arrival and confidence", async () => {
    await typeAndExtract(stormBulletin);
    const event = (stormBulletin as { event: EventSummary }).event;
    // The three fields that were missing from this endpoint's shape, now asserted on
    // screen rather than trusted: a hand-written type could not catch their absence.
    expect(event.label).toBeTruthy();
    expect(event.intervals_to_arrival).not.toBeNull();
    expect(typeof event.extraction_confidence_pct).toBe("number");

    expect(
      screen.getByText(new RegExp(`${event.label} extracted by the`)),
    ).toBeTruthy();
    expect(
      screen.getByText(new RegExp(`arrival in ${event.intervals_to_arrival} intervals`)),
    ).toBeTruthy();
    expect(
      screen.getByText(
        new RegExp(`confidence ${event.extraction_confidence_pct}%`),
      ),
    ).toBeTruthy();
    // Each field shows the phrase it came from, per the PRD's extraction schema.
    for (const phrase of Object.values(event.source_phrases)) {
      expect(screen.getAllByText(new RegExp(`“${phrase}”`)).length).toBeGreaterThan(0);
    }
  });

  it("asks for confirmation before a high-impact extraction reaches the optimizer", async () => {
    await typeAndExtract(stormBulletin);
    expect((stormBulletin as { needs_confirmation: boolean }).needs_confirmation).toBe(true);
    expect(screen.getByText("Confirm and apply")).toBeTruthy();
    expect(screen.getByText("Reject")).toBeTruthy();
    expect(
      screen.getByText(/High-impact extractions wait for an operator/),
    ).toBeTruthy();
  });

  it("shows nothing extracted when the bulletin is an instruction", async () => {
    await typeAndExtract(injectionBulletin);
    expect((injectionBulletin as { event: unknown }).event).toBeNull();
    expect(screen.getByText("No event extracted")).toBeTruthy();
    expect(
      screen.getByText(/The text did not describe a grid event/),
    ).toBeTruthy();
    // No confirm button, because there is nothing to confirm.
    expect(screen.queryByText("Confirm and apply")).toBeNull();
  });
});

describe("nothing renders a hole where a value should be", () => {
  /** One symptom of a contract that drifted is `undefined`, `NaN` or `[object Object]`
   *  reaching the screen, which this sweeps the whole rendered tree for.
   *
   *  It is a net, not a proof: React renders a bare `{value}` of `undefined` as nothing
   *  at all, so a missing field used as a JSX child leaves a silent gap rather than a
   *  visible one. Only fields that pass through a template literal — `confidence
   *  ${pct}%` — show up here. The named assertions above are what actually pin the
   *  contract; this catches the ones nobody thought to name. */
  const HOLES = ["undefined", "NaN", "[object Object]", "null MW", "$null"];

  const sweep = (text: string, where: string) => {
    for (const hole of HOLES) {
      expect(text.includes(hole), `${where} rendered "${hole}"`).toBe(false);
    }
  };

  it("holds for every frame the backend produces", () => {
    for (const [name, frame] of [
      ["opening", opening],
      ["after an event", withEvent],
      ["in fallback", fallback],
    ] as const) {
      cleanup();
      const { container } = live(frame);
      sweep(container.textContent ?? "", `Live operations ${name}`);
    }
  });

  it("holds for a bulletin extraction", async () => {
    vi.spyOn(api, "bulletin").mockResolvedValue(stormBulletin as never);
    const { container } = render(
      <EventInjector config={config} frame={opening} busy={false} act={passthrough} />,
    );
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "a bulletin" } });
    fireEvent.click(screen.getByText("Extract event"));
    await screen.findByText(/extracted by/);
    sweep(container.textContent ?? "", "the bulletin panel");
  });

  it("holds for the simulation and reliability tabs", async () => {
    vi.spyOn(api, "simulation").mockResolvedValue(simulationFixture as never);
    const sim = render(<ScenarioSimulation config={config} />);
    await screen.findByText("KPI comparison");
    sweep(sim.container.textContent ?? "", "Scenario simulation");
    cleanup();

    vi.spyOn(api, "reliability").mockResolvedValue(reliabilityFixture as never);
    const rel = render(<ReliabilityTests />);
    await screen.findByText("Invariant suite");
    sweep(rel.container.textContent ?? "", "Reliability tests");
    cleanup();

    const how = render(<HowItDecides config={config} />);
    sweep(how.container.textContent ?? "", "How it decides");
  });
});
