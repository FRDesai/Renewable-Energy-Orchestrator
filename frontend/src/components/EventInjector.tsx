import { useState } from "react";
import { api } from "../api/client";
import type { BulletinResponse, ConfigResponse, Frame } from "../api/types";
import { Chip, Panel } from "./ui";

const STORM_EXAMPLE =
  "SEVERE WEATHER ADVISORY: a severe storm front will reach the service territory in " +
  "about 4 hours, lasting for 6 hours. High winds are forecast across region_b.";

/** PRD: "Event injector: one button per situation ... plus a text box that parses a
 *  pasted bulletin." The button sends only the type — the backend owns the defaults. */
export function EventInjector({
  config,
  frame,
  act,
  busy,
}: {
  config: ConfigResponse | null;
  frame: Frame;
  act: <T>(fn: () => Promise<T>, pick?: (r: T) => Frame | undefined) => Promise<T | null>;
  busy: boolean;
}) {
  const [text, setText] = useState("");
  const [extraction, setExtraction] = useState<BulletinResponse | null>(null);

  const injections = Object.entries(config?.injections ?? {});
  const pending = extraction?.needs_confirmation ? extraction : null;

  const submit = async () => {
    const result = await act(
      () => api.bulletin(text),
      (r) => r.frame,
    );
    if (result) setExtraction(result);
  };

  const confirm = async (accept: boolean) => {
    await act(
      () => api.confirmBulletin(accept),
      (r) => r.frame,
    );
    setExtraction(null);
    if (accept) setText("");
  };

  return (
    <div className="stack">
      <Panel
        title="Event injector"
        hint="re-plans the current interval, within one evaluation"
      >
        <div className="grid-buttons">
          {injections.map(([type, meta]) => (
            <button
              key={type}
              className="btn small"
              disabled={busy}
              onClick={() => void act(() => api.inject({ type }))}
              title={`Inject ${meta.label}`}
            >
              {meta.label}
            </button>
          ))}
        </div>
        {frame.events.length > 0 && (
          <div className="legend" style={{ marginTop: 10 }}>
            {frame.events.map((e, i) => (
              <Chip
                key={i}
                tone={
                  e.severity === "severe"
                    ? "critical"
                    : e.severity === "high"
                      ? "serious"
                      : "warning"
                }
              >
                {e.label} active
              </Chip>
            ))}
          </div>
        )}
      </Panel>

      <Panel title="Paste an operational bulletin" hint="text is data, never instructions">
        <textarea
          rows={3}
          value={text}
          placeholder={STORM_EXAMPLE}
          onChange={(event) => setText(event.target.value)}
        />
        <div className="row" style={{ marginTop: 8, justifyContent: "flex-start" }}>
          <button
            className="btn"
            disabled={busy || text.trim().length === 0}
            onClick={() => void submit()}
          >
            Extract event
          </button>
          <button
            className="btn small"
            disabled={busy}
            onClick={() => setText(STORM_EXAMPLE)}
          >
            Use the example storm
          </button>
        </div>

        {extraction && (
          <div
            className={`banner ${pending ? "warning" : extraction.event ? "info" : "critical"}`}
            style={{ marginTop: 10 }}
          >
            <div className="grow">
              {extraction.event ? (
                <>
                  <strong>
                    {extraction.event.label} extracted by the {extraction.source} parser
                  </strong>
                  <div className="small secondary">
                    {extraction.event.region ?? extraction.event.corridor ?? "territory-wide"}
                    {extraction.event.intervals_to_arrival !== null &&
                      ` · arrival in ${extraction.event.intervals_to_arrival} intervals`}
                    {` · confidence ${extraction.event.extraction_confidence_pct}%`}
                  </div>
                  {Object.keys(extraction.event.source_phrases).length > 0 && (
                    <div className="tiny muted" style={{ marginTop: 4 }}>
                      {Object.entries(extraction.event.source_phrases).map(([field, phrase]) => (
                        <div key={field}>
                          {field}: “{phrase}”
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <>
                  <strong>No event extracted</strong>
                  <div className="small secondary">
                    The text did not describe a grid event, so nothing reached the
                    optimizer.
                  </div>
                </>
              )}
              {extraction.issues.length > 0 && (
                <ul className="tiny muted" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
                  {extraction.issues.map((issue, i) => (
                    <li key={i}>{issue}</li>
                  ))}
                </ul>
              )}
              {pending && (
                <div className="row" style={{ marginTop: 8, justifyContent: "flex-start" }}>
                  <button
                    className="btn primary small"
                    disabled={busy}
                    onClick={() => void confirm(true)}
                  >
                    Confirm and apply
                  </button>
                  <button
                    className="btn small"
                    disabled={busy}
                    onClick={() => void confirm(false)}
                  >
                    Reject
                  </button>
                  <span className="tiny muted">
                    High-impact extractions wait for an operator before the optimizer
                    sees them.
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </Panel>
    </div>
  );
}
