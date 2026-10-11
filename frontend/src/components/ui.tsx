import type { ReactNode } from "react";
import type { LimitStatus, Severity } from "../api/types";

/** PRD UX rule: "state shown in form as well as numbers (status chips, severity
 *  colors)". Every chip carries a label, so colour is never the only signal. */
export type Tone = "good" | "warning" | "serious" | "critical" | "neutral";

export function Chip({
  tone = "neutral",
  children,
  title,
}: {
  tone?: Tone;
  children: ReactNode;
  title?: string;
}) {
  return (
    <span className={`chip ${tone}`} title={title}>
      <span className="dot" aria-hidden="true" />
      {children}
    </span>
  );
}

const ASSET_TONE: Record<string, Tone> = {
  online: "good",
  discharging: "good",
  charging: "good",
  protected: "good",
  normal: "neutral",
  holding: "neutral",
  idle: "neutral",
  reducing: "warning",
  shifted: "warning",
  curtailed: "warning",
  maintenance: "serious",
  tripped: "critical",
};

export const assetTone = (status: string): Tone => ASSET_TONE[status] ?? "neutral";

const ASSET_LABEL: Record<string, string> = {
  online: "Online",
  idle: "Idle",
  curtailed: "Curtailed",
  maintenance: "Maintenance",
  tripped: "Tripped",
  discharging: "Discharging",
  charging: "Charging",
  holding: "Holding",
  protected: "Protected",
  normal: "Normal",
  reducing: "Reducing",
  shifted: "Shifted",
};

export const assetLabel = (status: string): string =>
  ASSET_LABEL[status] ?? status.replace(/_/g, " ");

const LIMIT_TONE: Record<LimitStatus, Tone> = {
  at_limit: "critical",
  near_limit: "warning",
  normal: "good",
  idle: "neutral",
};

const LIMIT_LABEL: Record<LimitStatus, string> = {
  at_limit: "At limit",
  near_limit: "Near limit",
  normal: "Within limit",
  idle: "Idle",
};

export function LimitChip({ status }: { status: LimitStatus }) {
  return <Chip tone={LIMIT_TONE[status]}>{LIMIT_LABEL[status]}</Chip>;
}

export const severityTone = (severity: Severity): Tone =>
  severity === "critical" ? "critical" : severity === "warning" ? "warning" : "neutral";

/** A bar with an optional limit marker, so "how close to the cap" is visible in form. */
export function Meter({
  value,
  max,
  limit,
  color = "var(--series-1)",
}: {
  value: number;
  max: number;
  limit?: number;
  color?: string;
}) {
  const safeMax = max > 0 ? max : 1;
  const width = Math.max(0, Math.min(100, (value / safeMax) * 100));
  const limitAt =
    limit !== undefined && limit > 0 && limit < safeMax ? (limit / safeMax) * 100 : null;
  return (
    <div className="meter" role="img" aria-label={`${width.toFixed(0)} percent of scale`}>
      <span style={{ width: `${width}%`, background: color }} />
      {limitAt !== null && <span className="limit" style={{ left: `${limitAt}%` }} />}
    </div>
  );
}

export function Stat({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: Tone;
}) {
  const color =
    tone === "critical"
      ? "var(--critical)"
      : tone === "warning"
        ? "var(--text-primary)"
        : tone === "good"
          ? "var(--success-text)"
          : undefined;
  return (
    <div className="stat">
      <span className="label">{label}</span>
      <span className="value" style={color ? { color } : undefined}>
        {value}
      </span>
      {sub !== undefined && <span className="sub">{sub}</span>}
    </div>
  );
}

export function Panel({
  title,
  hint,
  actions,
  children,
  flush,
  scroll,
}: {
  title: string;
  hint?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  flush?: boolean;
  scroll?: boolean;
}) {
  return (
    <section className="panel">
      <header>
        <h3>
          {title}
          {hint !== undefined && <span className="hint"> · {hint}</span>}
        </h3>
        {actions}
      </header>
      <div className={`body${flush ? " flush" : ""}${scroll ? " scroll" : ""}`}>
        {children}
      </div>
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="empty">{children}</p>;
}

/** A legend is always present for two or more series, so identity is never
 *  colour-alone. A single series needs none — the title names it. */
export function Legend({
  items,
}: {
  items: { label: string; color: string; line?: boolean }[];
}) {
  if (items.length < 2) return null;
  return (
    <div className="legend">
      {items.map((item) => (
        <span className="key" key={item.label}>
          <i
            className={item.line ? "line" : undefined}
            style={{ background: item.color }}
            aria-hidden="true"
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}
