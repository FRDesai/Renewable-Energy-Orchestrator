/** Number formatting for a control room.
 *
 * PRD UX rule: "every number carries its unit", and the agent's text uses operator
 * vocabulary. So there is no bare-number formatter here on purpose — every function
 * emits the unit with the figure, and anything non-finite becomes an em dash rather
 * than reaching an operator as `NaN`.
 */

const MINUS = "−"; // U+2212, not a hyphen: a hyphen reads as a dash at small sizes
const DASH = "—";

const isBad = (v: number): boolean => !Number.isFinite(v);

function group(value: number, digits = 0): string {
  return Math.abs(value).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** Drop a trailing ".0" so whole numbers read as whole numbers. */
function trim(value: number, digits: number): string {
  const text = group(value, digits);
  return digits > 0 && text.endsWith("." + "0".repeat(digits))
    ? text.slice(0, -(digits + 1))
    : text;
}

export function mw(value: number, digits = 1): string {
  if (isBad(value)) return DASH;
  return `${value < 0 ? MINUS : ""}${trim(value, digits)} MW`;
}

/** Battery dispatch: the sign is the information, so it is always shown. */
export function signedMw(value: number, digits = 1): string {
  if (isBad(value)) return DASH;
  if (Math.abs(value) < 0.05) return "0 MW";
  return `${value > 0 ? "+" : MINUS}${trim(value, digits)} MW`;
}

export function mwh(value: number, digits = 1): string {
  if (isBad(value)) return DASH;
  return `${value < 0 ? MINUS : ""}${trim(value, digits)} MWh`;
}

export function pct(value: number, digits = 0): string {
  if (isBad(value)) return DASH;
  return `${value < 0 ? MINUS : ""}${trim(value, digits)}%`;
}

export function usd(value: number, digits = 0): string {
  if (isBad(value)) return DASH;
  return `${value < 0 ? MINUS : ""}$${group(value, digits)}`;
}

export function perMwh(value: number): string {
  if (isBad(value)) return DASH;
  return `${value < 0 ? MINUS : ""}$${group(value, 2)}/MWh`;
}

export function perTonne(value: number): string {
  if (isBad(value)) return DASH;
  return `${value < 0 ? MINUS : ""}$${group(value, 2)}/t`;
}

export function hz(value: number): string {
  if (isBad(value)) return DASH;
  return `${group(value, 2)} Hz`;
}

export function ms(value: number): string {
  if (isBad(value)) return DASH;
  return `${group(value, 0)} ms`;
}

export function tonnes(value: number): string {
  if (isBad(value)) return DASH;
  return `${value < 0 ? MINUS : ""}${group(value, 0)} t CO₂`;
}

export function metresPerSecond(value: number): string {
  if (isBad(value)) return DASH;
  return `${group(value, 1)} m/s`;
}

/** A change, with its direction stated rather than implied by colour alone. */
export function deltaPct(value: number, digits = 1): string {
  if (isBad(value)) return DASH;
  if (Math.abs(value) < 0.05) return `${group(0, digits)}%`;
  return `${value > 0 ? "+" : MINUS}${group(value, digits)}%`;
}

export function count(value: number, singular: string, plural = `${singular}s`): string {
  if (isBad(value)) return DASH;
  return `${group(value, 0)} ${value === 1 ? singular : plural}`;
}

/** Interval index to a clock reading on the simulated 24-hour day. */
export function clockOf(interval: number, intervalMinutes: number): string {
  const minutes = ((interval * intervalMinutes) % 1440 + 1440) % 1440;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

/** Intervals to a duration an operator reads as hours and minutes. */
export function intervalsToDuration(intervals: number, intervalMinutes: number): string {
  if (isBad(intervals)) return DASH;
  const minutes = Math.round(intervals * intervalMinutes);
  const h = Math.floor(Math.abs(minutes) / 60);
  const m = Math.abs(minutes) % 60;
  const sign = minutes < 0 ? MINUS : "";
  if (h === 0) return `${sign}${m} min`;
  if (m === 0) return `${sign}${h} h`;
  return `${sign}${h} h ${m} min`;
}

/** Objective keys as an operator says them, with the unit the engine reports. */
export const OBJECTIVE_LABEL: Record<string, string> = {
  net_cost: "Net cost",
  reliability_risk: "Reliability risk",
  carbon: "Carbon",
  curtailment: "Curtailment",
  grid_stability: "Grid stability",
  battery_wear: "Battery wear",
};

export function objective(key: string, value: number): string {
  switch (key) {
    case "net_cost":
    case "battery_wear":
      return usd(value);
    case "carbon":
      return tonnes(value);
    case "reliability_risk":
      return `${trim(value, 1)} MWh-eq`;
    case "curtailment":
      return mwh(value);
    default:
      return isBad(value) ? DASH : `${trim(value, 1)} index`;
  }
}
