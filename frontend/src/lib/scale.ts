/** The small amount of chart geometry the console needs.
 *
 * Hand-rolled rather than a charting library: the four tabs need a stacked area with a
 * forecast band, a scatter, a fan and two bar forms, all of which are a few dozen lines
 * of SVG each, and all of which need exact control over the mark specs (2px surface gaps
 * between stacked segments, rounded data-ends, a crosshair that snaps to an interval).
 */

export type Scale = (value: number) => number;

/** A linear scale. Degenerate domains map to the middle of the range rather than NaN. */
export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  if (span === 0) {
    const mid = (r0 + r1) / 2;
    return () => mid;
  }
  const k = (r1 - r0) / span;
  return (value: number) => r0 + (value - d0) * k;
}

/** Round tick values covering [lo, hi], including zero whenever the domain spans it. */
export function niceTicks(lo: number, hi: number, target = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0];
  if (lo === hi) {
    const pad = Math.abs(lo) > 0 ? Math.abs(lo) / 2 : 1;
    lo -= pad;
    hi += pad;
  }
  const raw = (hi - lo) / Math.max(1, target);
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const normalised = raw / magnitude;
  const step =
    (normalised >= 5 ? 10 : normalised >= 2 ? 5 : normalised >= 1 ? 2 : 1) * magnitude;
  const start = Math.floor(lo / step) * step;
  const end = Math.ceil(hi / step) * step;
  const out: number[] = [];
  // Accumulate by index rather than by repeated addition, so float drift cannot
  // push a tick off its round value.
  for (let i = 0; start + i * step <= end + step * 1e-9; i += 1) {
    const value = start + i * step;
    out.push(Math.abs(value) < step * 1e-9 ? 0 : value);
    if (out.length > 64) break;
  }
  return out;
}

/** Turn series into stacked [y0, y1] pairs per point. A missing point counts as zero. */
export function stack(series: number[][]): [number, number][][] {
  const points = series.reduce((n, s) => Math.max(n, s.length), 0);
  const base = new Array<number>(points).fill(0);
  return series.map((values) =>
    base.map((offset, i) => {
      const value = values[i] ?? 0;
      const pair: [number, number] = [offset, offset + value];
      base[i] = offset + value;
      return pair;
    }),
  );
}

/** Polyline through the points, breaking the path where a value is null. */
export function linePath(
  values: (number | null)[],
  x: Scale,
  y: Scale,
  xOf: (i: number) => number = (i) => i,
): string {
  let d = "";
  let pendingMove = true;
  values.forEach((value, i) => {
    if (value === null || !Number.isFinite(value)) {
      pendingMove = true;
      return;
    }
    const px = round(x(xOf(i)));
    const py = round(y(value));
    d += `${pendingMove ? "M" : "L"}${px},${py}`;
    pendingMove = false;
  });
  return d;
}

/** Closed area between an upper and a lower edge — the P10–P90 band. */
export function bandPath(
  lower: number[],
  upper: number[],
  x: Scale,
  y: Scale,
  xOf: (i: number) => number = (i) => i,
): string {
  const n = Math.min(lower.length, upper.length);
  if (n === 0) return "";
  let d = "";
  for (let i = 0; i < n; i += 1) {
    const value = upper[i];
    if (value === undefined) continue;
    d += `${i === 0 ? "M" : "L"}${round(x(xOf(i)))},${round(y(value))}`;
  }
  for (let i = n - 1; i >= 0; i -= 1) {
    const value = lower[i];
    if (value === undefined) continue;
    d += `L${round(x(xOf(i)))},${round(y(value))}`;
  }
  return d + "Z";
}

/** Closed area between a series and a baseline value. */
export function areaPath(
  values: number[],
  x: Scale,
  y: Scale,
  baseline = 0,
  xOf: (i: number) => number = (i) => i,
): string {
  if (values.length === 0) return "";
  return bandPath(
    new Array<number>(values.length).fill(baseline),
    values,
    x,
    y,
    xOf,
  );
}

export function extent(values: number[], fallback: [number, number] = [0, 1]):
  [number, number] {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return fallback;
  return [Math.min(...finite), Math.max(...finite)];
}

/** Half-pixel rounding keeps hairlines crisp without blurring fills. */
function round(value: number): number {
  return Math.round(value * 100) / 100;
}

export { round as roundPx };
