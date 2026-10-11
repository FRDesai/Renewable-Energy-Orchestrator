import { useState } from "react";
import { linear, niceTicks, stack } from "../../lib/scale";
import { pct } from "../../lib/units";
import { Legend } from "../ui";

/** PRD F3.4: "the agent's strategy mix by hour." Stacked shares, so the stack
 *  always sums to one and the question "what was it doing at 18:00" is one column.
 *
 *  Five battery strategies use the first five categorical slots in their documented
 *  order, which is validated on the adjacent pairlist — the right pairlist for a stack,
 *  where only neighbouring segments touch. Segments carry a 2px surface gap.
 */
const SLOTS = [
  "var(--series-1)",
  "var(--series-2)",
  "var(--series-3)",
  "var(--series-4)",
  "var(--series-5)",
];

const WIDTH = 560;
const HEIGHT = 190;
const PAD = { top: 10, right: 14, bottom: 22, left: 36 };

export function StrategyMixBars({ rows }: { rows: Record<string, number>[] }) {
  const [hover, setHover] = useState<number | null>(null);
  if (rows.length === 0) return <p className="empty">No strategy mix recorded.</p>;

  const keys = Array.from(
    new Set(rows.flatMap((row) => Object.keys(row).filter((k) => k !== "hour"))),
  ).sort();
  const layers = stack(keys.map((key) => rows.map((row) => (row[key] ?? 0) * 100)));
  const x = linear([0, rows.length], [PAD.left, WIDTH - PAD.right]);
  const y = linear([0, 100], [HEIGHT - PAD.bottom, PAD.top]);
  const bandWidth = Math.max(2, (WIDTH - PAD.right - PAD.left) / rows.length - 2);

  return (
    <div className="chart-wrap">
      <svg
        className="chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Share of intervals by battery strategy, per hour"
        onPointerLeave={() => setHover(null)}
      >
        {[0, 25, 50, 75, 100].map((tick) => (
          <g key={tick}>
            <line
              className="grid-line"
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={y(tick)}
              y2={y(tick)}
            />
            <text x={PAD.left - 6} y={y(tick) + 3} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        <text x={2} y={PAD.top + 2} className="label-ink">
          %
        </text>

        {keys.map((key, ki) =>
          (layers[ki] ?? []).map(([y0, y1], ri) => {
            const height = y(y0) - y(y1);
            if (height <= 0.2) return null;
            return (
              <rect
                key={`${key}-${ri}`}
                x={x(ri) + 1}
                y={y(y1)}
                width={bandWidth}
                height={height}
                fill={SLOTS[ki % SLOTS.length]}
                stroke="var(--surface-1)"
                strokeWidth={2}
                rx={2}
                onPointerEnter={() => setHover(ri)}
              />
            );
          }),
        )}

        <line
          className="axis-line"
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={HEIGHT - PAD.bottom}
          y2={HEIGHT - PAD.bottom}
        />
        {rows.map((row, i) =>
          i % 3 === 0 ? (
            <text
              key={i}
              x={x(i) + bandWidth / 2}
              y={HEIGHT - PAD.bottom + 13}
              textAnchor="middle"
            >
              {String(row["hour"] ?? i).padStart(2, "0")}
            </text>
          ) : null,
        )}
      </svg>

      {hover !== null && rows[hover] && (
        <div className="tooltip" style={{ right: 8, top: 8 }}>
          <div className="t-title">
            Hour {String(rows[hover]["hour"] ?? hover).padStart(2, "0")}:00
          </div>
          {keys.map((key, ki) => {
            const share = (rows[hover]?.[key] ?? 0) * 100;
            if (share < 0.5) return null;
            return (
              <div className="t-row" key={key}>
                <span>
                  <i style={{ background: SLOTS[ki % SLOTS.length] }} />
                  {key}
                </span>
                <span>{pct(share)}</span>
              </div>
            );
          })}
        </div>
      )}

      <Legend
        items={keys.map((key, ki) => ({ label: key, color: SLOTS[ki % SLOTS.length]! }))}
      />
    </div>
  );
}

/** A latency histogram: one series, so no legend — the title names it. */
export function Histogram({
  values,
  unit,
  budget,
}: {
  values: number[];
  unit: string;
  budget?: number;
}) {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return <p className="empty">No measurements.</p>;

  const lo = 0;
  const hi = Math.max(...finite, budget ?? 0) * 1.05;
  const bins = 24;
  const counts = new Array<number>(bins).fill(0);
  for (const v of finite) {
    const idx = Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo || 1)) * bins));
    counts[idx] = (counts[idx] ?? 0) + 1;
  }
  const WIDTH_H = 520;
  const HEIGHT_H = 160;
  const P = { top: 10, right: 14, bottom: 24, left: 34 };
  const yTicks = niceTicks(0, Math.max(...counts), 3);
  const x = linear([lo, hi], [P.left, WIDTH_H - P.right]);
  const y = linear(
    [0, yTicks[yTicks.length - 1] ?? 1],
    [HEIGHT_H - P.bottom, P.top],
  );
  const barWidth = (WIDTH_H - P.right - P.left) / bins - 2;

  return (
    <div>
      <svg
        className="chart"
        viewBox={`0 0 ${WIDTH_H} ${HEIGHT_H}`}
        role="img"
        aria-label={`Distribution of decision latency in ${unit}`}
      >
        {yTicks.map((tick) => (
          <g key={tick}>
            <line
              className="grid-line"
              x1={P.left}
              x2={WIDTH_H - P.right}
              y1={y(tick)}
              y2={y(tick)}
            />
            <text x={P.left - 6} y={y(tick) + 3} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        {counts.map((c, i) => {
          const x0 = x(lo + ((hi - lo) * i) / bins);
          const height = y(0) - y(c);
          if (height <= 0) return null;
          return (
            <rect
              key={i}
              x={x0 + 1}
              y={y(c)}
              width={Math.max(1, barWidth)}
              height={height}
              rx={2}
              fill="var(--series-1)"
            />
          );
        })}
        {budget !== undefined && budget <= hi && (
          <>
            <line
              x1={x(budget)}
              x2={x(budget)}
              y1={P.top}
              y2={HEIGHT_H - P.bottom}
              stroke="var(--critical)"
              strokeWidth={1.5}
              strokeDasharray="3 3"
            />
            <text x={x(budget) - 4} y={P.top + 9} textAnchor="end" fill="var(--critical)">
              budget
            </text>
          </>
        )}
        <line
          className="axis-line"
          x1={P.left}
          x2={WIDTH_H - P.right}
          y1={HEIGHT_H - P.bottom}
          y2={HEIGHT_H - P.bottom}
        />
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <text
            key={f}
            x={x(lo + (hi - lo) * f)}
            y={HEIGHT_H - P.bottom + 13}
            textAnchor="middle"
          >
            {(lo + (hi - lo) * f).toFixed(0)}
          </text>
        ))}
        <text
          x={(WIDTH_H + P.left) / 2}
          y={HEIGHT_H - 2}
          textAnchor="middle"
          className="label-ink"
        >
          {unit}
        </text>
      </svg>
    </div>
  );
}
