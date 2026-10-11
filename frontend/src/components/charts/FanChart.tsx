import type { Fan } from "../../api/types";
import { bandPath, extent, linear, linePath, niceTicks } from "../../lib/scale";
import { Legend } from "../ui";

/** PRD F3.4: "Visualize cumulative cost and battery SoC as fan charts."
 *
 * One band and one median line per strategy, in the three slots that validate
 * all-pairs. Three is also the cap: past three series a fan chart becomes unreadable
 * regardless of palette.
 */
const WIDTH = 560;
const HEIGHT = 200;
const PAD = { top: 12, right: 14, bottom: 22, left: 58 };

export const STRATEGY_COLOR: Record<string, string> = {
  agent: "var(--series-1)",
  rule_based: "var(--series-2)",
  no_storage: "var(--series-3)",
};

export const STRATEGY_LABEL: Record<string, string> = {
  agent: "Agent",
  rule_based: "Rule-based",
  no_storage: "No storage",
};

export function FanChart({
  series,
  unitLabel,
  tickFormat = (v) => String(Math.round(v)),
  intervalMinutes,
}: {
  series: { strategy: string; fan: Fan }[];
  unitLabel: string;
  tickFormat?: (value: number) => string;
  intervalMinutes: number;
}) {
  const present = series.filter((s) => s.fan.p50.length > 0);
  if (present.length === 0) return <p className="empty">No paths to show.</p>;

  const steps = Math.max(...present.map((s) => s.fan.p50.length));
  const all = present.flatMap((s) => [...s.fan.p10, ...s.fan.p90]);
  const [lo, hi] = extent(all, [0, 1]);
  const ticks = niceTicks(Math.min(0, lo), hi, 4);
  const x = linear([0, Math.max(1, steps - 1)], [PAD.left, WIDTH - PAD.right]);
  const y = linear(
    [ticks[0] ?? lo, ticks[ticks.length - 1] ?? hi],
    [HEIGHT - PAD.bottom, PAD.top],
  );

  const perHour = Math.round(60 / intervalMinutes);
  const hourTicks: number[] = [];
  for (let i = 0; i < steps; i += perHour * 3) hourTicks.push(i);

  return (
    <div>
      <svg
        className="chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${unitLabel} across paths, P10 to P90 band with the median per strategy`}
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              className="grid-line"
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={y(tick)}
              y2={y(tick)}
            />
            <text x={PAD.left - 6} y={y(tick) + 3} textAnchor="end">
              {tickFormat(tick)}
            </text>
          </g>
        ))}
        <text x={4} y={PAD.top + 2} className="label-ink">
          {unitLabel}
        </text>

        {present.map(({ strategy, fan }) => (
          <path
            key={`band-${strategy}`}
            d={bandPath(fan.p10, fan.p90, x, y)}
            fill={STRATEGY_COLOR[strategy] ?? "var(--series-1)"}
            opacity={0.17}
          />
        ))}
        {present.map(({ strategy, fan }) => (
          <path
            key={`line-${strategy}`}
            d={linePath(fan.p50, x, y)}
            fill="none"
            stroke={STRATEGY_COLOR[strategy] ?? "var(--series-1)"}
            strokeWidth={2}
          />
        ))}

        <line
          className="axis-line"
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={HEIGHT - PAD.bottom}
          y2={HEIGHT - PAD.bottom}
        />
        {hourTicks.map((i) => (
          <text key={i} x={x(i)} y={HEIGHT - PAD.bottom + 13} textAnchor="middle">
            {String(Math.floor((i * intervalMinutes) / 60)).padStart(2, "0")}:00
          </text>
        ))}
      </svg>
      <Legend
        items={present.map((s) => ({
          label: STRATEGY_LABEL[s.strategy] ?? s.strategy,
          color: STRATEGY_COLOR[s.strategy] ?? "var(--series-1)",
          line: true,
        }))}
      />
    </div>
  );
}
