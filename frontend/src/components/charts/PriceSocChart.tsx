import type { Forecast, HistoryRow } from "../../api/types";
import { bandPath, linear, linePath, niceTicks } from "../../lib/scale";
import { Legend } from "../ui";

/** PRD: "A second chart for price and battery SoC with the plan's projected SoC."
 *
 * Two stacked panels sharing one x-axis, not one chart with two y-scales. Price is
 * $/MWh and SoC is %, so a single pair of axes would be a dual-axis chart — the one
 * chart form that is never correct. Small multiples give the same read without it.
 */
const WIDTH = 760;
const PANEL_H = 96;
const PAD = { top: 10, right: 54, bottom: 18, left: 46 };

export function PriceSocChart({
  history,
  forecast,
  intervalMinutes,
  socMaxPct,
  reserveFloorPct,
}: {
  history: HistoryRow[];
  forecast: Forecast;
  intervalMinutes: number;
  socMaxPct: number;
  reserveFloorPct: number;
}) {
  const past = history.slice(-48);
  if (past.length === 0) return <p className="empty">No actuals yet.</p>;

  const steps = forecast.price.p50.length;
  const total = past.length + steps;
  const x = linear([0, Math.max(1, total - 1)], [PAD.left, WIDTH - PAD.right]);
  const futureX = (i: number) => past.length - 1 + i;
  const nowX = x(past.length - 1);

  const prices = [
    ...past.map((r) => r.price_per_mwh),
    ...forecast.price.p10,
    ...forecast.price.p90,
  ];
  const priceTicks = niceTicks(Math.min(0, ...prices), Math.max(...prices, 1), 3);
  const priceY = linear(
    [priceTicks[0] ?? 0, priceTicks[priceTicks.length - 1] ?? 1],
    [PANEL_H - PAD.bottom, PAD.top],
  );
  const socY = linear([0, socMaxPct], [PANEL_H - PAD.bottom, PAD.top]);

  const hourTicks = past
    .map((row, i) => ({ row, i }))
    .filter(({ row }) => (row.interval * intervalMinutes) % 180 === 0);

  return (
    <div className="stack" style={{ gap: 2 }}>
      <svg
        className="chart"
        viewBox={`0 0 ${WIDTH} ${PANEL_H}`}
        role="img"
        aria-label="Spot price, actual and forecast P10 to P90"
      >
        {priceTicks.map((tick) => (
          <g key={tick}>
            <line
              className="grid-line"
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={priceY(tick)}
              y2={priceY(tick)}
            />
            <text x={PAD.left - 6} y={priceY(tick) + 3} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        <text x={6} y={PAD.top + 2} className="label-ink">
          $/MWh
        </text>
        <path
          d={bandPath(forecast.price.p10, forecast.price.p90, x, priceY, futureX)}
          fill="var(--series-1)"
          opacity={0.18}
        />
        <path
          d={linePath(forecast.price.p50, x, priceY, futureX)}
          fill="none"
          stroke="var(--series-1)"
          strokeWidth={2}
          strokeDasharray="4 3"
        />
        <path
          d={linePath(
            past.map((r) => r.price_per_mwh),
            x,
            priceY,
          )}
          fill="none"
          stroke="var(--series-1)"
          strokeWidth={2}
        />
        {priceTicks.includes(0) && (
          <line
            className="axis-line"
            x1={PAD.left}
            x2={WIDTH - PAD.right}
            y1={priceY(0)}
            y2={priceY(0)}
          />
        )}
        <line className="now-line" x1={nowX} x2={nowX} y1={PAD.top} y2={PANEL_H - PAD.bottom} />
      </svg>

      <svg
        className="chart"
        viewBox={`0 0 ${WIDTH} ${PANEL_H}`}
        role="img"
        aria-label="Fleet state of charge, actual and the plan's projection"
      >
        {[0, 25, 50, 75, socMaxPct].map((tick) => (
          <g key={tick}>
            <line
              className="grid-line"
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={socY(tick)}
              y2={socY(tick)}
            />
            <text x={PAD.left - 6} y={socY(tick) + 3} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        <text x={6} y={PAD.top + 2} className="label-ink">
          SoC %
        </text>
        {/* the reserve floor normal dispatch holds above */}
        <line
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={socY(reserveFloorPct)}
          y2={socY(reserveFloorPct)}
          stroke="var(--critical)"
          strokeWidth={1}
          strokeDasharray="2 3"
        />
        <text x={WIDTH - PAD.right + 4} y={socY(reserveFloorPct) + 3} fill="var(--critical)">
          reserve
        </text>
        <path
          d={linePath(forecast.projected_fleet_soc_pct, x, socY, futureX)}
          fill="none"
          stroke="var(--battery)"
          strokeWidth={2}
          strokeDasharray="4 3"
        />
        <path
          d={linePath(
            past.map((r) => r.fleet_soc_pct),
            x,
            socY,
          )}
          fill="none"
          stroke="var(--battery)"
          strokeWidth={2}
        />
        <line className="now-line" x1={nowX} x2={nowX} y1={PAD.top} y2={PANEL_H - PAD.bottom} />
        <line
          className="axis-line"
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={PANEL_H - PAD.bottom}
          y2={PANEL_H - PAD.bottom}
        />
        {hourTicks.map(({ row, i }) => (
          <text key={row.interval} x={x(i)} y={PANEL_H - PAD.bottom + 13} textAnchor="middle">
            {row.time}
          </text>
        ))}
      </svg>

      <Legend
        items={[
          { label: "Spot price", color: "var(--series-1)", line: true },
          { label: "Fleet SoC", color: "var(--battery)", line: true },
          { label: "Forecast / plan (dashed)", color: "var(--text-muted)", line: true },
        ]}
      />
    </div>
  );
}
