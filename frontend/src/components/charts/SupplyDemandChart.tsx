import { useMemo, useState } from "react";
import type { Forecast, HistoryRow } from "../../api/types";
import { bandPath, linear, linePath, niceTicks, stack } from "../../lib/scale";
import { mw, perMwh } from "../../lib/units";
import { Legend } from "../ui";

/** PRD: "last 12 h actuals stacked by source, next 4 h as a P10–P90 band, a 'now' line."
 *
 * The stack order is solar → wind → battery discharge → import, coloured with the
 * yellow/aqua/violet/orange ordering that was run through the palette validator in both
 * light and dark before this was drawn. Segments carry a 2px surface gap so adjacent
 * fills never touch.
 */
const LAYERS = [
  { key: "solar_mw", label: "Solar", color: "var(--solar)" },
  { key: "wind_mw", label: "Wind", color: "var(--wind)" },
  { key: "battery_discharge_mw", label: "Battery discharge", color: "var(--battery)" },
  { key: "import_mw", label: "Import", color: "var(--import)" },
] as const;

const PAD = { top: 10, right: 54, bottom: 22, left: 46 };
const WIDTH = 760;
const HEIGHT = 230;

export function SupplyDemandChart({
  history,
  forecast,
  intervalMinutes,
  nowInterval,
}: {
  history: HistoryRow[];
  forecast: Forecast;
  intervalMinutes: number;
  nowInterval: number;
}) {
  const [hover, setHover] = useState<number | null>(null);

  const model = useMemo(() => {
    const past = history.slice(-48);
    const steps = forecast.demand.p50.length;
    // x is an interval offset from the first history row, so past and future share one
    // axis without a second scale.
    const firstInterval = past[0]?.interval ?? nowInterval;
    const total = past.length + steps;
    const layers = stack(LAYERS.map((layer) => past.map((row) => row[layer.key])));
    const supplyTop = layers[layers.length - 1]?.map((pair) => pair[1]) ?? [];
    const demand = past.map((row) => row.demand_mw);
    const hi = Math.max(
      ...supplyTop,
      ...demand,
      ...forecast.demand.p90,
      ...forecast.solar.p90.map((v, i) => v + (forecast.wind.p90[i] ?? 0)),
      1,
    );
    return { past, steps, firstInterval, total, layers, supplyTop, demand, hi };
  }, [history, forecast, nowInterval]);

  const { past, steps, firstInterval, total, layers, supplyTop, demand, hi } = model;
  if (past.length === 0) return <p className="empty">No actuals yet.</p>;

  const x = linear([0, Math.max(1, total - 1)], [PAD.left, WIDTH - PAD.right]);
  const ticks = niceTicks(0, hi, 4);
  const y = linear([0, ticks[ticks.length - 1] ?? hi], [HEIGHT - PAD.bottom, PAD.top]);
  const nowX = x(past.length - 1);

  // The forecast band is indexed from "now", so step 0 sits on the now line.
  const futureX = (i: number) => past.length - 1 + i;
  const renewableP10 = forecast.solar.p10.map((v, i) => v + (forecast.wind.p10[i] ?? 0));
  const renewableP90 = forecast.solar.p90.map((v, i) => v + (forecast.wind.p90[i] ?? 0));

  const hourTicks = past
    .map((row, i) => ({ row, i }))
    .filter(({ row }) => (row.interval * intervalMinutes) % 180 === 0);

  const hovered = hover !== null ? past[hover] : undefined;

  return (
    <div className="chart-wrap">
      <svg
        className="chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Supply by source against demand, with the next four hours as a P10 to P90 band"
        onPointerMove={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          const px = ((event.clientX - box.left) / box.width) * WIDTH;
          const idx = Math.round(
            ((px - PAD.left) / (WIDTH - PAD.right - PAD.left)) * (total - 1),
          );
          setHover(idx >= 0 && idx < past.length ? idx : null);
        }}
        onPointerLeave={() => setHover(null)}
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
              {tick}
            </text>
          </g>
        ))}
        <text x={6} y={PAD.top + 2} className="label-ink">
          MW
        </text>

        {/* actuals, stacked */}
        {LAYERS.map((layer, li) => {
          const pairs = layers[li];
          if (!pairs) return null;
          return (
            <path
              key={layer.key}
              d={bandPath(
                pairs.map((p) => p[0]),
                pairs.map((p) => p[1]),
                x,
                y,
              )}
              fill={layer.color}
              stroke="var(--surface-1)"
              strokeWidth={2}
              strokeLinejoin="round"
            />
          );
        })}

        {/* forecast: renewable availability band and the demand band */}
        <path
          d={bandPath(renewableP10, renewableP90, x, y, futureX)}
          fill="var(--wind)"
          opacity={0.2}
        />
        <path
          d={bandPath(forecast.demand.p10, forecast.demand.p90, x, y, futureX)}
          fill="var(--demand)"
          opacity={0.1}
        />
        <path
          d={linePath(forecast.demand.p50, x, y, futureX)}
          fill="none"
          stroke="var(--demand)"
          strokeWidth={2}
          strokeDasharray="4 3"
        />

        {/* demand actual, over the stack */}
        <path
          d={linePath(demand, x, y)}
          fill="none"
          stroke="var(--demand)"
          strokeWidth={2}
        />

        <line
          className="now-line"
          x1={nowX}
          x2={nowX}
          y1={PAD.top}
          y2={HEIGHT - PAD.bottom}
        />
        <text x={nowX + 4} y={PAD.top + 9} className="label-ink">
          now
        </text>
        <text x={WIDTH - PAD.right + 4} y={PAD.top + 9}>
          +{forecast.horizon_hours} h
        </text>

        <line
          className="axis-line"
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={HEIGHT - PAD.bottom}
          y2={HEIGHT - PAD.bottom}
        />
        {hourTicks.map(({ row, i }) => (
          <text key={row.interval} x={x(i)} y={HEIGHT - PAD.bottom + 13} textAnchor="middle">
            {row.time}
          </text>
        ))}

        {hover !== null && past[hover] && (
          <>
            <line
              className="axis-line"
              x1={x(hover)}
              x2={x(hover)}
              y1={PAD.top}
              y2={HEIGHT - PAD.bottom}
            />
            <circle
              cx={x(hover)}
              cy={y(supplyTop[hover] ?? 0)}
              r={4}
              fill="var(--text-primary)"
              stroke="var(--surface-1)"
              strokeWidth={2}
            />
          </>
        )}
      </svg>

      {hovered && (
        <div
          className="tooltip"
          style={{
            left: `calc(${((x(hover ?? 0) / WIDTH) * 100).toFixed(2)}% + 10px)`,
            top: 6,
          }}
        >
          <div className="t-title">
            {hovered.time} · interval {hovered.interval}
          </div>
          {LAYERS.map((layer) => (
            <div className="t-row" key={layer.key}>
              <span>
                <i style={{ background: layer.color }} />
                {layer.label}
              </span>
              <span>{mw(hovered[layer.key])}</span>
            </div>
          ))}
          <div className="t-row">
            <span>Demand</span>
            <span>{mw(hovered.demand_mw)}</span>
          </div>
          {hovered.export_mw > 0.05 && (
            <div className="t-row">
              <span>Export</span>
              <span>{mw(hovered.export_mw)}</span>
            </div>
          )}
          {hovered.curtailed_mw > 0.05 && (
            <div className="t-row">
              <span>Curtailed</span>
              <span>{mw(hovered.curtailed_mw)}</span>
            </div>
          )}
          <div className="t-row">
            <span>Price</span>
            <span>{perMwh(hovered.price_per_mwh)}</span>
          </div>
        </div>
      )}

      <Legend
        items={[
          ...LAYERS.map((l) => ({ label: l.label, color: l.color })),
          { label: "Demand", color: "var(--demand)", line: true },
          { label: "Forecast P10–P90", color: "var(--wind)" },
        ]}
      />
      <p className="tiny muted" style={{ marginTop: 4 }}>
        Actuals to the left of the now line are what the plant did; to the right is the
        P10–P90 band from the {forecast.demand.source === "forecaster" ? "quantile forecasters" : "scenario fan"} over{" "}
        {steps} steps. Interval {firstInterval} onward.
      </p>
    </div>
  );
}
