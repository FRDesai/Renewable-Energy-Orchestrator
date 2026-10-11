import { useState } from "react";
import type { ScatterPoint } from "../../api/types";
import { extent, linear, niceTicks } from "../../lib/scale";
import { usd } from "../../lib/units";
import { Legend } from "../ui";

/** PRD F2.5: "Show the Pareto front and the runner-up with per-objective deltas."
 *
 * All 80 clusters, cost against reliability risk. A scatter needs all-pairs colour
 * separation rather than adjacent, so it is capped at the three slots that validate
 * all-pairs in both modes — blue, orange, aqua — with guardrail removals in muted ink
 * and a cross glyph, so the category survives without colour.
 */
const WIDTH = 520;
const HEIGHT = 230;
const PAD = { top: 12, right: 14, bottom: 34, left: 56 };

export function ParetoScatter({
  points,
  chosenIndex,
}: {
  points: ScatterPoint[];
  chosenIndex: number;
}) {
  const [hover, setHover] = useState<ScatterPoint | null>(null);
  if (points.length === 0) return <p className="empty">No clusters scored.</p>;

  const [cx0, cx1] = extent(points.map((p) => p.net_cost_usd));
  const [ry0, ry1] = extent(points.map((p) => p.reliability_risk_mwh));
  const xTicks = niceTicks(cx0, cx1, 4);
  const yTicks = niceTicks(Math.min(0, ry0), ry1, 3);
  const x = linear(
    [xTicks[0] ?? cx0, xTicks[xTicks.length - 1] ?? cx1],
    [PAD.left, WIDTH - PAD.right],
  );
  const y = linear(
    [yTicks[0] ?? ry0, yTicks[yTicks.length - 1] ?? ry1],
    [HEIGHT - PAD.bottom, PAD.top],
  );

  const removed = points.filter((p) => !p.survived_guardrails);
  const plain = points.filter((p) => p.survived_guardrails && !p.on_pareto_front);
  const front = points.filter(
    (p) => p.survived_guardrails && p.on_pareto_front && p.index !== chosenIndex,
  );
  const chosen = points.find((p) => p.index === chosenIndex);

  return (
    <div className="chart-wrap">
      <svg
        className="chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Clusters plotted by risk-adjusted net cost against reliability risk"
        onPointerLeave={() => setHover(null)}
      >
        {yTicks.map((tick) => (
          <g key={`y${tick}`}>
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
        {xTicks.map((tick) => (
          <text
            key={`x${tick}`}
            x={x(tick)}
            y={HEIGHT - PAD.bottom + 13}
            textAnchor="middle"
          >
            {Math.abs(tick) >= 1000 ? `${Math.round(tick / 1000)}k` : tick}
          </text>
        ))}
        <text x={4} y={PAD.top + 2} className="label-ink">
          MWh-eq risk
        </text>
        <text
          x={(WIDTH + PAD.left) / 2}
          y={HEIGHT - 4}
          textAnchor="middle"
          className="label-ink"
        >
          risk-adjusted net cost ($)
        </text>
        <line
          className="axis-line"
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={HEIGHT - PAD.bottom}
          y2={HEIGHT - PAD.bottom}
        />

        {/* guardrail removals: muted, cross glyph — readable without colour */}
        {removed.map((p) => (
          <g
            key={p.index}
            onPointerEnter={() => setHover(p)}
            style={{ cursor: "pointer" }}
          >
            <path
              d={`M${x(p.net_cost_usd) - 3.5},${y(p.reliability_risk_mwh) - 3.5}l7,7M${
                x(p.net_cost_usd) + 3.5
              },${y(p.reliability_risk_mwh) - 3.5}l-7,7`}
              stroke="var(--text-muted)"
              strokeWidth={1.5}
            />
            <circle
              cx={x(p.net_cost_usd)}
              cy={y(p.reliability_risk_mwh)}
              r={8}
              fill="transparent"
            />
          </g>
        ))}
        {plain.map((p) => (
          <circle
            key={p.index}
            cx={x(p.net_cost_usd)}
            cy={y(p.reliability_risk_mwh)}
            r={4}
            fill="var(--series-1)"
            opacity={0.55}
            stroke="var(--surface-1)"
            strokeWidth={2}
            onPointerEnter={() => setHover(p)}
            style={{ cursor: "pointer" }}
          />
        ))}
        {front.map((p) => (
          <circle
            key={p.index}
            cx={x(p.net_cost_usd)}
            cy={y(p.reliability_risk_mwh)}
            r={5}
            fill="var(--series-3)"
            stroke="var(--surface-1)"
            strokeWidth={2}
            onPointerEnter={() => setHover(p)}
            style={{ cursor: "pointer" }}
          />
        ))}
        {chosen && (
          <>
            <circle
              cx={x(chosen.net_cost_usd)}
              cy={y(chosen.reliability_risk_mwh)}
              r={7}
              fill="var(--series-2)"
              stroke="var(--surface-1)"
              strokeWidth={2.5}
              onPointerEnter={() => setHover(chosen)}
            />
            <text
              x={x(chosen.net_cost_usd) + 11}
              y={y(chosen.reliability_risk_mwh) + 4}
              className="label-ink"
            >
              committed
            </text>
          </>
        )}
      </svg>

      {hover && (
        <div className="tooltip" style={{ right: 8, top: 8 }}>
          <div className="t-title">
            #{hover.index} {hover.label}
          </div>
          <div className="t-row">
            <span>Net cost</span>
            <span>{usd(hover.net_cost_usd)}</span>
          </div>
          <div className="t-row">
            <span>Reliability risk</span>
            <span>{hover.reliability_risk_mwh.toFixed(2)} MWh-eq</span>
          </div>
          <div className="t-row">
            <span>Carbon</span>
            <span>{hover.carbon_tonnes.toFixed(1)} t CO₂</span>
          </div>
          <div className="t-row">
            <span>Status</span>
            <span>
              {!hover.survived_guardrails
                ? "removed"
                : hover.index === chosenIndex
                  ? "committed"
                  : hover.on_pareto_front
                    ? "on front"
                    : "dominated"}
            </span>
          </div>
        </div>
      )}

      <Legend
        items={[
          { label: "Committed", color: "var(--series-2)" },
          { label: "Pareto front", color: "var(--series-3)" },
          { label: "Dominated", color: "var(--series-1)" },
          { label: "Removed by a guardrail (×)", color: "var(--text-muted)" },
        ]}
      />
    </div>
  );
}
