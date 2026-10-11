import type { ConfigResponse, Frame } from "../api/types";
import {
  hz,
  intervalsToDuration,
  metresPerSecond,
  mw,
  mwh,
  pct,
  perMwh,
  perTonne,
  signedMw,
} from "../lib/units";
import { assetLabel, assetTone, Chip, LimitChip, Meter, Panel, Stat } from "./ui";

/** PRD: "Portfolio panel: every solar and wind farm (output vs capacity, curtailed or
 *  under maintenance), both batteries (SoC, MW, availability), each industrial consumer,
 *  tie-line and corridor flow vs limits, price, carbon price, DR status, storm alerts,
 *  maintenance plan." */
export function PortfolioPanel({
  frame,
  config,
}: {
  frame: Frame;
  config: ConfigResponse | null;
}) {
  const { portfolio: p, clock } = frame;
  const t = p.totals;
  const reserveFloor = config?.portfolio.battery_limits["reserve_floor_pct"] ?? 10;

  return (
    <div className="stack">
      <Panel title="Portfolio" hint={`interval ${clock.interval} · ${clock.time}`}>
        <div className="statrow" style={{ marginBottom: 12 }}>
          <Stat
            label="Renewable now"
            value={mw(t.renewable_available_mw)}
            sub={`${mw(t.renewable_used_mw)} used`}
          />
          <Stat label="Demand" value={mw(t.demand_mw)} sub={`municipal ${mw(p.municipal_mw)}`} />
          <Stat
            label={t.net_load_mw >= 0 ? "Deficit" : "Surplus"}
            value={mw(Math.abs(t.net_load_mw))}
            sub={t.net_load_mw >= 0 ? "to cover" : "to absorb or store"}
          />
          <Stat
            label="Fleet SoC"
            value={pct(t.fleet_soc_pct, 1)}
            sub={`reserve floor ${pct(reserveFloor)}`}
            tone={t.fleet_soc_pct <= reserveFloor + 1 ? "warning" : undefined}
          />
          <Stat
            label="Curtailed"
            value={mw(t.curtailed_mw)}
            tone={t.curtailed_mw > 0.05 ? "warning" : undefined}
          />
          <Stat
            label="Unserved"
            value={mw(t.unserved_mw)}
            tone={t.unserved_mw > 0.01 ? "critical" : "good"}
          />
        </div>

        <h4 style={{ marginBottom: 6 }}>Solar · {mw(p.solar.reduce((a, f) => a + f.output_mw, 0))} of {mw(p.solar.reduce((a, f) => a + f.capacity_mw, 0))}</h4>
        <div className="rows">
          {p.solar.map((farm) => (
            <div className="row" key={farm.id}>
              <span className="name" style={{ flex: "1 1 120px" }}>
                {farm.name}
              </span>
              <Meter value={farm.output_mw} max={farm.capacity_mw} color="var(--solar)" />
              <span className="num small" style={{ width: 118, textAlign: "right" }}>
                {mw(farm.output_mw)} / {mw(farm.capacity_mw)}
              </span>
              <span className="tiny muted" style={{ width: 74, textAlign: "right" }}>
                cloud {pct(farm.cloud_cover_pct)}
              </span>
              <Chip tone={assetTone(farm.status)}>
                {farm.curtailed_mw > 0.05
                  ? `Curtailed ${mw(farm.curtailed_mw)}`
                  : assetLabel(farm.status)}
              </Chip>
            </div>
          ))}
        </div>

        <h4 style={{ margin: "12px 0 6px" }}>Wind · {mw(p.wind.reduce((a, f) => a + f.output_mw, 0))} of {mw(p.wind.reduce((a, f) => a + f.capacity_mw, 0))}</h4>
        <div className="rows">
          {p.wind.map((farm) => (
            <div className="row" key={farm.id}>
              <span className="name" style={{ flex: "1 1 120px" }}>
                {farm.name}
                <span className="tiny muted"> · {farm.corridor}</span>
              </span>
              <Meter value={farm.output_mw} max={farm.capacity_mw} color="var(--wind)" />
              <span className="num small" style={{ width: 118, textAlign: "right" }}>
                {mw(farm.output_mw)} / {mw(farm.capacity_mw)}
              </span>
              <span className="tiny muted" style={{ width: 74, textAlign: "right" }}>
                {metresPerSecond(farm.wind_speed_ms)}
              </span>
              <Chip tone={assetTone(farm.status)}>
                {farm.curtailed_mw > 0.05
                  ? `Curtailed ${mw(farm.curtailed_mw)}`
                  : assetLabel(farm.status)}
              </Chip>
            </div>
          ))}
        </div>

        <h4 style={{ margin: "12px 0 6px" }}>Storage</h4>
        <div className="rows">
          {p.batteries.map((b) => (
            <div className="row" key={b.id}>
              <span className="name" style={{ flex: "1 1 120px" }}>
                {b.name}
                <span className="tiny muted">
                  {" "}
                  · {mw(b.power_mw)} / {mwh(b.energy_mwh)}
                </span>
              </span>
              <Meter
                value={b.soc_pct}
                max={100}
                limit={reserveFloor}
                color="var(--battery)"
              />
              <span className="num small" style={{ width: 118, textAlign: "right" }}>
                {pct(b.soc_pct, 1)} · {mwh(b.stored_mwh)}
              </span>
              <span className="num small" style={{ width: 74, textAlign: "right" }}>
                {signedMw(b.dispatch_mw)}
              </span>
              <Chip tone={assetTone(b.status)}>{assetLabel(b.status)}</Chip>
            </div>
          ))}
        </div>

        <h4 style={{ margin: "12px 0 6px" }}>Industrial consumers</h4>
        <div className="rows">
          {p.consumers.map((c) => (
            <div className="row" key={c.id}>
              <span className="name" style={{ flex: "1 1 120px" }}>
                {c.name}
              </span>
              <Meter value={c.demand_mw} max={Math.max(c.demand_mw, c.demand_mw + c.contracted_dr_mw)} />
              <span className="num small" style={{ width: 118, textAlign: "right" }}>
                {mw(c.demand_mw)}
              </span>
              <span className="tiny muted" style={{ width: 96, textAlign: "right" }}>
                DR {mw(c.contracted_dr_mw)} · shift {mw(c.shiftable_mw)}
              </span>
              <Chip tone={assetTone(c.status)}>
                {c.dr_mw > 0.05
                  ? `Reducing ${mw(c.dr_mw)}`
                  : c.shift_mw > 0.05
                    ? `Shifted ${mw(c.shift_mw)}`
                    : assetLabel(c.status)}
              </Chip>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Grid, market and limits">
        <div className="rows">
          <div className="row">
            <span className="name">Tie-line import</span>
            <Meter
              value={p.grid.import_mw}
              max={p.grid.import_limit_mw}
              color="var(--import)"
            />
            <span className="num small" style={{ width: 128, textAlign: "right" }}>
              {mw(p.grid.import_mw)} / {mw(p.grid.import_limit_mw)}
            </span>
            <LimitChip status={p.grid.import_status} />
          </div>
          <div className="row">
            <span className="name">Tie-line export</span>
            <Meter value={p.grid.export_mw} max={p.grid.export_limit_mw} />
            <span className="num small" style={{ width: 128, textAlign: "right" }}>
              {mw(p.grid.export_mw)} / {mw(p.grid.export_limit_mw)}
            </span>
            <LimitChip status={p.grid.export_status} />
          </div>
          {p.corridors.map((c) => (
            <div className="row" key={c.corridor}>
              <span className="name">{c.corridor} corridor</span>
              <Meter value={c.flow_mw} max={Math.max(c.limit_mw, c.flow_mw, 1)} color="var(--wind)" />
              <span className="num small" style={{ width: 128, textAlign: "right" }}>
                {mw(c.flow_mw)} /{" "}
                {c.limit_mw > 1e5 ? "unconstrained" : mw(c.limit_mw)}
              </span>
              <LimitChip status={c.status} />
            </div>
          ))}
        </div>

        <div className="statrow" style={{ marginTop: 12 }}>
          <Stat label="Spot price" value={perMwh(p.grid.price_per_mwh)} />
          <Stat label="Carbon price" value={perTonne(p.grid.carbon_price_per_tonne)} />
          <Stat
            label="Frequency"
            value={hz(p.grid.frequency_hz)}
            tone={p.grid.frequency_status === "low" ? "critical" : "good"}
            sub={p.grid.frequency_status === "low" ? "reflex active" : "nominal"}
          />
          <Stat
            label="Demand response"
            value={p.demand_response.active ? mw(p.demand_response.dr_mw) : "Off"}
            sub={`${p.demand_response.calls_today} call(s) today of ${mw(
              p.demand_response.contracted_mw,
            )} contracted`}
            tone={p.demand_response.active ? "warning" : undefined}
          />
        </div>
        {p.demand_response.iso_call && (
          <p className="small" style={{ marginTop: 8 }}>
            <Chip tone="warning">ISO demand-response call active</Chip>
          </p>
        )}
      </Panel>

      {p.storm_alerts.length > 0 && (
        <Panel title="Storm alerts">
          <div className="rows">
            {p.storm_alerts.map((s, i) => (
              <div className="row" key={i}>
                <span className="name">
                  {s.label}
                  <span className="tiny muted"> · {s.severity}</span>
                </span>
                <span className="small">
                  {s.intervals_to_arrival !== null && s.intervals_to_arrival > 0
                    ? `arrives in ${intervalsToDuration(
                        s.intervals_to_arrival,
                        clock.interval_minutes,
                      )}`
                    : "on site"}
                </span>
                {s.extracted_by && (
                  <Chip tone="neutral">
                    extracted by {s.extracted_by} · {pct(s.extraction_confidence_pct)}
                  </Chip>
                )}
                <Chip tone={s.severity === "severe" ? "critical" : "warning"}>
                  {s.region ?? s.corridor ?? "territory-wide"}
                </Chip>
              </div>
            ))}
          </div>
        </Panel>
      )}

      <Panel title="Maintenance plan" hint={`${frame.maintenance.scheduled.length} window(s)`}>
        {frame.maintenance.scheduled.length === 0 ? (
          <p className="empty">No windows scheduled.</p>
        ) : (
          <div className="rows">
            {frame.maintenance.scheduled.map((w) => {
              const valuation = frame.maintenance.windows.find((v) => v.window_id === w.id);
              return (
                <div className="row" key={w.id}>
                  <span className="name">
                    {valuation?.asset_name ?? w.asset_id}
                    <span className="tiny muted"> · {w.crew}</span>
                  </span>
                  <span className="small num">
                    {intervalsToDuration(
                      w.end_interval - w.start_interval,
                      clock.interval_minutes,
                    )}{" "}
                    from interval {w.start_interval}
                  </span>
                  <Chip
                    tone={
                      w.status === "in_progress"
                        ? "serious"
                        : w.status === "complete"
                          ? "good"
                          : "neutral"
                    }
                  >
                    {w.status.replace(/_/g, " ")}
                  </Chip>
                  {valuation?.recommend_delay && <Chip tone="warning">move proposed</Chip>}
                </div>
              );
            })}
          </div>
        )}
        {frame.maintenance.notes.length > 0 && (
          <ul className="tiny muted" style={{ margin: "8px 0 0", paddingLeft: 18 }}>
            {frame.maintenance.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
