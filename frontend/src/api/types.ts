/** The frame contract.
 *
 * One frame is everything the console renders for one 15-minute interval. These types
 * mirror `orchestrator/api/session.py` exactly; the backend builds the frame, the
 * console only reads it. Where a figure carries a unit, the unit is in the key name —
 * that is the backend's own invariant (a test walks the frame and fails on any numeric
 * key without a unit suffix), so it is worth preserving here rather than renaming.
 */

export interface Clock {
  interval: number;
  minute_of_day: number;
  time: string;
  day: number;
  interval_minutes: number;
  running: boolean;
  speed_x: number;
}

export interface Scenario {
  preset: string;
  label: string;
  description: string;
  seed: number;
}

export type Mode = "agent" | "rule_based_fallback" | "degraded";

export interface Verification {
  ok: boolean;
  reason: string;
  checked: number;
  claims?: unknown[];
}

export interface Status {
  mode: Mode;
  committed: boolean;
  reason: string;
  llm_available: boolean;
  explanation_source: "llm" | "template";
  fallback_reason: string | null;
  reserve_held_pct: number | null;
  evaluation_ms: number;
  cycle_ms: number;
  verification: Verification | null;
}

export interface SolarPanelRow {
  id: string;
  name: string;
  capacity_mw: number;
  output_mw: number;
  curtailed_mw: number;
  cloud_region: string;
  cloud_cover_pct: number;
  utilisation_pct: number;
  status: string;
}

export interface WindPanelRow {
  id: string;
  name: string;
  capacity_mw: number;
  output_mw: number;
  curtailed_mw: number;
  corridor: string;
  wind_speed_ms: number;
  utilisation_pct: number;
  status: string;
}

export interface BatteryPanelRow {
  id: string;
  name: string;
  power_mw: number;
  energy_mwh: number;
  soc_pct: number;
  stored_mwh: number;
  dispatch_mw: number;
  available: boolean;
  status: "tripped" | "discharging" | "charging" | "holding";
}

export interface ConsumerPanelRow {
  id: string;
  name: string;
  demand_mw: number;
  contracted_dr_mw: number;
  shiftable_mw: number;
  dr_mw: number;
  shift_mw: number;
  critical: boolean;
  status: string;
}

export interface CorridorRow {
  corridor: string;
  flow_mw: number;
  limit_mw: number;
  status: LimitStatus;
}

export type LimitStatus = "at_limit" | "near_limit" | "normal" | "idle";

export interface GridPanel {
  import_mw: number;
  export_mw: number;
  import_limit_mw: number;
  export_limit_mw: number;
  price_per_mwh: number;
  carbon_price_per_tonne: number;
  frequency_hz: number;
  import_status: LimitStatus;
  export_status: LimitStatus;
  frequency_status: "low" | "nominal";
}

export interface DemandResponsePanel {
  active: boolean;
  iso_call: boolean;
  dr_mw: number;
  contracted_mw: number;
  calls_today: number;
  shift_debt_mwh: number;
}

export interface EventSummary {
  type: string;
  label: string;
  severity: "low" | "moderate" | "high" | "severe";
  start_interval: number;
  end_interval: number | null;
  arrival_interval: number | null;
  intervals_to_arrival: number | null;
  region: string | null;
  corridor: string | null;
  affected_assets: string[];
  note: string;
  extracted_by: string | null;
  extraction_confidence_pct: number;
  confirmed_by_operator: boolean;
  needs_confirmation: boolean;
  source_phrases: Record<string, string>;
}

export interface PortfolioTotals {
  renewable_available_mw: number;
  renewable_used_mw: number;
  demand_mw: number;
  net_load_mw: number;
  curtailed_mw: number;
  unserved_mw: number;
  fleet_soc_pct: number;
}

export interface Portfolio {
  solar: SolarPanelRow[];
  wind: WindPanelRow[];
  batteries: BatteryPanelRow[];
  consumers: ConsumerPanelRow[];
  corridors: CorridorRow[];
  municipal_mw: number;
  grid: GridPanel;
  demand_response: DemandResponsePanel;
  storm_alerts: EventSummary[];
  totals: PortfolioTotals;
}

export interface HistoryRow {
  interval: number;
  time: string;
  solar_mw: number;
  wind_mw: number;
  curtailed_mw: number;
  battery_discharge_mw: number;
  battery_charge_mw: number;
  import_mw: number;
  export_mw: number;
  demand_mw: number;
  dr_mw: number;
  unserved_mw: number;
  price_per_mwh: number;
  fleet_soc_pct: number;
  mode: Mode;
}

export interface Band {
  p10: number[];
  p50: number[];
  p90: number[];
  source: "forecaster" | "scenario_fan" | "unavailable";
}

export interface Forecast {
  steps: number;
  horizon_hours: number;
  scenarios: number;
  projected_fleet_soc_pct: number[];
  solar: Band;
  wind: Band;
  demand: Band;
  price: Band;
}

export interface ActiveRegime {
  key: string;
  label: string;
  trigger: string;
  multipliers?: Record<string, number>;
  risk_aversion?: number;
}

export interface GuardrailRow {
  key: string;
  rule: string;
  applied: boolean;
  removed_count: number;
  note: string;
}

export interface ClusterLabel {
  index: number;
  battery: string;
  market: string;
  curtailment: string;
  demand_response: boolean;
  load_shift: boolean;
  label: string;
}

export interface ClusterScoreRow {
  index: number;
  cluster: ClusterLabel;
  score: number;
  objectives: Record<string, number>;
  contributions: Record<string, number>;
  cost_p10: number;
  cost_p50: number;
  cost_p90: number;
  unserved_mwh: number;
  survived_guardrails: boolean;
  on_pareto_front: boolean;
}

export interface ScatterPoint {
  index: number;
  label: string;
  score: number;
  net_cost_usd: number;
  reliability_risk_mwh: number;
  carbon_tonnes: number;
  survived_guardrails: boolean;
  on_pareto_front: boolean;
}

export interface CommittedInterval {
  battery_mw: Record<string, number>;
  battery_soc_pct: Record<string, number>;
  import_mw: number;
  export_mw: number;
  curtailed_mw: number;
  dr_mw: number;
  shift_mw: number;
  unserved_mw: number;
  price_per_mwh: number;
  frequency_hz: number;
  forced_injection_mw: number;
  reserve_shortfall_mw: number;
}

export interface Trace {
  situation: { regimes: ActiveRegime[]; text: string };
  criteria: {
    base_weights: Record<string, number>;
    final_weights: Record<string, number>;
    multiplier_applied: Record<string, number>;
    shifted: Record<string, [number, number]>;
    risk_lambda: number;
    base_risk_lambda: number;
    text: string;
  };
  options: {
    clusters_evaluated: number;
    scenarios: number;
    steps: number;
    guardrails: GuardrailRow[];
    guardrails_removed: number;
    surviving: number;
    top5: ClusterScoreRow[];
    pareto_front: number[];
    scatter: ScatterPoint[];
    text: string;
  };
  committed: {
    index: number;
    cluster: ClusterLabel;
    held_by_hysteresis: boolean;
    objectives: Record<string, number>;
    contributions: Record<string, number>;
    cost_p10_usd: number;
    cost_p50_usd: number;
    cost_p90_usd: number;
    interval: CommittedInterval;
  };
  tradeoff: {
    alternative: {
      index: number;
      label: string;
      survived_guardrails: boolean;
      objectives: Record<string, number>;
    } | null;
    objective_deltas: Record<string, number>;
    deciding_objective: string | null;
    text: string;
  };
  explanation: string;
  explanation_source: "llm" | "template";
}

export interface GatedAction {
  verb: string;
  asset_id: string;
  asset_name: string;
  mw: number;
  unit: string;
  reason: string;
  requires_approval: boolean;
  verdict: "executed" | "queued" | "rejected";
  gate_reason: string;
  signature: string;
}

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
  result: string;
  ok: boolean;
  side_effect: boolean;
  needs_approval: boolean;
  error: string | null;
}

export interface AgentPanelData {
  llm_available: boolean;
  unavailable_reason: string;
  model: string;
  iterations: number;
  tool_calls: ToolCall[];
  fallback_reason: string | null;
  usage: { calls: number; input_tokens: number; output_tokens: number; errors: number };
}

export interface Proposal {
  id: string;
  kind: "set_point" | "maintenance" | "crew";
  status: "pending" | "approved" | "rejected" | "superseded";
  interval: number;
  summary: string;
  reason?: string;
  decided_by?: string | null;
  verb?: string;
  asset_id?: string;
  asset_name?: string;
  mw?: number;
  unit?: string;
  window_id?: string;
  current_start?: number;
  proposed_start?: number;
  saving_usd?: number;
}

export interface MaintenanceWindowRow {
  window_id: string;
  asset_id: string;
  asset_name: string;
  current_start: number;
  best_start: number;
  current_loss_usd: number;
  best_loss_usd: number;
  saving_usd: number;
  storm_overlap: boolean;
  recommend_delay: boolean;
  reason: string;
}

export interface MaintenancePanel {
  windows: MaintenanceWindowRow[];
  notes: string[];
  proposed_delays: Proposal[];
  work_orders: Proposal[];
  scheduled: {
    id: string;
    asset_id: string;
    status: string;
    start_interval: number;
    end_interval: number;
    deadline_interval: number;
    crew: string;
  }[];
}

export interface DataQuality {
  stale: string[];
  substituted: string[];
  intervals_stale: Record<string, number>;
  degraded: boolean;
  alarm: boolean;
  limit_intervals: number;
}

export type Severity = "info" | "warning" | "critical";

export interface Alarm {
  kind: string;
  severity: Severity;
  message: string;
}

export interface PlanSummary {
  cluster: string | null;
  cluster_index: number | null;
  mode: Mode;
  battery_mw: Record<string, number>;
  import_mw: number;
  export_mw: number;
  curtailed_mw: number;
  dr_mw: number;
  unserved_mw: number;
  regimes: string[];
}

export interface ChangedAction {
  verb: string;
  asset_id: string;
  asset_name: string;
  before_mw: number | null;
  after_mw: number | null;
  reason: string;
}

export interface ResponseToChange {
  trigger: string;
  event: EventSummary | null;
  before: PlanSummary;
  after: PlanSummary;
  changed_actions: ChangedAction[];
  evaluation_ms: number;
  cycle_ms: number;
}

export interface Frame {
  clock: Clock;
  scenario: Scenario;
  status: Status;
  portfolio: Portfolio;
  history: HistoryRow[];
  forecast: Forecast;
  trace: Trace;
  actions: GatedAction[];
  agent: AgentPanelData;
  approvals: Proposal[];
  maintenance: MaintenancePanel;
  events: EventSummary[];
  data_quality: DataQuality;
  alarms: Alarm[];
  response_to_change: ResponseToChange | null;
}

export interface LogEntry {
  kind: string;
  interval: number;
  time: string;
  severity: Severity;
  message: string;
  detail: Record<string, unknown>;
}

// ---- configuration -------------------------------------------------------
export interface ConfigResponse {
  portfolio: {
    version: number;
    interval_minutes: number;
    solar: { id: string; name: string; capacity_mw: number; cloud_region: string }[];
    wind: { id: string; name: string; capacity_mw: number; corridor: string }[];
    batteries: {
      id: string;
      name: string;
      power_mw: number;
      energy_mwh: number;
      one_way_efficiency: number;
      wear_cost_per_mwh: number;
    }[];
    consumers: {
      id: string;
      name: string;
      base_mw: number;
      shiftable_mw: number;
      dr_mw: number;
      critical: boolean;
    }[];
    battery_limits: Record<string, number>;
    grid: Record<string, number>;
    cloud_regions: string[];
    corridors: string[];
  };
  objectives: {
    version: number;
    lookahead_hours: number;
    steps: number;
    scenarios: number;
    base_weights: Record<string, number>;
    risk: Record<string, number>;
    hysteresis_margin: number;
    guardrails: Record<string, unknown>;
    regimes: ActiveRegime[];
    autonomy: Record<string, unknown>;
    maintenance: Record<string, unknown>;
  };
  market: {
    version: number;
    price: Record<string, number>;
    carbon: Record<string, unknown>;
    demand_response: Record<string, number>;
    load_shift: Record<string, number>;
    value_of_lost_load: Record<string, number>;
  };
  presets: Record<string, { label: string; description: string }>;
  injections: Record<string, { label: string; defaults: Record<string, unknown> }>;
  path_choices: number[];
  strategies: string[];
  change_log: string[];
}

// ---- F3 simulation -------------------------------------------------------
export interface StrategyKpis {
  strategy: string;
  paths: number;
  [metric: string]:
    | string
    | number
    | { mean: number; p10?: number; p50?: number; p90?: number };
}

export interface Fan {
  p10: number[];
  p50: number[];
  p90: number[];
}

export interface SimulationResult {
  preset: string;
  paths: number;
  intervals: number;
  seeds: number[];
  kpis: StrategyKpis[];
  comparison: Record<
    string,
    {
      cost_delta_pct: number;
      net_cost_delta_pct: number;
      shed_delta_pct: number;
      co2_delta_pct: number;
      win_rate_cost: number;
      win_rate_net_cost: number;
    }
  >;
  fans: Record<
    string,
    { cumulative_cost: Fan; fleet_soc_pct: Fan; price_per_mwh: Fan }
  >;
  strategy_mix_by_hour?: Record<string, number>[];
}

export interface SimulationJob {
  id: string;
  status: "running" | "done" | "error";
  preset: string;
  paths: number;
  intervals: number;
  done: number;
  total: number;
  stage?: string;
  result: SimulationResult | null;
  error: string | null;
}

export interface SimulationArtifact {
  available: boolean;
  reason?: string;
  generated_from?: string;
  presets?: Record<string, SimulationResult>;
}

// ---- D2 reliability ------------------------------------------------------
export interface InvariantRow {
  key: string;
  label: string;
  pass_condition: string;
  passed: number;
  failed: number;
  applicable: number;
  rate: number;
  failures: string[];
}

export interface ReliabilityArtifact {
  available: boolean;
  reason?: string;
  episodes?: number;
  intervals_each?: number;
  decisions?: number;
  elapsed_s?: number;
  invariants?: InvariantRow[];
  benchmark?: Record<string, number>;
  rows?: {
    episode: number;
    preset: string;
    world_seed: number;
    start: number;
    events: number;
    agent: Record<string, number>;
    rule_based: Record<string, number>;
    no_storage: Record<string, number>;
  }[];
}

export interface HealthResponse {
  status: string;
  clusters: number;
  forecaster_trained: boolean;
  llm_available: boolean;
  config_version: { portfolio: number; objectives: number; market: number };
  interval: number | null;
}

export interface AskAnswer {
  question: string;
  text: string;
  source: "llm" | "template";
  verification: Verification | null;
  citations: string[];
}

export interface BulletinResponse {
  source: "llm" | "rules" | "none";
  issues: string[];
  needs_confirmation: boolean;
  event: EventSummary | null;
  text: string;
  frame?: Frame;
}

export interface ReplayResponse {
  replay_key: {
    interval: number;
    seed: number | null;
    preset: string | null;
    world_seed: number | null;
    chosen_index: number | null;
    config_version: number | null;
  };
  superseded: number;
  decision: Record<string, unknown>;
}
