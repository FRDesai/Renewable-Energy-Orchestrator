"""Dispatch simulation: the trusted numeric core.

Every MW figure the agent ever executes comes out of here. Vectorised over
(n_clusters, n_scenarios, n_batteries) and stepped over the look-ahead, so one call
evaluates all 80 clusters against the whole scenario fan.

Per-step order of operations, which is what makes the PRD's invariants hold by
construction rather than by checking afterwards:

  1. Availability (scenario) minus maintenance outages.
  2. Corridor limit -> forced curtailment of units behind the constrained corridor.
  3. Demand after demand response and load shifting.
  4. Battery target from the cluster's strategy.
  5. Frequency reflex: below 49.9 Hz batteries must inject.
  6. Clamp to power ratings and SoC.
  7. Export cap (derated, and zero when voluntarily curtailing at negative prices):
     reduce discharge first, then absorb into charging, then curtail.
  8. Import cap: discharge to the emergency floor before any load is shed.

Grid exchange is a single signed number, so importing and exporting in the same
interval is impossible by construction -- the "no wash trading" invariant cannot
fail. Likewise step 7 exhausts discharge reduction before curtailing, so
"discharge while curtailing for limits" cannot occur.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..config_loader import Config
from ..domain.actions import CLUSTERS, ActionCluster
from .clusters import ARBITRAGE, DISCHARGE, ENCODING, HOLD, RESERVE, SOAK, ClusterEncoding

OBJECTIVES = (
    "net_cost",
    "reliability_risk",
    "carbon",
    "curtailment",
    "grid_stability",
    "battery_wear",
)


@dataclass
class ScenarioSet:
    """The scenario fan the optimizer scores against.

    Shapes: (S, T) for scenario-varying series, (T,) for announced limits that are
    the same in every scenario. Scenario 0 is the expected path.
    """

    solar_by_farm: dict[str, np.ndarray]       # farm id -> (S, T) available
    wind_by_farm: dict[str, np.ndarray]        # farm id -> (S, T) available
    wind_corridor_of: dict[str, str]           # farm id -> corridor
    demand_mw: np.ndarray                # (S, T) total load
    price_per_mwh: np.ndarray            # (S, T)
    frequency_hz: np.ndarray             # (S, T)
    import_limit_mw: np.ndarray          # (T,)
    export_limit_mw: np.ndarray          # (T,)
    corridor_limit_mw: dict[str, np.ndarray]   # corridor -> (T,)
    peak_price: np.ndarray               # (S, T) bool: price at/above the day's 70th pct
    iso_dr_call: np.ndarray              # (S, T) bool
    dr_capacity_mw: float
    shift_capacity_mw: float
    storm_soc_target_pct: np.ndarray     # (T,) required fleet SoC, 0 when no storm
    price_quantile_next24: float         # upper-quartile price, for stored-energy value
    arbitrage_low_per_mwh: np.ndarray    # (S,) charge below this
    arbitrage_high_per_mwh: np.ndarray   # (S,) discharge above this

    def __post_init__(self) -> None:
        self.solar_mw = sum(self.solar_by_farm.values())
        self.wind_mw = {}
        for farm_id, series in self.wind_by_farm.items():
            corridor = self.wind_corridor_of[farm_id]
            self.wind_mw[corridor] = self.wind_mw.get(corridor, 0.0) + series

    @property
    def n_scenarios(self) -> int:
        return self.demand_mw.shape[0]

    @property
    def n_steps(self) -> int:
        return self.demand_mw.shape[1]

    def farms_in(self, corridor: str) -> tuple[str, ...]:
        return tuple(f for f, c in self.wind_corridor_of.items() if c == corridor)


@dataclass
class StepRecord:
    """Per-interval detail for one cluster/scenario -- what the invariants check."""

    step: int
    solar_available_mw: float
    wind_available_mw: float
    solar_curtailed_mw: float
    wind_curtailed_mw: float
    curtailed_by_corridor_mw: float
    curtailed_by_export_mw: float
    curtailed_at_negative_price_mw: float
    demand_mw: float
    dr_mw: float
    shift_mw: float
    battery_mw: dict[str, float]         # + discharge, - charge
    battery_soc_pct: dict[str, float]    # end of step
    grid_mw: float                       # + import, - export
    unserved_mw: float
    price_per_mwh: float
    frequency_hz: float
    import_limit_mw: float
    export_limit_mw: float
    corridor_flow_mw: dict[str, float]
    corridor_limit_mw: dict[str, float]
    reserve_available_mw: float
    reserve_shortfall_mw: float
    forced_injection_mw: float = 0.0   # the frequency reflex floor for this step
    solar_curtailed_by_farm: dict[str, float] = field(default_factory=dict)
    wind_curtailed_by_farm: dict[str, float] = field(default_factory=dict)

    @property
    def renewable_used_mw(self) -> float:
        return (
            self.solar_available_mw + self.wind_available_mw
            - self.solar_curtailed_mw - self.wind_curtailed_mw
        )

    @property
    def total_curtailed_mw(self) -> float:
        return self.solar_curtailed_mw + self.wind_curtailed_mw

    @property
    def battery_discharge_mw(self) -> float:
        return sum(v for v in self.battery_mw.values() if v > 0)

    @property
    def battery_charge_mw(self) -> float:
        return sum(-v for v in self.battery_mw.values() if v < 0)

    @property
    def import_mw(self) -> float:
        return max(0.0, self.grid_mw)

    @property
    def export_mw(self) -> float:
        return max(0.0, -self.grid_mw)


@dataclass
class DispatchOutcome:
    """Objective values per (cluster, scenario), plus the raw pieces the explainer
    and the invariant suite need."""

    objectives: dict[str, np.ndarray]     # each (C, S)
    unserved_mwh: np.ndarray              # (C, S)
    reserve_shortfall_mwh: np.ndarray     # (C, S)
    curtailed_mwh: np.ndarray             # (C, S)
    import_mwh: np.ndarray                # (C, S)
    export_mwh: np.ndarray                # (C, S)
    throughput_mwh: np.ndarray            # (C, S)
    end_soc_mwh: np.ndarray               # (C, S)
    clusters: tuple[ActionCluster, ...]
    soc_path_mwh: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    injected_mw: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    records: dict[tuple[int, int], tuple[StepRecord, ...]] = field(default_factory=dict)

    def trace(self, cluster_index: int, scenario_index: int = 0) -> tuple[StepRecord, ...]:
        return self.records[(cluster_index, scenario_index)]

    def objective_at(self, cluster_index: int, scenario_index: int) -> dict[str, float]:
        return {k: float(v[cluster_index, scenario_index]) for k, v in self.objectives.items()}


class DispatchEngine:
    def __init__(self, config: Config, encoding: ClusterEncoding = ENCODING) -> None:
        self.config = config
        self.enc = encoding
        p = config.portfolio
        self.h = p.interval_hours
        self.batteries = p.batteries
        self.n_batteries = len(p.batteries)
        # Merit order for charge/discharge allocation: cheapest wear first.
        self.merit = sorted(range(self.n_batteries), key=lambda i: p.batteries[i].wear_cost_per_mwh)
        self.power = np.array([b.power_mw for b in p.batteries])
        self.energy = np.array([b.energy_mwh for b in p.batteries])
        self.eff = np.array([b.one_way_efficiency for b in p.batteries])
        self.wear = np.array([b.wear_cost_per_mwh for b in p.batteries])
        lim = p.battery_limits
        # Two floors. Normal dispatch holds above `reserve`; only the emergency
        # cascade, run to avoid shedding load, may go below it down to `hard`.
        self.hard_floor_mwh = self.energy * lim.soc_min_pct / 100.0
        self.reserve_floor_mwh = self.energy * lim.reserve_floor_pct / 100.0
        self.soc_max_mwh = self.energy * lim.soc_max_pct / 100.0
        self.spread = config.market.price["buy_spread_per_mwh"]
        self.carbon_intensity = config.market.carbon["grid_intensity_t_per_mwh"]
        self.dr_pay = config.market.demand_response["customer_payment_per_mwh"]
        self.iso_pay = config.market.demand_response["iso_call_payment_per_mwh"]
        self.shift_pay = config.market.load_shift["payment_per_mwh"]
        self.reserve_req = config.objectives.reliability["fast_reserve_requirement_mw"]
        self.unserved_w = config.objectives.reliability["unserved_weight"]
        self.shortfall_w = config.objectives.reliability["reserve_shortfall_weight"]
        self.stored_q = config.objectives.stored_energy["price_quantile"]
        self.stored_haircut = config.objectives.stored_energy["efficiency_haircut"]
        self.forbid_discharge_while_curtailing = bool(
            config.objectives.guardrails.get("forbid_discharge_while_curtailing", True)
        )

    # ---- battery limits ---------------------------------------------------
    def _max_discharge_mw(self, soc: np.ndarray, floor_mwh: np.ndarray,
                          available: np.ndarray) -> np.ndarray:
        """MW deliverable to the busbar for one interval without crossing `floor`."""
        from_energy = np.maximum(0.0, soc - floor_mwh) * self.eff / self.h
        return np.minimum(self.power, from_energy) * available

    def _max_charge_mw(self, soc: np.ndarray, available: np.ndarray) -> np.ndarray:
        """MW absorbable from the busbar for one interval without exceeding soc_max."""
        headroom = np.maximum(0.0, self.soc_max_mwh - soc)
        from_energy = headroom / (self.eff * self.h)
        return np.minimum(self.power, from_energy) * available

    def _allocate(self, total: np.ndarray, caps: np.ndarray) -> np.ndarray:
        """Split a non-negative fleet total across batteries in merit order.

        total: (C, S) >= 0.  caps: (C, S, B) >= 0.  Returns (C, S, B).
        Exact: each battery takes min(remaining, its cap) in a fixed order.
        """
        out = np.zeros_like(caps)
        remaining = total.copy()
        for b in self.merit:
            take = np.minimum(remaining, caps[:, :, b])
            out[:, :, b] = take
            remaining = remaining - take
        return out

    # ---- strategy ---------------------------------------------------------
    def _battery_target(
        self,
        net_mw: np.ndarray,          # (C, S) demand after DR/shift minus renewables used
        price: np.ndarray,           # (1, S)
        soc_total: np.ndarray,       # (C, S)
        max_dis: np.ndarray,         # (C, S)
        max_chg: np.ndarray,         # (C, S)
        scen: ScenarioSet,
        step: int,
        import_limit: float,
    ) -> np.ndarray:
        """Signed fleet target in MW: positive discharge, negative charge."""
        enc = self.enc
        surplus_charge = np.minimum(0.0, net_mw)          # negative when surplus
        deficit = np.maximum(0.0, net_mw)
        lo = scen.arbitrage_low_per_mwh.reshape(1, -1)
        hi = scen.arbitrage_high_per_mwh.reshape(1, -1)
        fleet_energy = float(self.energy.sum())

        # Soak surplus: absorb everything spare, never discharge.
        t_soak = surplus_charge

        # Price arbitrage: charge below the horizon's low band, discharge above the
        # high band. This is the one strategy that charges from the grid on purpose.
        t_arb = np.where(price < lo, -max_chg, np.where(price > hi, max_dis, surplus_charge))

        # Hold for higher price: charge from surplus, discharge only at top prices.
        t_hold = np.where(price >= hi, max_dis, surplus_charge)

        # Reserve for emergencies: hold a reserve level, and -- the PRD's carbon fix --
        # fill it from surplus rather than from grid imports. Grid charging is only
        # allowed when a storm pre-charge target is not yet met.
        storm_target_pct = float(scen.storm_soc_target_pct[step])
        target_mwh = fleet_energy * storm_target_pct / 100.0
        below_target = soc_total < target_mwh
        grid_charge_allowed = below_target & (storm_target_pct > 0.0)
        reserve_charge = np.where(
            grid_charge_allowed,
            -max_chg,                       # pre-charge for the storm, carbon priced in
            surplus_charge,                 # otherwise surplus only
        )
        # Discharge only to cover a deficit the tie-line cannot carry.
        beyond_import = np.maximum(0.0, deficit - import_limit)
        t_reserve = np.where(beyond_import > 0, np.minimum(max_dis, beyond_import),
                             reserve_charge)

        # Discharge now: everything available.
        t_dis = max_dis

        target = np.where(
            enc.is_strategy(SOAK), t_soak,
            np.where(enc.is_strategy(ARBITRAGE), t_arb,
                     np.where(enc.is_strategy(HOLD), t_hold,
                              np.where(enc.is_strategy(RESERVE), t_reserve, t_dis))),
        )

        # Market posture: "batteries before imports" meets any deficit from storage
        # first, up to what is available.
        before_imports = np.minimum(max_dis, deficit)
        target = np.where(
            enc.batteries_before_imports & (deficit > 0),
            np.maximum(target, before_imports),
            target,
        )
        return target

    # ---- the simulation ---------------------------------------------------
    def simulate(
        self,
        scen: ScenarioSet,
        soc_start_pct: dict[str, float],
        available: dict[str, bool],
        *,
        record: tuple[int, int] | Sequence[tuple[int, int]] | None = None,
    ) -> DispatchOutcome:
        """`record=(cluster_index, scenario_index)` -- or a sequence of such pairs --
        also returns the per-step detail for those paths, so the actions committed are
        literally the ones that were scored."""
        paths: tuple[tuple[int, int], ...]
        if record is None:
            paths = ()
        elif isinstance(record, tuple) and len(record) == 2 and isinstance(record[0], int):
            paths = (record,)
        else:
            paths = tuple(record)  # type: ignore[arg-type]
        C, S, B = len(self.enc), scen.n_scenarios, self.n_batteries
        T = scen.n_steps
        h = self.h
        p = self.config.portfolio

        avail = np.array([1.0 if available.get(b.id, True) else 0.0 for b in self.batteries])
        avail3 = np.broadcast_to(avail, (C, S, B))
        soc = np.empty((C, S, B))
        for i, b in enumerate(self.batteries):
            soc[:, :, i] = self.energy[i] * soc_start_pct[b.id] / 100.0

        zeros = np.zeros((C, S))
        acc = {k: zeros.copy() for k in OBJECTIVES}
        unserved = zeros.copy()
        shortfall = zeros.copy()
        curtailed = zeros.copy()
        imported = zeros.copy()
        exported = zeros.copy()
        throughput = zeros.copy()
        prev_grid = zeros.copy()
        soc_path = np.empty((C, S, T))
        injected = np.empty((C, S, T))
        shift_debt = zeros.copy()      # MWh of load deferred, awaiting repayment
        records: dict[tuple[int, int], list[StepRecord]] = {k: [] for k in paths}

        north_cap_series = scen.corridor_limit_mw.get("north")
        for t in range(T):
            price = scen.price_per_mwh[:, t].reshape(1, S)
            imp_lim = float(scen.import_limit_mw[t])
            exp_lim = float(scen.export_limit_mw[t])

            # --- 1/2. availability and the forced corridor curtailment ---------
            solar_av = np.broadcast_to(scen.solar_mw[:, t].reshape(1, S), (C, S)).copy()
            wind_av = {k: np.broadcast_to(v[:, t].reshape(1, S), (C, S)).copy()
                       for k, v in scen.wind_mw.items()}
            curt_corridor = zeros.copy()
            if north_cap_series is not None and "north" in wind_av:
                cap = float(north_cap_series[t])
                over = np.maximum(0.0, wind_av["north"] - cap)
                wind_av["north"] = wind_av["north"] - over
                curt_corridor = curt_corridor + over
            wind_total = sum(wind_av.values())
            renewable = solar_av + wind_total

            # --- 3. demand after demand response and load shifting ------------
            demand = np.broadcast_to(scen.demand_mw[:, t].reshape(1, S), (C, S)).copy()
            peak = np.broadcast_to(scen.peak_price[:, t].reshape(1, S), (C, S))
            iso = np.broadcast_to(scen.iso_dr_call[:, t].reshape(1, S), (C, S))
            dr_window = peak | iso
            dr_mw = np.where(self.enc.demand_response & dr_window, scen.dr_capacity_mw, 0.0)
            shift_now = np.where(self.enc.load_shift & peak, scen.shift_capacity_mw, 0.0)
            # Repay deferred load off-peak, as fast as the shiftable block allows.
            repay = np.where(peak, 0.0, np.minimum(shift_debt / h, scen.shift_capacity_mw))
            shift_net = shift_now - repay          # + reduces load now, - adds it back
            shift_debt = np.maximum(0.0, shift_debt + shift_net * h)
            demand_eff = np.maximum(0.0, demand - dr_mw - shift_net)

            net = demand_eff - renewable           # + deficit, - surplus

            # --- 4/5/6. battery target, frequency reflex, clamping ------------
            caps_dis = self._max_discharge_mw(soc, self.reserve_floor_mwh, avail3)
            caps_chg = self._max_charge_mw(soc, avail3)
            max_dis = caps_dis.sum(axis=-1)
            max_chg = caps_chg.sum(axis=-1)

            target = self._battery_target(
                net, price, soc.sum(axis=-1), max_dis, max_chg, scen, t, imp_lim)

            # A dip below the reflex threshold forces injection in every cluster.
            # This is a safety reflex and outranks both the curtailment rule below and
            # the export cap further down: grid stability beats spilled energy, which
            # is exactly why the frequency-event regime lifts the stability weight.
            freq = np.broadcast_to(scen.frequency_hz[:, t].reshape(1, S), (C, S))
            dipping = freq < p.grid.frequency_reflex_hz
            # The reflex may draw into the reserve band (down to the hard floor): a
            # battery sitting at the reserve floor would otherwise fail to inject at
            # all, and grid stability is the one thing the PRD will not trade.
            caps_reflex = self._max_discharge_mw(soc, self.hard_floor_mwh, avail3)
            reflex_floor = np.where(
                dipping,
                np.minimum(caps_reflex.sum(axis=-1), max(self.reserve_req, 0.0)),
                0.0)
            # The lower bound on the fleet figure. Outside a dip it is full charging
            # power, NOT zero: zero is a legal dispatch value, so flooring at it
            # silently forbade every strategy from charging, and storage could only
            # ever fill as a side effect of the export cascade below.
            floor = np.where(dipping, reflex_floor, -max_chg)

            # Never discharge *voluntarily* while forced to spill for a transmission
            # limit (the PRD's open curtailment defect) -- but never below the reflex,
            # and charging into that spill is still allowed.
            if self.forbid_discharge_while_curtailing:
                target = np.where(
                    curt_corridor > 1e-9,
                    np.maximum(np.minimum(target, 0.0), floor),
                    target,
                )
            target = np.maximum(target, floor)
            ceiling = np.where(dipping, caps_reflex.sum(axis=-1), max_dis)
            battery = np.clip(target, -max_chg, ceiling)

            # --- 7. export cap --------------------------------------------------
            # Voluntary curtailment at negative prices collapses the export cap to
            # zero for clusters running that rule.
            negative = price < 0.0
            effective_exp = np.where(
                self.enc.curtail_at_negative & negative, 0.0, exp_lim)
            grid = net - battery
            over_export = np.maximum(0.0, -grid - effective_exp)
            # Walk the battery down as far as it is allowed to go -- giving up
            # discharge first, then charging into the surplus -- before spilling
            # anything. During a frequency dip the floor is the forced injection, so
            # the reflex survives: a battery cannot inject and charge at once.
            min_battery = floor
            reduction = np.minimum(np.maximum(battery - min_battery, 0.0), over_export)
            battery = battery - reduction
            over_export = over_export - reduction
            # Whatever is left is spilled. Spilling raises the net, because that
            # renewable energy is no longer available to serve load or export.
            curt_limit = over_export
            net = net + curt_limit
            grid = net - battery

            # --- 8. import cap: batteries to the emergency floor before shedding
            over_import = np.maximum(0.0, grid - imp_lim)
            caps_emerg = self._max_discharge_mw(soc, self.hard_floor_mwh, avail3)
            # Relief is measured from wherever the fleet currently sits, including from
            # a *charging* figure: swinging from -80 MW to +120 MW relieves 200 MW, not
            # 120. Measuring from max(battery, 0) understated it, which let load be shed
            # while a half-full battery was still absorbing power.
            extra_room = np.maximum(0.0, caps_emerg.sum(axis=-1) - battery)
            extra = np.minimum(over_import, extra_room)
            battery = battery + extra
            over_import = over_import - extra
            unserved_mw = over_import
            grid = np.minimum(net - battery, imp_lim)

            # --- allocate the fleet figure to individual batteries -------------
            dis_total = np.maximum(battery, 0.0)
            chg_total = np.maximum(-battery, 0.0)
            # Allocate normal discharge against the reserve-floor caps, so no single
            # battery is pushed below its reserve while another sits high. Only the
            # emergency top-up draws on the headroom between reserve and hard floor.
            normal = np.minimum(dis_total, caps_dis.sum(axis=-1))
            emergency = dis_total - normal
            per_dis = (self._allocate(normal, caps_dis)
                       + self._allocate(emergency, np.maximum(caps_emerg - caps_dis, 0.0)))
            per_chg = self._allocate(chg_total, caps_chg)
            per_battery = per_dis - per_chg

            # SoC update: discharging draws mw*h/eff from the cell, charging adds
            # mw*h*eff to it.
            delta = np.where(
                per_battery >= 0,
                -(per_battery * h) / self.eff,
                (-per_battery * h) * self.eff,
            )
            soc = np.clip(soc + delta, 0.0, self.soc_max_mwh)

            # --- curtailment attribution ---------------------------------------
            curt_total = curt_corridor + curt_limit
            # Corridor spill is wind by definition; limit-driven spill follows the
            # configured merit order (solar first, then wind).
            curtailed = curtailed + curt_total * h

            # --- objectives -----------------------------------------------------
            imp_mwh = np.maximum(grid, 0.0) * h
            exp_mwh = np.maximum(-grid, 0.0) * h
            dr_mwh = dr_mw * h
            shift_mwh = np.maximum(shift_now, 0.0) * h
            iso_credit = np.where(iso, dr_mwh * self.iso_pay, 0.0)

            acc["net_cost"] += (
                imp_mwh * (price + self.spread)
                - exp_mwh * price
                + dr_mwh * self.dr_pay
                - iso_credit
                + shift_mwh * self.shift_pay
            )
            acc["carbon"] += imp_mwh * self.carbon_intensity
            acc["curtailment"] += curt_total * h
            tp = np.abs(per_battery).sum(axis=-1) * h
            throughput += tp
            acc["battery_wear"] += (np.abs(per_battery) * h * self.wear).sum(axis=-1)

            reserve_avail = caps_dis.sum(axis=-1) - np.maximum(battery, 0.0)
            reserve_short = np.maximum(0.0, self.reserve_req - np.maximum(reserve_avail, 0.0))
            shortfall += reserve_short * h
            unserved += unserved_mw * h

            ramp = np.abs(grid - prev_grid) / max(1.0, p.grid.import_limit_mw)
            acc["grid_stability"] += ramp + reserve_short / max(1.0, self.reserve_req)
            prev_grid = grid

            imported += imp_mwh
            exported += exp_mwh
            soc_path[:, :, t] = soc.sum(axis=-1)
            injected[:, :, t] = np.maximum(battery, 0.0)

            for path in paths:
                records[path].append(self._record_step(
                    t, path, scen,
                    solar_av=solar_av, wind_av=wind_av,
                    curt_corridor=curt_corridor, curt_limit=curt_limit,
                    demand=demand, dr_mw=dr_mw, shift_net=shift_net,
                    per_battery=per_battery, soc=soc, grid=grid,
                    unserved_mw=unserved_mw, price=price, freq=freq,
                    imp_lim=imp_lim, exp_lim=exp_lim,
                    reserve_avail=reserve_avail, reserve_short=reserve_short,
                    reflex_floor=reflex_floor,
                ))

        # Reliability objective in MWh-equivalent.
        acc["reliability_risk"] = self.unserved_w * unserved + self.shortfall_w * shortfall

        # Value of energy left in storage at the horizon end, so a 4 h view does not
        # drain the fleet cheaply before the evening peak.
        end_soc = soc.sum(axis=-1)
        acc["net_cost"] = acc["net_cost"] - (
            end_soc * scen.price_quantile_next24 * self.stored_haircut
        )

        return DispatchOutcome(
            objectives=acc,
            unserved_mwh=unserved,
            reserve_shortfall_mwh=shortfall,
            curtailed_mwh=curtailed,
            import_mwh=imported,
            export_mwh=exported,
            throughput_mwh=throughput,
            end_soc_mwh=end_soc,
            soc_path_mwh=soc_path,
            injected_mw=injected,
            clusters=self.enc.clusters,
            records={k: tuple(v) for k, v in records.items()},
        )

    # ---- per-step recorder -------------------------------------------------
    def _record_step(
        self, t, record, scen, *, solar_av, wind_av, curt_corridor, curt_limit,
        demand, dr_mw, shift_net, per_battery, soc, grid, unserved_mw, price, freq,
        imp_lim, exp_lim, reserve_avail, reserve_short, reflex_floor,
    ) -> StepRecord:
        c, s = record
        solar_farms = {f: float(v[s, t]) for f, v in scen.solar_by_farm.items()}
        wind_farms = {f: float(v[s, t]) for f, v in scen.wind_by_farm.items()}

        # Corridor spill necessarily falls on the units behind that corridor; share it
        # pro-rata by their availability (see DECISIONS.md D12).
        wind_curt = {f: 0.0 for f in wind_farms}
        corridor_spill = float(curt_corridor[c, s])
        if corridor_spill > 1e-12:
            north = scen.farms_in("north")
            avail = sum(wind_farms[f] for f in north)
            if avail > 1e-12:
                for f in north:
                    wind_curt[f] = corridor_spill * wind_farms[f] / avail

        # Limit-driven spill follows the merit order: solar first, then wind.
        solar_curt = {f: 0.0 for f in solar_farms}
        remaining = float(curt_limit[c, s])
        solar_pool = sum(solar_farms.values())
        if remaining > 1e-12 and solar_pool > 1e-12:
            take = min(remaining, solar_pool)
            for f, v in solar_farms.items():
                solar_curt[f] = take * v / solar_pool
            remaining -= take
        if remaining > 1e-12:
            pool = sum(max(0.0, wind_farms[f] - wind_curt[f]) for f in wind_farms)
            if pool > 1e-12:
                take = min(remaining, pool)
                for f in wind_farms:
                    free = max(0.0, wind_farms[f] - wind_curt[f])
                    wind_curt[f] += take * free / pool

        corridor_flow = {}
        for corridor in set(scen.wind_corridor_of.values()):
            farms = scen.farms_in(corridor)
            corridor_flow[corridor] = sum(
                wind_farms[f] - wind_curt[f] for f in farms)

        return StepRecord(
            step=t,
            solar_available_mw=sum(solar_farms.values()),
            wind_available_mw=sum(wind_farms.values()),
            solar_curtailed_mw=sum(solar_curt.values()),
            wind_curtailed_mw=sum(wind_curt.values()),
            curtailed_by_corridor_mw=corridor_spill,
            curtailed_by_export_mw=float(curt_limit[c, s]) if not (
                self.enc.curtail_at_negative[c, 0] and price[0, s] < 0
            ) else 0.0,
            curtailed_at_negative_price_mw=float(curt_limit[c, s]) if (
                self.enc.curtail_at_negative[c, 0] and price[0, s] < 0
            ) else 0.0,
            demand_mw=float(demand[c, s]),
            dr_mw=float(dr_mw[c, s]) if np.ndim(dr_mw) else float(dr_mw),
            shift_mw=float(shift_net[c, s]) if np.ndim(shift_net) else float(shift_net),
            battery_mw={b.id: float(per_battery[c, s, i])
                        for i, b in enumerate(self.batteries)},
            battery_soc_pct={b.id: float(100.0 * soc[c, s, i] / self.energy[i])
                             for i, b in enumerate(self.batteries)},
            grid_mw=float(grid[c, s]),
            unserved_mw=float(unserved_mw[c, s]),
            price_per_mwh=float(price[0, s]),
            frequency_hz=float(freq[c, s]),
            import_limit_mw=imp_lim,
            export_limit_mw=exp_lim,
            corridor_flow_mw=corridor_flow,
            corridor_limit_mw={
                k: float(v[t]) for k, v in scen.corridor_limit_mw.items()},
            reserve_available_mw=float(reserve_avail[c, s]),
            reserve_shortfall_mw=float(reserve_short[c, s]),
            forced_injection_mw=float(reflex_floor[c, s]),
            solar_curtailed_by_farm=solar_curt,
            wind_curtailed_by_farm=wind_curt,
        )
