"""M0: the loaded config must match the PRD's reference scenario exactly."""
import pytest

from orchestrator.config_loader import load_config
from orchestrator.domain.actions import CLUSTERS, N_CLUSTERS


def test_solar_fleet_matches_prd(config):
    p = config.portfolio
    assert len(p.solar_farms) == 5
    assert p.solar_capacity_mw == pytest.approx(400.0)
    by_name = {f.name: f.capacity_mw for f in p.solar_farms}
    assert by_name == {
        "Mesa Verde": 120.0, "Dry Creek": 90.0, "Redrock": 80.0,
        "Alder Flats": 60.0, "Pinecrest": 50.0,
    }
    # "Three cloud regions, so clouds hit farms unevenly."
    assert len(p.cloud_regions) == 3
    sizes = sorted(
        sum(f.capacity_mw for f in p.solar_farms if f.cloud_region == r)
        for r in p.cloud_regions
    )
    assert len(set(sizes)) == 3, "cloud regions must be unevenly sized"


def test_wind_fleet_and_corridor_matches_prd(config):
    p = config.portfolio
    assert len(p.wind_farms) == 3
    assert p.wind_capacity_mw == pytest.approx(380.0)
    by_name = {f.name: f.capacity_mw for f in p.wind_farms}
    assert by_name == {"Ridgeback": 160.0, "Gale Point": 120.0, "Coastal Bluff": 100.0}

    north = sum(f.capacity_mw for f in p.wind_farms if f.corridor == "north")
    # The corridor constraint must actually bind: capacity behind it exceeds its limit.
    assert north > p.grid.north_corridor_limit_mw
    assert p.grid.north_corridor_limit_mw == pytest.approx(220.0)
    assert p.grid.north_corridor_derated_mw == pytest.approx(150.0)

    c = p.wind_curve
    assert (c.cut_in_ms, c.rated_ms, c.cut_out_ms) == (3.0, 12.5, 25.0)
    assert c.output_fraction(2.9) == 0.0
    assert c.output_fraction(25.0) == 0.0, "cut-out must produce zero"
    assert c.output_fraction(12.5) == pytest.approx(1.0)
    assert c.output_fraction(30.0) == 0.0
    assert 0.0 < c.output_fraction(8.0) < 1.0


def test_storage_matches_prd(config):
    p = config.portfolio
    assert p.battery_power_mw == pytest.approx(200.0)
    assert p.battery_energy_mwh == pytest.approx(800.0)
    harbor = p.battery("harbor")
    valley = p.battery("valley")
    assert (harbor.power_mw, harbor.energy_mwh) == (120.0, 480.0)
    assert (valley.power_mw, valley.energy_mwh) == (80.0, 320.0)
    # "94-95% one-way efficiency ... wear $14 and $18 per MWh throughput"
    assert {harbor.one_way_efficiency, valley.one_way_efficiency} == {0.95, 0.94}
    assert {harbor.wear_cost_per_mwh, valley.wear_cost_per_mwh} == {14.0, 18.0}
    lim = p.battery_limits
    assert (lim.soc_min_pct, lim.soc_max_pct) == (8.0, 95.0)
    # The reserve floor must sit inside the operating range, above the hard floor:
    # normal dispatch holds above it, the emergency cascade may dip below.
    assert lim.soc_min_pct < lim.reserve_floor_pct < lim.soc_max_pct


def test_consumers_and_load_band_match_prd(config):
    p = config.portfolio
    by_name = {c.name: c for c in p.consumers}
    assert by_name["Northgate Steel"].base_mw == 60.0
    assert by_name["Northgate Steel"].shiftable_mw == 20.0
    assert by_name["Cobalt Data Center"].base_mw == 40.0
    assert by_name["Cobalt Data Center"].critical, "data center is critical"
    assert by_name["Sinclair Chemicals"].dr_mw == 12.0
    assert by_name["Polar Cold Storage"].shiftable_mw == 8.0
    assert (p.municipal.min_mw, p.municipal.max_mw) == (95.0, 180.0)
    assert p.municipal.critical, "municipal load is critical"

    # Critical loads are never curtailed by the agent.
    assert all(c.dr_mw == 0.0 and c.shiftable_mw == 0.0 for c in p.consumers if c.critical)

    # PRD load band 220-330 MW must be reachable.
    peak = sum(c.base_mw for c in p.consumers) + p.municipal.max_mw
    assert peak == pytest.approx(330.0)


def test_grid_and_market_match_prd(config):
    g = config.portfolio.grid
    assert (g.import_limit_mw, g.export_limit_mw) == (220.0, 200.0)
    assert g.export_derated_mw == 80.0
    assert g.frequency_reflex_hz == 49.9

    m = config.market
    assert m.price["buy_spread_per_mwh"] == 6.0
    assert m.carbon["price_per_tonne"] == 85.0
    assert m.carbon["grid_intensity_t_per_mwh"] == 0.42
    assert m.demand_response["customer_payment_per_mwh"] == 120.0
    assert m.demand_response["iso_call_payment_per_mwh"] == 300.0


def test_objectives_match_prd(config):
    o = config.objectives
    # Six objectives with the PRD's default weights (normalised; they already sum to 1).
    assert o.base_weights == pytest.approx({
        "net_cost": 0.30, "reliability_risk": 0.20, "carbon": 0.18,
        "curtailment": 0.14, "grid_stability": 0.10, "battery_wear": 0.08,
    })
    assert sum(o.base_weights.values()) == pytest.approx(1.0)
    # 4-hour look-ahead over 16 steps, 20-24 scenarios.
    assert o.lookahead_hours == 4
    assert o.steps == 16
    assert 20 <= o.scenarios <= 24
    assert o.hysteresis_margin == 0.015
    assert o.risk["lambda_base"] == 0.4
    assert o.risk["lambda_max"] == 1.5
    assert o.stored_energy["efficiency_haircut"] == 0.93
    assert o.reliability["unserved_weight"] == 5.0
    assert o.reliability["reserve_shortfall_weight"] == 0.4
    assert o.maintenance["min_saving_to_delay_usd"] == 2500.0


def test_all_ten_regimes_present_with_triggers(config):
    o = config.objectives
    assert set(o.regimes) == {
        "grid_stress", "frequency_event", "asset_contingency", "scarcity_ahead",
        "price_spike", "peak_pricing", "transmission_constrained",
        "renewable_surplus", "negative_prices", "dr_program_active",
    }
    assert o.regimes["grid_stress"].multipliers["reliability_risk"] == 2.4
    assert o.regimes["grid_stress"].risk_aversion == 0.6
    assert o.regimes["price_spike"].multipliers["net_cost"] == 1.9
    assert o.regimes["renewable_surplus"].multipliers["curtailment"] == 2.0
    # Every regime states a measured trigger the operator can read.
    assert all(r.trigger and r.label for r in o.regimes.values())


def test_eighty_action_clusters():
    assert N_CLUSTERS == 80
    assert len({(c.battery, c.market, c.curtailment, c.demand_response, c.load_shift)
                for c in CLUSTERS}) == 80
    assert [c.index for c in CLUSTERS] == list(range(80))


def test_weight_slider_renormalises_and_logs(config):
    updated = config.with_base_weights({"reliability_risk": 0.5}, note="operator slider")
    assert sum(updated.objectives.base_weights.values()) == pytest.approx(1.0)
    assert updated.objectives.base_weights["reliability_risk"] > \
        config.objectives.base_weights["reliability_risk"]
    assert len(updated.change_log) == 1 and "base_weights" in updated.change_log[0]
    # Original is untouched: config is immutable.
    assert config.change_log == ()


def test_unknown_objective_is_rejected(config):
    with pytest.raises(KeyError):
        config.with_base_weights({"profit": 1.0})


def test_horizon_consistency_is_enforced(tmp_path, config):
    import shutil
    for name in ("portfolio.yaml", "objectives.yaml", "market.yaml"):
        shutil.copy(config.source_dir / name, tmp_path / name)
    text = (tmp_path / "objectives.yaml").read_text().replace("steps: 16", "steps: 12")
    (tmp_path / "objectives.yaml").write_text(text)
    with pytest.raises(ValueError, match="horizon mismatch"):
        load_config(tmp_path)
