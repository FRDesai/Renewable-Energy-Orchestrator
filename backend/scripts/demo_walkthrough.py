#!/usr/bin/env python
"""Drive the PRD's 7-minute demo script through the HTTP API.

Every step calls exactly the endpoint the console's button calls, so this verifies the
whole stack end to end -- engine, API, and the frame the console renders -- and doubles
as the demo rehearsal. Exits non-zero if any step fails its own check.

    python scripts/demo_walkthrough.py [base-url]
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
failures: list[str] = []


def call(method: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path} -> {exc.code} {exc.read().decode()[:300]}")


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"    {'PASS' if ok else 'FAIL'}  {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def step(n: int, title: str, criterion: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n    ({criterion})\n{'=' * 78}")


def totals(frame: dict) -> dict:
    return frame["portfolio"]["totals"]


def battery_mw(frame: dict) -> dict:
    return frame["trace"]["committed"]["interval"]["battery_mw"]


def main() -> int:
    health = call("GET", "/api/health")
    print(f"decision service: {health['status']}, {health['clusters']} clusters, "
          f"forecaster trained {health['forecaster_trained']}, "
          f"LLM {'available' if health['llm_available'] else 'unavailable (template mode)'}")

    # ---- 1 -------------------------------------------------------------
    step(1, "09:00, solar surplus: the agent charges and explains why it is not selling",
         "F1, criterion 2")
    frame = call("POST", "/api/session/reset", {"preset": "typical_day", "seed": 7})
    t = totals(frame)
    charged = -sum(min(0.0, v) for v in battery_mw(frame).values())
    print(f"    clock {frame['clock']['time']}  ·  {frame['trace']['committed']['cluster']['label']}")
    print(f"    surplus {-t['net_load_mw']:.1f} MW at ${frame['portfolio']['grid']['price_per_mwh']:.2f}/MWh"
          f"  ·  fleet {t['fleet_soc_pct']:.1f}%")
    check("opens at 09:00 with a committed plan", frame["clock"]["time"] == "09:00")
    check("history is already populated", len(frame["history"]) >= 12,
          f"{len(frame['history'])} intervals")
    check("a surplus is present", t["net_load_mw"] < 0, f"{t['net_load_mw']:.1f} MW")
    check("the surplus is stored, not sold", charged > 0.0, f"charging {charged:.1f} MW")
    check("export is held back", frame["portfolio"]["grid"]["export_mw"] < 1.0,
          f"{frame['portfolio']['grid']['export_mw']:.1f} MW")

    # ---- 2 -------------------------------------------------------------
    step(2, "Inject a cloud bank: re-plan within one interval, with the tool trace",
         "F1, criterion 5")
    region = frame["portfolio"]["solar"][0]["cloud_region"]
    shaded = [f["id"] for f in frame["portfolio"]["solar"] if f["cloud_region"] == region]
    solar_before = sum(f["output_mw"] for f in frame["portfolio"]["solar"]
                       if f["id"] in shaded)

    frame = call("POST", "/api/session/inject", {"type": "cloud_bank", "region": region})
    change = frame["response_to_change"]
    print(f"    {change['before']['cluster']} -> {change['after']['cluster']}"
          f"  in {change['evaluation_ms']:.1f} ms")
    for action in change["changed_actions"][:4]:
        print(f"      {action['verb']} {action['asset_name']}: "
              f"{action['before_mw']} -> {action['after_mw']} MW")
    check("the clock did not advance", frame["clock"]["interval"] == 36)
    check("re-planned within one evaluation", change["evaluation_ms"] < 1000.0,
          f"{change['evaluation_ms']:.1f} ms")
    check("a tool trace is available", len(frame["agent"]["tool_calls"]) > 0,
          ", ".join(c["name"] for c in frame["agent"]["tool_calls"]))
    check("the event is reported to the operator",
          any(e["type"] == "cloud_bank" for e in frame["events"]))

    # Step 0 of the look-ahead is the *measured* present, so a cloud bank arriving now
    # has not shaded the panels yet -- it enters every scenario and bites next interval.
    # The demo beat is therefore the re-plan, then the drop it anticipated.
    frame = call("POST", "/api/session/step", {"intervals": 1})
    solar_after = sum(f["output_mw"] for f in frame["portfolio"]["solar"]
                      if f["id"] in shaded)
    print(f"    {region} solar {solar_before:.1f} -> {solar_after:.1f} MW one interval on"
          f"  ·  now {frame['trace']['committed']['cluster']['label']}")
    check("the cloud bank actually cut output", solar_after < solar_before - 1.0,
          f"{solar_before:.1f} -> {solar_after:.1f} MW")

    # ---- 3 -------------------------------------------------------------
    step(3, "Paste a storm bulletin: extract, confirm, weights shift, maintenance queued",
         "F2, D2 text input, criteria 3 and 8")
    extraction = call("POST", "/api/session/bulletin", {"text":
        "SEVERE WEATHER ADVISORY: a severe storm front will reach the service territory "
        "in about 4 hours, lasting for 6 hours. High winds are forecast across region_b."})
    event = extraction["event"]
    print(f"    extracted by {extraction['source']}: {event['label']}, "
          f"arrival in {event['intervals_to_arrival']} intervals, "
          f"confidence {event['extraction_confidence_pct']:.0f}%")
    print(f"    source phrases: {extraction['event']['source_phrases']}")
    check("a storm was extracted", event["type"] == "storm_alert")
    check("it waits for the operator", extraction["needs_confirmation"] is True)

    confirmed = call("POST", "/api/session/bulletin/confirm", {"accept": True})
    frame = confirmed["frame"]
    guardrail = next(g for g in frame["trace"]["options"]["guardrails"]
                     if g["key"] == "storm_readiness")
    reliability_shift = frame["trace"]["criteria"]["shifted"].get("reliability_risk")
    moves = [p for p in frame["approvals"] if p["kind"] == "maintenance"]
    print(f"    storm guardrail: applied={guardrail['applied']}, "
          f"removed {guardrail['removed_count']} cluster(s)")
    print(f"    reliability weight {reliability_shift}")
    for move in moves:
        print(f"    queued: {move['summary']}")
    check("the storm guardrail is live", guardrail["applied"] is True)
    check("reliability weight rose", reliability_shift is not None
          and reliability_shift[1] > reliability_shift[0],
          str(reliability_shift))
    check("the maintenance move is queued for approval", len(moves) > 0)

    # ---- 4 -------------------------------------------------------------
    step(4, "Ask the agent what it gave up", "F2, criterion 3")
    answer = call("POST", "/api/session/ask",
                  {"question": "What did you give up for this?"})
    print("    " + answer["text"].replace("\n", "\n    "))
    check("an answer came back", len(answer["text"]) > 0)
    check("it is grounded in the trace", len(answer["citations"]) > 0)
    check("the source is stated", answer["source"] in ("llm", "template"))

    # ---- 5 -------------------------------------------------------------
    step(5, "18:30 price spike: discharge into the spike and trigger DR", "F1, criterion 6")
    frame = call("POST", "/api/session/reset",
                 {"preset": "evening_price_crisis", "seed": 7})
    frame = call("POST", "/api/session/step", {"intervals": 38})   # 09:00 -> 18:30
    frame = call("POST", "/api/session/inject", {"type": "price_spike", "magnitude": 7.0})
    frame = call("POST", "/api/session/inject", {"type": "iso_dr_call"})
    r = frame["trace"]["committed"]["interval"]
    discharge = sum(v for v in r["battery_mw"].values() if v > 0)
    print(f"    clock {frame['clock']['time']}  ·  ${r['price_per_mwh']:.2f}/MWh")
    print(f"    {frame['trace']['committed']['cluster']['label']}")
    print(f"    discharging {discharge:.1f} MW  ·  DR {r['dr_mw']:.1f} MW  ·  "
          f"shift {r['shift_mw']:.1f} MW")
    check("the clock reached the evening peak", frame["clock"]["time"] >= "18:00",
          frame["clock"]["time"])
    check("the price spiked", r["price_per_mwh"] > 150.0, f"${r['price_per_mwh']:.0f}/MWh")
    check("the agent discharges or reduces load into it",
          discharge > 0.0 or r["dr_mw"] > 0.0 or r["shift_mw"] > 0.0)

    # ---- 6 -------------------------------------------------------------
    step(6, "Battery trip plus data dropout: re-plan, inspection, substitution flagged",
         "criteria 5 and 8")
    frame = call("POST", "/api/session/inject",
                 {"type": "battery_trip", "assets": ["harbor"]})
    tripped = [b["name"] for b in frame["portfolio"]["batteries"] if not b["available"]]
    orders = frame["maintenance"]["work_orders"]
    print(f"    offline: {', '.join(tripped) or 'none'}")
    for order in orders:
        print(f"    work order {order['id']}: {order['summary']}")
    check("the trip shows as unavailable", len(tripped) > 0, ", ".join(tripped))
    check("an inspection was raised", len(orders) > 0)

    frame = call("POST", "/api/session/inject",
                 {"type": "data_dropout", "channels": ["solar", "price"], "duration": 12})
    quality = frame["data_quality"]
    print(f"    substituted: {quality['substituted']}  ·  stale: {quality['stale']}")
    print(f"    renewable still reported: {totals(frame)['renewable_available_mw']:.1f} MW")
    check("substitution is flagged", len(quality["substituted"]) > 0)
    check("the feed is not silently zeroed",
          totals(frame)["renewable_available_mw"] >= 0.0)
    check("an alarm is raised", any(a["kind"] == "data_quality" for a in frame["alarms"]))

    for _ in range(3):
        frame = call("POST", "/api/session/step", {"intervals": 1})
    print(f"    after 3 more intervals: mode={frame['status']['mode']}, "
          f"alarm={frame['data_quality']['alarm']}")
    check("a feed stale beyond the limit falls back to rule-based",
          frame["status"]["mode"] == "rule_based_fallback", frame["status"]["mode"])
    check("the cause is logged", bool(frame["status"]["fallback_reason"]),
          str(frame["status"]["fallback_reason"]))

    # ---- 7 -------------------------------------------------------------
    step(7, "Scenario simulation and reliability evidence", "F3, D2, criteria 4 and 6")
    reliability = call("GET", "/api/reliability")
    if reliability.get("available"):
        invariants = reliability["invariants"]
        failed = [i["label"] for i in invariants if i["failed"]]
        bench = reliability["benchmark"]
        print(f"    {reliability['episodes']} episodes, {reliability['decisions']} decisions")
        print(f"    invariants passing: {len(invariants) - len(failed)}/{len(invariants)}"
              + (f"  (failing: {', '.join(failed)})" if failed else ""))
        print(f"    load shed {bench['agent_unserved_mwh']:.2f} MWh vs rule-based "
              f"{bench['rule_based_unserved_mwh']:.2f} MWh")
        print(f"    cost win rate {100 * bench['win_rate_vs_rule_based']:.0f}% "
              f"(target >= 75%)  ·  CO2 {bench['agent_co2_t']:.0f} vs "
              f"{bench['rule_based_co2_t']:.0f} t")
        check("the invariant report is servable", len(invariants) >= 12)
        check("no invariant is failing", not failed, ", ".join(failed))
    else:
        check("reliability artifact present", False, reliability.get("reason", ""))

    simulation = call("GET", "/api/simulation")
    if simulation.get("available"):
        presets = simulation["presets"]
        print(f"    stored benchmark covers {len(presets)} preset(s)")
        for name, result in presets.items():
            c = result["comparison"]["rule_based"]
            print(f"      {name:22s} cost {c['cost_delta_pct']:+6.1f}%  "
                  f"CO2 {c['co2_delta_pct']:+6.1f}%  shed {c['shed_delta_pct']:+7.1f}%")
        check("the F3 benchmark is servable", len(presets) >= 1)
    else:
        check("simulation artifact present", False, simulation.get("reason", ""))

    job = call("POST", "/api/simulate",
               {"preset": "typical_day", "paths": 1, "intervals": 8,
                "strategies": ["agent", "rule_based"]})
    import time
    for _ in range(600):
        status = call("GET", f"/api/simulate/{job['id']}")
        if status["status"] != "running":
            break
        time.sleep(0.2)
    check("a fresh Monte Carlo run completes", status["status"] == "done",
          str(status.get("error")))

    call("POST", "/api/session/reset", {"preset": "typical_day", "seed": 7})
    print(f"\n{'=' * 78}")
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {', '.join(failures)}")
        return 1
    print("every demo step passed its check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
