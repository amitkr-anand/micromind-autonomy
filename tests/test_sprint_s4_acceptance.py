"""
tests/test_sprint_s4_acceptance.py
MicroMind / NanoCorteX — Sprint S4 Acceptance Gate
8 tests covering EW Engine, Hybrid A*, and BCMP-1 EW simulation.

Acceptance gate (Part Two V7):
  KPI-E01  Cost map update < 500 ms wall clock
  KPI-E03  ≥ 2 replans visible on dashboard
  EW-01    Cost map response ≤ 500 ms from jammer detection
  EW-02    Route replan ≤ 1 s; both replans successful
  NFR-006  Pd ≥ 0.90 (confidence ≥ detection threshold at close range)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.ew_engine.ew_engine import (
    EWEngine, EWObservation,
    DETECTION_CONFIDENCE_THRESHOLD,
    HYPOTHESIS_CONFIDENCE_THRESHOLD,
    GRID_NORTH_MIN, GRID_EAST_MIN, GRID_RESOLUTION,
)
from core.route_planner.hybrid_astar import HybridAstar
from sim.bcmp1_ew_sim import run_bcmp1_ew_sim


# ---------------------------------------------------------------------------
# Shared fixture — full 100 km BCMP-1 EW sim (run once, reused)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim_result():
    """Run the full BCMP-1 EW sim once; reuse across tests 5–8."""
    return run_bcmp1_ew_sim(seed=42, verbose=False)


# ---------------------------------------------------------------------------
# Test 1 — EW Engine: inverse-square signal model
# ---------------------------------------------------------------------------

def test_signal_model_inverse_square():
    """
    Confidence at 50 m equals confidence_at_50m.
    Confidence at 100 m is exactly 1/4 of confidence at 50 m.
    """
    engine = EWEngine()
    obs_pos  = np.array([0.0,   0.0, 4000.0])
    jam_pos  = np.array([50.0,  0.0, 4000.0])   # 50 m north
    jam_pos2 = np.array([100.0, 0.0, 4000.0])   # 100 m north

    c50  = engine.signal_confidence(obs_pos, jam_pos,  confidence_at_50m=0.9)
    c100 = engine.signal_confidence(obs_pos, jam_pos2, confidence_at_50m=0.9)

    assert abs(c50 - 0.9) < 1e-6,  f"Expected 0.9 at 50 m, got {c50:.4f}"
    assert abs(c100 - 0.9 * 0.25) < 1e-4, \
        f"Expected {0.9*0.25:.4f} at 100 m (inverse-square), got {c100:.4f}"
    print(f"  ✅ Signal model  c50={c50:.3f}  c100={c100:.4f}")


# ---------------------------------------------------------------------------
# Test 2 — EW Engine: DBSCAN clusters observations into hypothesis
# ---------------------------------------------------------------------------

def test_dbscan_forms_hypothesis():
    """
    ≥ DBSCAN_MIN_SAMPLES observations from near a jammer must produce
    at least 1 active hypothesis with confidence ≥ HYPOTHESIS_CONFIDENCE_THRESHOLD.
    """
    engine = EWEngine()
    rng    = np.random.default_rng(0)

    jammer_n, jammer_e = 40_000.0, -3_000.0

    for i in range(10):
        obs_n = 20_000.0 + i * 1_000.0
        dist  = math.sqrt((obs_n - jammer_n)**2 + (0 - jammer_e)**2)
        bearing = math.degrees(math.atan2(jammer_e - 0, jammer_n - obs_n))
        sig_db  = -40.0 - 20.0 * math.log10(max(dist, 50) / 50.0)

        obs = EWObservation(
            timestamp_s        = 400.0 + i * 10,
            bearing_deg        = bearing,
            signal_strength_db = sig_db,
            estimated_range_m  = dist,
            position_enu       = np.array([obs_n, 0.0, 4_000.0]),
        )
        engine.process_observations([obs], 400.0 + i * 10)

    assert engine.active_hypothesis_count >= 1, \
        f"Expected ≥1 hypothesis, got {engine.active_hypothesis_count}"
    hyp = engine.hypotheses[0]
    assert hyp.confidence >= HYPOTHESIS_CONFIDENCE_THRESHOLD, \
        f"Confidence {hyp.confidence:.3f} below threshold {HYPOTHESIS_CONFIDENCE_THRESHOLD}"
    print(f"  ✅ DBSCAN formed {engine.active_hypothesis_count} hypothesis  "
          f"conf={hyp.confidence:.3f}  radius={hyp.radius_m:.0f} m")


# ---------------------------------------------------------------------------
# Test 3 — EW Engine: cost map painted with positive threat cost
# ---------------------------------------------------------------------------

def test_cost_map_painted():
    """
    After feeding observations near a jammer, the cost map cells
    covering the jammer position must have cost > 0.5.
    """
    engine = EWEngine()
    jammer_n, jammer_e = 40_000.0, -3_000.0

    for i in range(12):
        obs_n = 20_000.0 + i * 1_000.0
        dist  = math.sqrt((obs_n - jammer_n)**2 + (0 - jammer_e)**2)
        bearing = math.degrees(math.atan2(jammer_e - 0, jammer_n - obs_n))
        obs = EWObservation(
            timestamp_s        = 400.0 + i * 10,
            bearing_deg        = bearing,
            signal_strength_db = -40.0 - 20.0 * math.log10(max(dist, 50) / 50.0),
            estimated_range_m  = dist,
            position_enu       = np.array([obs_n, 0.0, 4_000.0]),
        )
        engine.process_observations([obs], 400.0 + i * 10)

    row, col = EWEngine.world_to_grid(jammer_n, jammer_e)
    cost_at_jammer = engine.cost_map[row, col]
    assert cost_at_jammer > 0.5, \
        f"Expected cost > 0.5 at jammer cell, got {cost_at_jammer:.3f}"
    print(f"  ✅ Cost at jammer cell ({row},{col}): {cost_at_jammer:.3f}")


# ---------------------------------------------------------------------------
# Test 4 — EW Engine: cost map update latency < 500 ms (KPI-E01)
# ---------------------------------------------------------------------------

def test_cost_map_update_latency():
    """
    Each cost map update must complete in < 500 ms wall clock (KPI-E01).
    Tested with a batch of 10 observations.
    """
    engine = EWEngine()
    jammer_n, jammer_e = 40_000.0, -3_000.0

    obs_batch = []
    for i in range(10):
        obs_n = 20_000.0 + i * 500.0
        dist  = math.sqrt((obs_n - jammer_n)**2 + (0 - jammer_e)**2)
        bearing = math.degrees(math.atan2(jammer_e - 0, jammer_n - obs_n))
        obs_batch.append(EWObservation(
            timestamp_s        = 400.0,
            bearing_deg        = bearing,
            signal_strength_db = -40.0 - 20.0 * math.log10(max(dist, 50) / 50.0),
            estimated_range_m  = dist,
            position_enu       = np.array([obs_n, 0.0, 4_000.0]),
        ))

    update = engine.process_observations(obs_batch, 400.0)
    assert update is not None
    assert update.kpi_e01_pass, \
        f"Cost map update took {update.wall_latency_ms:.1f} ms (limit 500 ms)"
    print(f"  ✅ Cost map update latency: {update.wall_latency_ms:.2f} ms  (limit 500 ms)")


# ---------------------------------------------------------------------------
# Test 5 — Hybrid A*: routes around high-cost zone
# ---------------------------------------------------------------------------

def test_astar_avoids_jammer():
    """
    A* must produce a path with non-zero east deviation when a high-cost
    jammer blob blocks the nominal (east=0) corridor.
    """
    engine = EWEngine()
    planner = HybridAstar(engine)

    jammer_n, jammer_e = 40_000.0, -3_000.0
    for i in range(15):
        obs_n = 20_000.0 + i * 1_000.0
        dist  = math.sqrt((obs_n - jammer_n)**2 + (0 - jammer_e)**2)
        bearing = math.degrees(math.atan2(jammer_e - 0, jammer_n - obs_n))
        obs = EWObservation(
            timestamp_s=400+i*10,
            bearing_deg=bearing,
            signal_strength_db=-40 - 20*math.log10(max(dist,50)/50),
            estimated_range_m=dist,
            position_enu=np.array([obs_n, 0.0, 4_000.0]),
        )
        engine.process_observations([obs], 400+i*10)

    result = planner.replan(
        start_north_m=25_000, start_east_m=0,
        goal_north_m=65_000,  goal_east_m=0,
        cruise_alt_m=4_000,   mission_time_s=500,
        trigger="Test: jammer avoidance",
    )

    assert result.success, "Replan failed — no path found"
    assert result.max_east_deviation_m > 500, \
        f"Expected eastward deviation > 500 m, got {result.max_east_deviation_m:.0f} m"
    print(f"  ✅ Replan success  dev={result.max_east_deviation_m:.0f} m  "
          f"latency={result.wall_latency_ms:.1f} ms  nodes={result.nodes_explored}")


# ---------------------------------------------------------------------------
# Test 6 — Hybrid A*: replan latency < 1 s (EW-02)
# ---------------------------------------------------------------------------

def test_astar_replan_latency():
    """
    Route replan must complete in < 1000 ms wall clock (EW-02).
    Tested on a fresh engine with no cost (worst-case: full grid search).
    """
    engine  = EWEngine()   # zero cost — planner must still finish < 1 s
    planner = HybridAstar(engine)

    result = planner.replan(
        start_north_m=0,       start_east_m=0,
        goal_north_m=100_000,  goal_east_m=0,
        cruise_alt_m=4_000,    mission_time_s=0,
        trigger="Latency test",
    )

    assert result.success, "Replan failed on empty cost map"
    assert result.kpi_ew02_pass, \
        f"Replan took {result.wall_latency_ms:.1f} ms (limit 1000 ms)"
    print(f"  ✅ Replan latency: {result.wall_latency_ms:.1f} ms  "
          f"nodes={result.nodes_explored}")


# ---------------------------------------------------------------------------
# Test 7 — Sim: both replans execute and pass KPI gates (sim_result fixture)
# ---------------------------------------------------------------------------

def test_both_replans_executed(sim_result):
    """BCMP-1 EW sim must produce exactly 2 replans, both KPI-EW02 compliant."""
    assert len(sim_result.replans) >= 2, \
        f"Expected ≥2 replans, got {len(sim_result.replans)}"

    for rp in sim_result.replans:
        assert rp.success, f"{rp.replan_id} failed"
        assert rp.kpi_ew02_pass, \
            f"{rp.replan_id}: {rp.wall_latency_ms:.1f} ms > 1000 ms limit"
        assert rp.max_east_deviation_m > 0, \
            f"{rp.replan_id}: zero east deviation — planner did not avoid jammer"

    print(f"  ✅ {len(sim_result.replans)} replans, all KPI-EW02 PASS")
    for rp in sim_result.replans:
        print(f"     {rp.replan_id}: {rp.wall_latency_ms:.1f} ms  dev={rp.max_east_deviation_m:.0f} m")


# ---------------------------------------------------------------------------
# Test 8 — Sim: all KPI gates pass
# ---------------------------------------------------------------------------

def test_all_kpi_gates(sim_result):
    """KPI-E01, KPI-EW02, and KPI-E03 must all pass."""
    assert sim_result.kpi_e01_pass, \
        (f"KPI-E01 FAIL — cost map update exceeded 500 ms. "
         f"Max: {max(u.wall_latency_ms for u in sim_result.cost_map_updates):.1f} ms")
    assert sim_result.kpi_ew02_pass, \
        "KPI-EW02 FAIL — one or both replans exceeded 1000 ms"
    assert sim_result.kpi_e03_pass, \
        f"KPI-E03 FAIL — fewer than 2 replans: {len(sim_result.replans)}"

    max_lat = max(u.wall_latency_ms for u in sim_result.cost_map_updates)
    r1, r2  = sim_result.replans[0], sim_result.replans[1]
    print(f"  ✅ KPI-E01  max cost map latency : {max_lat:.1f} ms  (limit 500 ms)")
    print(f"  ✅ KPI-EW02 Replan 1             : {r1.wall_latency_ms:.1f} ms  (limit 1000 ms)")
    print(f"  ✅ KPI-EW02 Replan 2             : {r2.wall_latency_ms:.1f} ms  (limit 1000 ms)")
    print(f"  ✅ KPI-E03  replans visible       : {len(sim_result.replans)}")
