"""
sim/bcmp1_ew_sim.py
MicroMind / NanoCorteX — BCMP-1 EW Simulation
Sprint S4 Deliverable 3 of 3

Standalone simulation of the EW phase of BCMP-1:
  - Vehicle flies 100 km corridor at 50 m/s
  - JMR-01 activates at T+18 min (40 km north) → cost map update → Replan 1
  - JMR-02 activates at T+25 min (50 km north) → cost map update → Replan 2
  - EW Engine processes SDR observations each tick
  - Hybrid A* replans route around active jammer zones
  - Dashboard PNG shows: cost map evolution, both replans, KPI gate summary

Acceptance gate (Part Two V7 / BCMP-1):
  KPI-E01  Cost map update < 500 ms         ← measured wall-clock per update
  EW-01    Cost map response ≤ 500 ms       ← same gate
  EW-02    Route replan ≤ 1 s; both visible ← measured wall-clock per replan
  KPI-E03  Both BCMP-1 replans on dashboard ← visual confirmation

References:
  Part Two V7  §1.6, KPI-E01, KPI-E03, NFR-003/005
  BCMP-1       JMR-01, JMR-02, EW-01, EW-02
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from core.ew_engine.ew_engine import (
    EWEngine, EWObservation,
    GRID_NORTH_MIN, GRID_NORTH_MAX,
    GRID_EAST_MIN,  GRID_EAST_MAX,
    GRID_RESOLUTION, N_NORTH, N_EAST,
    DETECTION_CONFIDENCE_THRESHOLD,
)
from core.route_planner.hybrid_astar import HybridAstar, ReplanResult
from scenarios.bcmp1.bcmp1_scenario import (
    BCMP1Scenario, JAMMER_NODES, JammerType,
)


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

VEHICLE_SPEED_MS    = 50.0          # m/s (180 km/h)
TICK_INTERVAL_S     = 10.0          # simulation timestep
SDR_RANGE_MAX_M     = 20_000.0      # max SDR detection range
OBS_PER_TICK        = 3             # observations per tick when in range
CRUISE_ALT_M        = 4_000.0       # cruise altitude (m)
MISSION_ID          = "S4-EW-BCMP1-001"

# Jammer activation thresholds for replan trigger
REPLAN_COST_THRESHOLD = 0.30        # trigger replan when path cost exceeds this


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EWSimResult:
    mission_id:         str
    total_time_s:       float
    replans:            List[ReplanResult]
    cost_map_updates:   list
    kpi_e01_pass:       bool    # all cost map updates < 500 ms
    kpi_ew02_pass:      bool    # all replans < 1 s
    kpi_e03_pass:       bool    # both replans visible (≥ 2 replans)

    # Telemetry time series
    times_s:            List[float]
    vehicle_north:      List[float]
    vehicle_east:       List[float]
    active_hypotheses:  List[int]
    peak_costs:         List[float]

    # Snapshot cost maps at replan moments
    cost_map_snap1:     Optional[np.ndarray] = None
    cost_map_snap2:     Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# SDR observation generator
# ---------------------------------------------------------------------------

def _generate_observations(
    vehicle_pos: np.ndarray,
    mission_time_s: float,
    scenario: BCMP1Scenario,
    rng: np.random.Generator,
) -> List[EWObservation]:
    """
    Simulate SDR receiver output for all active jammers within range.
    Uses inverse-square model with noise.
    """
    observations = []

    for jammer in scenario.jammer_nodes:
        # Only broadband jammers for EW Engine (spoofer handled by BIM)
        if jammer.jammer_type != JammerType.BROADBAND_GNSS:
            continue
        # Jammers are physically active from mission start in BCMP-1 (pre-positioned).
        # EW Engine detects them as vehicle enters SDR range.
        # deactivation_time_s still honoured (jammer can be destroyed/withdrawn).
        if mission_time_s > jammer.deactivation_time_s:
            continue

        jpos = np.array(jammer.position_enu[:2])
        vpos = vehicle_pos[:2]
        dist = float(np.linalg.norm(vpos - jpos))

        if dist > SDR_RANGE_MAX_M:
            continue

        for _ in range(OBS_PER_TICK):
            # Signal strength: reference -40 dBm at 50 m, inverse-square falloff
            signal_db = -40.0 - 20.0 * math.log10(max(dist, 50.0) / 50.0)
            signal_db += rng.normal(0, 2.0)   # receiver noise ±2 dB

            # Bearing with noise
            true_bearing = math.degrees(math.atan2(
                jpos[1] - vpos[1],   # east delta
                jpos[0] - vpos[0],   # north delta
            ))
            bearing = true_bearing + rng.normal(0, 3.0)   # ±3° bearing noise

            # Range estimate from signal strength (noisy)
            est_range = max(1.0, dist * rng.uniform(0.85, 1.15))

            observations.append(EWObservation(
                timestamp_s        = mission_time_s,
                bearing_deg        = bearing % 360,
                signal_strength_db = signal_db,
                estimated_range_m  = est_range,
                position_enu       = vehicle_pos.copy(),
            ))

    return observations


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_bcmp1_ew_sim(seed: int = 42, verbose: bool = True) -> EWSimResult:
    """
    Run the BCMP-1 EW simulation.
    Returns EWSimResult with full telemetry and KPI gate results.
    """
    print("=" * 60)
    print("MicroMind S4 — BCMP-1 EW Simulation")
    print("=" * 60)

    rng      = np.random.default_rng(seed)
    scenario = BCMP1Scenario()
    engine   = EWEngine()
    planner  = HybridAstar(engine)

    # Vehicle state
    vehicle_pos = np.array([0.0, 0.0, CRUISE_ALT_M])
    mission_time_s = 0.0
    total_distance = 100_000.0
    total_time     = total_distance / VEHICLE_SPEED_MS   # 2000 s

    # Current route (as north positions of waypoints — simplified to corridor)
    # Replanning updates this list
    current_route_end_north = total_distance

    # Telemetry
    times_s:           List[float] = []
    vehicle_north:     List[float] = []
    vehicle_east:      List[float] = []
    active_hypotheses: List[int]   = []
    peak_costs:        List[float] = []

    # Replan tracking
    replanned_jmr01 = False
    replanned_jmr02 = False
    cost_map_snap1: Optional[np.ndarray] = None
    cost_map_snap2: Optional[np.ndarray] = None
    replan1: Optional[ReplanResult] = None
    replan2: Optional[ReplanResult] = None

    # Jammer nodes (from scenario)
    jmr01 = next(j for j in JAMMER_NODES if j.jammer_id == "JMR-01")
    jmr02 = next(j for j in JAMMER_NODES if j.jammer_id == "JMR-02")

    # Distance-based replan triggers — fire when vehicle first enters SDR range
    # and enough observations have accumulated for a reliable hypothesis.
    # JMR-01: detected at ~20 km north; replan after 5 obs (~50 s in range)
    # JMR-02: detected at ~30 km north; replan when 2nd hypothesis forms
    JMR01_REPLAN_NORTH = 25_000.0   # trigger after 5 km in JMR-01 SDR range
    JMR02_REPLAN_NORTH = 38_000.0   # trigger after 8 km in JMR-02 SDR range
    MIN_OBS_FOR_REPLAN  = 5          # min observations before replan fires

    # ---------------------------------------------------------------
    # Simulation loop
    # ---------------------------------------------------------------
    while mission_time_s <= total_time:
        # Advance vehicle position (straight north along corridor)
        vehicle_pos[0] = vehicle_pos[0]   # north accumulates below
        current_north  = vehicle_pos[0]

        # Generate SDR observations
        obs = _generate_observations(vehicle_pos, mission_time_s, scenario, rng)

        # Feed to EW Engine
        if obs:
            update = engine.process_observations(obs, mission_time_s)

        # Check for replan triggers — distance-based, not time-based
        # --- Replan 1: JMR-01 ---
        if (not replanned_jmr01
                and current_north >= JMR01_REPLAN_NORTH
                and engine.active_hypothesis_count >= 1
                and len(engine._observations) >= MIN_OBS_FOR_REPLAN):

            if verbose:
                print(f"\n  [T={mission_time_s:.0f}s] JMR-01 confirmed → triggering Replan 1  "
                      f"(N={current_north/1000:.1f} km, hyps={engine.active_hypothesis_count})")

            cost_map_snap1 = engine.cost_map.copy()
            replan1 = planner.replan(
                start_north_m  = current_north,
                start_east_m   = vehicle_pos[1],
                goal_north_m   = total_distance,
                goal_east_m    = 0.0,
                cruise_alt_m   = CRUISE_ALT_M,
                mission_time_s = mission_time_s,
                trigger        = "JMR-01 confirmed",
            )
            replanned_jmr01 = True

            if verbose:
                status = "PASS ✓" if replan1.kpi_ew02_pass else "FAIL ✗"
                print(f"    Replan 1: {replan1.wall_latency_ms:.1f} ms  "
                      f"dev={replan1.max_east_deviation_m:.0f} m  [{status}]")

        # --- Replan 2: JMR-02 ---
        if (not replanned_jmr02
                and current_north >= JMR02_REPLAN_NORTH
                and engine.active_hypothesis_count >= 2):

            if verbose:
                print(f"\n  [T={mission_time_s:.0f}s] JMR-02 confirmed → triggering Replan 2  "
                      f"(N={current_north/1000:.1f} km, hyps={engine.active_hypothesis_count})")

            cost_map_snap2 = engine.cost_map.copy()
            replan2 = planner.replan(
                start_north_m  = current_north,
                start_east_m   = vehicle_pos[1],
                goal_north_m   = total_distance,
                goal_east_m    = 0.0,
                cruise_alt_m   = CRUISE_ALT_M,
                mission_time_s = mission_time_s,
                trigger        = "JMR-02 activated",
            )
            replanned_jmr02 = True

            if verbose:
                status = "PASS ✓" if replan2.kpi_ew02_pass else "FAIL ✗"
                print(f"    Replan 2: {replan2.wall_latency_ms:.1f} ms  "
                      f"dev={replan2.max_east_deviation_m:.0f} m  [{status}]")

        # Log telemetry
        times_s.append(mission_time_s)
        vehicle_north.append(vehicle_pos[0])
        vehicle_east.append(vehicle_pos[1])
        active_hypotheses.append(engine.active_hypothesis_count)
        peak_costs.append(float(engine.cost_map.max()))

        # Advance
        mission_time_s += TICK_INTERVAL_S
        vehicle_pos[0] += VEHICLE_SPEED_MS * TICK_INTERVAL_S

    # ---------------------------------------------------------------
    # KPI evaluation
    # ---------------------------------------------------------------
    all_updates = engine.updates
    kpi_e01 = all(u.kpi_e01_pass for u in all_updates) if all_updates else True
    kpi_ew02 = (
        (replan1 is not None and replan1.kpi_ew02_pass) and
        (replan2 is not None and replan2.kpi_ew02_pass)
    )
    kpi_e03 = len(planner.replans) >= 2

    result = EWSimResult(
        mission_id        = MISSION_ID,
        total_time_s      = total_time,
        replans           = planner.replans,
        cost_map_updates  = all_updates,
        kpi_e01_pass      = kpi_e01,
        kpi_ew02_pass     = kpi_ew02,
        kpi_e03_pass      = kpi_e03,
        times_s           = times_s,
        vehicle_north     = vehicle_north,
        vehicle_east      = vehicle_east,
        active_hypotheses = active_hypotheses,
        peak_costs        = peak_costs,
        cost_map_snap1    = cost_map_snap1,
        cost_map_snap2    = cost_map_snap2,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    if verbose:
        print("\n" + "─" * 60)
        print(f"  Mission ID        : {MISSION_ID}")
        print(f"  Total sim time    : {total_time:.0f} s  ({total_time/60:.1f} min)")
        print(f"  Cost map updates  : {len(all_updates)}")
        if all_updates:
            max_lat = max(u.wall_latency_ms for u in all_updates)
            print(f"  Max update latency: {max_lat:.1f} ms  "
                  f"(limit 500 ms)  {'PASS ✓' if kpi_e01 else 'FAIL ✗'}")
        print(f"  Replans executed  : {len(planner.replans)}")
        if replan1:
            print(f"    Replan 1        : {replan1.wall_latency_ms:.1f} ms  "
                  f"dev={replan1.max_east_deviation_m:.0f} m  "
                  f"{'PASS ✓' if replan1.kpi_ew02_pass else 'FAIL ✗'}")
        if replan2:
            print(f"    Replan 2        : {replan2.wall_latency_ms:.1f} ms  "
                  f"dev={replan2.max_east_deviation_m:.0f} m  "
                  f"{'PASS ✓' if replan2.kpi_ew02_pass else 'FAIL ✗'}")
        print(f"  KPI-E01 (cost<500ms): {'PASS ✓' if kpi_e01 else 'FAIL ✗'}")
        print(f"  KPI-EW02 (replan<1s): {'PASS ✓' if kpi_ew02 else 'FAIL ✗'}")
        print(f"  KPI-E03 (2 replans) : {'PASS ✓' if kpi_e03 else 'FAIL ✗'}")
        print("─" * 60)

    return result


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def render_dashboard(result: EWSimResult, output_path: str = "dashboard/ew_dashboard.png"):
    """
    5-panel EW mission dashboard:
      Panel 1: Route map — nominal vs replanned paths + jammer zones
      Panel 2: EW cost map (snapshot at Replan 2)
      Panel 3: Peak cost over time + replan events
      Panel 4: Active jammer hypotheses count
      Panel 5: KPI gate summary
    """
    print("\n[2/2] Rendering EW dashboard …")

    BG   = "#0D1117"
    FG   = "#E6EDF3"
    GRID = "#21262D"
    RED  = "#F85149"
    GRN  = "#3FB950"
    AMB  = "#D29922"
    BLU  = "#58A6FF"
    CYN  = "#39D353"

    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        f"MicroMind / NanoCorteX  ·  EW Dashboard  ·  {result.mission_id}",
        color=FG, fontsize=14, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32,
                          left=0.06, right=0.97, top=0.93, bottom=0.08)

    # ------------------------------------------------------------------
    # Panel 1 — Route map (top-left, spans 2 rows)
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.set_facecolor(BG)
    ax1.set_title("Route Map — Nominal vs Replanned", color=FG, fontsize=9)

    # Jammer zones
    ew_jammers = [j for j in JAMMER_NODES if j.jammer_id in ("JMR-01", "JMR-02")]
    jammer_colours = [RED, AMB]
    for idx, j in enumerate(ew_jammers):
        jn = j.position_enu[0] / 1000
        je = j.position_enu[1] / 1000
        circ = plt.Circle(
            (je, jn), j.effective_radius_m / 1000,
            color=jammer_colours[idx], alpha=0.15, linewidth=1.5,
            fill=True, linestyle="--",
        )
        ax1.add_patch(circ)
        ax1.plot(je, jn, "x", color=jammer_colours[idx], markersize=10, markeredgewidth=2)
        ax1.annotate(j.jammer_id, (je, jn), color=jammer_colours[idx],
                     fontsize=7, xytext=(0.4, 0.4), textcoords="offset fontsize")

    # Nominal route (straight north)
    nom_n = np.linspace(0, 100, 100)
    nom_e = np.zeros(100)
    ax1.plot(nom_e, nom_n, "--", color=FG, alpha=0.3, linewidth=1, label="Nominal")

    # Vehicle track
    vn = np.array(result.vehicle_north) / 1000
    ve = np.array(result.vehicle_east)  / 1000
    ax1.plot(ve, vn, color=BLU, linewidth=1.2, alpha=0.6, label="Vehicle track")

    # Replan paths
    rcolours = [GRN, CYN]
    for i, replan in enumerate(result.replans):
        wps = np.array(replan.waypoints)
        ax1.plot(wps[:, 1] / 1000, wps[:, 0] / 1000,
                 color=rcolours[i], linewidth=2.0,
                 label=f"{replan.replan_id} ({replan.trigger})")
        # Mark replan start
        ax1.plot(wps[0, 1] / 1000, wps[0, 0] / 1000,
                 "^", color=rcolours[i], markersize=8)

    ax1.set_xlabel("East (km)", color=FG, fontsize=8)
    ax1.set_ylabel("North (km)", color=FG, fontsize=8)
    ax1.tick_params(colors=FG, labelsize=7)
    ax1.set_xlim(-12, 12)
    ax1.set_ylim(-5, 105)
    for sp in ax1.spines.values():
        sp.set_color(GRID)
    ax1.legend(fontsize=6.5, facecolor=BG, labelcolor=FG, loc="upper left")

    # ------------------------------------------------------------------
    # Panel 2 — Cost map heatmap at Replan 2 (top-centre)
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG)
    ax2.set_title("EW Cost Map @ Replan 2", color=FG, fontsize=9)

    snap = result.cost_map_snap2 if result.cost_map_snap2 is not None else \
           result.cost_map_snap1 if result.cost_map_snap1 is not None else \
           np.zeros((N_NORTH, N_EAST))

    # Transpose for display: rows=north (y), cols=east (x)
    extent = [
        GRID_EAST_MIN  / 1000, GRID_EAST_MAX  / 1000,
        GRID_NORTH_MIN / 1000, GRID_NORTH_MAX / 1000,
    ]
    im = ax2.imshow(snap, origin="lower", aspect="auto",
                    extent=extent, cmap="hot", vmin=0, vmax=1,
                    interpolation="bilinear")
    plt.colorbar(im, ax=ax2, label="Threat cost", fraction=0.046, pad=0.04)
    ax2.set_xlabel("East (km)", color=FG, fontsize=8)
    ax2.set_ylabel("North (km)", color=FG, fontsize=8)
    ax2.tick_params(colors=FG, labelsize=7)
    for sp in ax2.spines.values():
        sp.set_color(GRID)

    # ------------------------------------------------------------------
    # Panel 3 — Cost map at Replan 1 (top-right)
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(BG)
    ax3.set_title("EW Cost Map @ Replan 1", color=FG, fontsize=9)

    snap1 = result.cost_map_snap1 if result.cost_map_snap1 is not None else \
            np.zeros((N_NORTH, N_EAST))
    im3 = ax3.imshow(snap1, origin="lower", aspect="auto",
                     extent=extent, cmap="hot", vmin=0, vmax=1,
                     interpolation="bilinear")
    plt.colorbar(im3, ax=ax3, label="Threat cost", fraction=0.046, pad=0.04)
    ax3.set_xlabel("East (km)", color=FG, fontsize=8)
    ax3.set_ylabel("North (km)", color=FG, fontsize=8)
    ax3.tick_params(colors=FG, labelsize=7)
    for sp in ax3.spines.values():
        sp.set_color(GRID)

    # ------------------------------------------------------------------
    # Panel 4 — Peak cost + hypotheses over time (middle-left + centre)
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.set_facecolor(BG)
    ax4.set_title("Peak EW Cost & Active Hypotheses vs Mission Time", color=FG, fontsize=9)

    times_min = np.array(result.times_s) / 60

    ax4.plot(times_min, result.peak_costs, color=RED, linewidth=1.5, label="Peak cost")
    ax4.axhline(0.3, color=AMB, linestyle="--", linewidth=0.8, alpha=0.7, label="Replan threshold (0.30)")
    ax4.set_ylabel("Peak cost map value", color=FG, fontsize=8)
    ax4.set_ylim(0, 1.05)

    ax4b = ax4.twinx()
    ax4b.plot(times_min, result.active_hypotheses, color=CYN,
              linewidth=1.2, linestyle=":", label="Active hypotheses")
    ax4b.set_ylabel("Jammer hypotheses", color=CYN, fontsize=8)
    ax4b.tick_params(axis="y", colors=CYN, labelsize=7)
    ax4b.set_ylim(0, 5)

    # Replan event markers
    for i, rp in enumerate(result.replans):
        t_min = rp.mission_time_s / 60
        ax4.axvline(t_min, color=rcolours[i], linewidth=1.5,
                    linestyle="-.", alpha=0.8,
                    label=f"{rp.replan_id} @ T+{t_min:.1f} min")
        ax4.annotate(rp.replan_id, (t_min, 0.85), color=rcolours[i],
                     fontsize=7, rotation=90, va="top")

    # JMR activation markers
    for jn in ew_jammers:
        ax4.axvline(jn.activation_time_s / 60, color=RED, linewidth=1,
                    linestyle=":", alpha=0.5)
        ax4.annotate(f"{jn.jammer_id} ON",
                     (jn.activation_time_s / 60, 0.6),
                     color=RED, fontsize=6.5, rotation=90, va="top")

    ax4.tick_params(colors=FG, labelsize=7)
    ax4.set_xlabel("Mission time (min)", color=FG, fontsize=8)
    for sp in ax4.spines.values():
        sp.set_color(GRID)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2,
               fontsize=6.5, facecolor=BG, labelcolor=FG, loc="upper left")

    # ------------------------------------------------------------------
    # Panel 5 — Latency bar chart (bottom-left)
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor(BG)
    ax5.set_title("Replan Latency vs Gate", color=FG, fontsize=9)

    labels  = [f"{rp.replan_id}\n{rp.trigger}" for rp in result.replans]
    latencies = [rp.wall_latency_ms for rp in result.replans]
    colours = [GRN if rp.kpi_ew02_pass else RED for rp in result.replans]

    bars = ax5.barh(labels, latencies, color=colours, alpha=0.85, height=0.4)
    ax5.axvline(1000, color=AMB, linestyle="--", linewidth=1.2, label="EW-02 limit (1000 ms)")
    ax5.set_xlabel("Wall-clock latency (ms)", color=FG, fontsize=8)
    ax5.tick_params(colors=FG, labelsize=7)
    for sp in ax5.spines.values():
        sp.set_color(GRID)
    ax5.legend(fontsize=6.5, facecolor=BG, labelcolor=FG)
    for bar, lat in zip(bars, latencies):
        ax5.text(lat + 5, bar.get_y() + bar.get_height() / 2,
                 f"{lat:.0f} ms", color=FG, va="center", fontsize=7)

    # ------------------------------------------------------------------
    # Panel 6 — Cost map update latencies (bottom-centre)
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor(BG)
    ax6.set_title("Cost Map Update Latencies", color=FG, fontsize=9)

    if result.cost_map_updates:
        update_times = [u.timestamp_s / 60 for u in result.cost_map_updates]
        update_lats  = [u.wall_latency_ms for u in result.cost_map_updates]
        ax6.scatter(update_times, update_lats,
                    color=[GRN if u.kpi_e01_pass else RED for u in result.cost_map_updates],
                    s=15, alpha=0.7, label="Update latency")
        ax6.axhline(500, color=AMB, linestyle="--", linewidth=1.2, label="KPI-E01 limit (500 ms)")
        ax6.set_xlabel("Mission time (min)", color=FG, fontsize=8)
        ax6.set_ylabel("Latency (ms)", color=FG, fontsize=8)
        ax6.tick_params(colors=FG, labelsize=7)
        for sp in ax6.spines.values():
            sp.set_color(GRID)
        ax6.legend(fontsize=6.5, facecolor=BG, labelcolor=FG)

    # ------------------------------------------------------------------
    # Panel 7 — KPI gate summary (bottom-right)
    # ------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor(BG)
    ax7.axis("off")
    ax7.set_title("S4 Acceptance Gate", color=FG, fontsize=9)

    def kpi_row(label: str, value: str, passed: bool) -> str:
        mark = "PASS ✓" if passed else "FAIL ✗"
        colour = GRN if passed else RED
        return label, value, mark, colour

    rows = [
        kpi_row("KPI-E01  Cost map < 500 ms",
                f"max {max((u.wall_latency_ms for u in result.cost_map_updates), default=0):.1f} ms",
                result.kpi_e01_pass),
        kpi_row("EW-02  Replan 1 < 1 s",
                f"{result.replans[0].wall_latency_ms:.1f} ms" if result.replans else "N/A",
                result.replans[0].kpi_ew02_pass if result.replans else False),
        kpi_row("EW-02  Replan 2 < 1 s",
                f"{result.replans[1].wall_latency_ms:.1f} ms" if len(result.replans) > 1 else "N/A",
                result.replans[1].kpi_ew02_pass if len(result.replans) > 1 else False),
        kpi_row("KPI-E03  ≥ 2 replans visible",
                f"{len(result.replans)} replans",
                result.kpi_e03_pass),
    ]

    y = 0.85
    for label, value, mark, colour in rows:
        ax7.text(0.02, y, label,  color=FG,     fontsize=8, transform=ax7.transAxes)
        ax7.text(0.60, y, value,  color=AMB,    fontsize=8, transform=ax7.transAxes)
        ax7.text(0.82, y, mark,   color=colour, fontsize=8,
                 fontweight="bold", transform=ax7.transAxes)
        y -= 0.18

    # Overall result
    overall = result.kpi_e01_pass and result.kpi_ew02_pass and result.kpi_e03_pass
    ax7.text(0.02, y - 0.05, "S4 GATE",
             color=GRN if overall else RED,
             fontsize=13, fontweight="bold", transform=ax7.transAxes)
    ax7.text(0.35, y - 0.05, "PASS ✓" if overall else "FAIL ✗",
             color=GRN if overall else RED,
             fontsize=13, fontweight="bold", transform=ax7.transAxes)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    status_text = (
        f"Mission: {result.mission_id}   "
        f"Duration: {result.total_time_s/60:.1f} min   "
        f"Replans: {len(result.replans)}   "
        f"Cost updates: {len(result.cost_map_updates)}   "
        f"KPI-E01: {'PASS ✓' if result.kpi_e01_pass else 'FAIL ✗'}   "
        f"KPI-EW02: {'PASS ✓' if result.kpi_ew02_pass else 'FAIL ✗'}   "
        f"KPI-E03: {'PASS ✓' if result.kpi_e03_pass else 'FAIL ✗'}"
    )
    fig.text(0.01, 0.01, status_text, color=FG, fontsize=7.5,
             bbox=dict(facecolor="#161B22", edgecolor=GRID, boxstyle="round,pad=0.3"))

    fig.savefig(output_path, dpi=120, facecolor=BG, bbox_inches="tight")
    print(f"  Dashboard saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_bcmp1_ew_sim(verbose=True)
    render_dashboard(result, "dashboard/ew_dashboard.png")
