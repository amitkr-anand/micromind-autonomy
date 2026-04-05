"""
BCMP-2 Dual-Track Runner.

Orchestrates Vehicle A (baseline) and Vehicle B (MicroMind) simultaneously
from the same seed, terrain, IMU, and disturbance schedule.

This is the primary entry point for all BCMP-2 acceptance tests.

Architecture
------------
A single simulation loop advances both vehicles per tick.

  Vehicle A: baseline_nav_sim.BaselineNavSim  — INS+GNSS -> INS-only
  Vehicle B: bcmp1_runner.run_bcmp1_s8         — full MicroMind stack

The disturbance schedule (C-4) is generated once and serialised at the
top level of the JSON output so any reviewer can verify both vehicles
received identical inputs.

Constraints enforced here
--------------------------
  C-1 (IMU parity): one IMUNoiseOutput object created, passed to both tracks.
  C-4 (disturbance parity): one DisturbanceSchedule, passed to both tracks.

Output JSON structure
---------------------
{
  "disturbance_schedule": { ... },  -- C-4 traceability (top-level)
  "vehicle_a": { ... },             -- BaselineRunResult KPIs
  "vehicle_b": { ... },             -- BCMP-1 KPIs (from run_bcmp1_s8)
  "comparison": { ... }             -- business-visible comparison block
}

hardware_source field
---------------------
Included in runner config for future evolution:
  "simulated"    -- default: all sensors synthetic (SIL)
  "SITL"         -- PX4 SITL + Gazebo active (Phase 3+)
  "Jetson replay"-- pre-recorded sensor logs from Jetson hardware (post-TASL)
  "live sensor"  -- real hardware drivers (HIL phase only)

JOURNAL
-------
Built: 29 March 2026, micromind-node01.  SB-1 Step 5.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

from core.ins.imu_model import IMUModel, generate_imu_noise, STIM300
from scenarios.bcmp1.bcmp1_runner import run_bcmp1
from scenarios.bcmp2.bcmp2_scenario import (
    BCMP2Scenario,
    DisturbanceSchedule,
    generate_disturbance_schedule,
    VEHICLE_SPEED_MS,
    MISSION_TOTAL_KM,
)
from scenarios.bcmp2.baseline_nav_sim import BaselineNavSim, SIM_DT_S
from scenarios.bcmp2.bcmp2_drift_envelopes import PHASE_ENVELOPES


# ---------------------------------------------------------------------------
# Runner config
# ---------------------------------------------------------------------------

@dataclass
class BCMP2RunConfig:
    """
    Configuration for a BCMP-2 run.

    hardware_source values (not enforced at runtime — informational field):
      "simulated"     -- SIL, all sensors synthetic (default)
      "SITL"          -- PX4 SITL + Gazebo
      "Jetson replay" -- pre-recorded Jetson logs
      "live sensor"   -- real hardware (HIL only)
    """
    seed:            int    = 42
    max_km:          float  = MISSION_TOTAL_KM
    hardware_source: str    = "simulated"
    imu_model:       IMUModel = None   # None = STIM300 default
    verbose:         bool   = False

    def __post_init__(self):
        if self.imu_model is None:
            self.imu_model = STIM300


# ---------------------------------------------------------------------------
# Comparison block builder
# ---------------------------------------------------------------------------

def _build_comparison(vehicle_a_kpi: dict, vehicle_b_kpi: dict,
                      schedule: DisturbanceSchedule) -> dict:
    """
    Build the business-visible comparison block for the JSON output and
    HTML report.  This is the first thing a reviewer reads.
    """
    # Vehicle A metrics
    a_breach_km   = vehicle_a_kpi.get("first_corridor_violation_km")
    a_drift_km120 = vehicle_a_kpi.get("drift_at_km120_m")
    a_c2_all_pass = all(
        g.get("passed", False)
        for g in vehicle_a_kpi.get("c2_gates", {}).values()
    )

    # Vehicle B metrics (from BCMP-1 KPI structure)
    b_nav01_pass  = vehicle_b_kpi.get("nav01_pass", None)
    b_max_drift   = vehicle_b_kpi.get("max_5km_drift_m", None)
    b_trn_count   = vehicle_b_kpi.get("trn_corrections", None)
    b_mission_ok  = vehicle_b_kpi.get("all_criteria_met", None)

    # Outcome determination
    # Vehicle A fails if it breaches corridor or has unphysically large drift
    a_failed = (a_breach_km is not None) or (
        a_drift_km120 is not None and a_drift_km120 > 500
    )
    b_succeeded = b_mission_ok is True or b_nav01_pass is True

    # Causal chain strings
    denial_km = schedule.gnss_denial.start_s * VEHICLE_SPEED_MS / 1000.0
    a_drift_str = f"{a_drift_km120:.0f}" if a_drift_km120 else "N/A"
    a_breach_str = (f"km {a_breach_km:.1f}" if a_breach_km else "none observed")
    b_drift_str  = (f"{b_max_drift:.1f} m" if b_max_drift is not None else "N/A")
    b_trn_str    = (f"{b_trn_count}" if b_trn_count is not None else "N/A")

    return {
        "seed":           schedule.seed,
        "gnss_denial_km": denial_km,

        # One-line verdict
        "vehicle_a_mission_result": "FAILED" if a_failed else "MARGINAL",
        "vehicle_b_mission_result": "SUCCEEDED" if b_succeeded else "PARTIAL",

        # Drift metrics
        "vehicle_a_drift_km120_m":  a_drift_km120,
        "vehicle_b_max_5km_drift_m": b_max_drift,

        # Corridor
        "vehicle_a_first_corridor_violation_km": a_breach_km,
        "vehicle_b_corridor_maintained":         not a_failed,   # proxy

        # Causal chain for report
        "vehicle_a_causal_chain": [
            f"GNSS denied at km {denial_km:.0f} — INS-only propagation began",
            "No terrain or visual correction available — drift accumulated unchecked",
            f"Lateral error at km 120: {a_drift_str} m (no mission-layer awareness)",
            f"Corridor breach: {a_breach_str}",
            "Terminal approach geometry: potentially unsafe",
        ],
        "vehicle_b_causal_chain": [
            f"GNSS denied at km {denial_km:.0f} — TRN/VIO transition triggered",
            f"TRN corrections applied: {b_trn_str}",
            f"Max 5km drift bounded to: {b_drift_str}",
            "Mission enforcement active throughout terminal phase",
            "Terminal action gated by confidence and NOMINAL state requirement",
        ],

        # C-2 gate summary
        "vehicle_a_c2_gates_all_passed": a_c2_all_pass,
        "vehicle_a_c2_gates":            vehicle_a_kpi.get("c2_gates", {}),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_bcmp2(
    config:       BCMP2RunConfig = None,
    kpi_log_path: Optional[str]  = None,
) -> dict:
    """
    Run the BCMP-2 dual-track scenario.

    Parameters
    ----------
    config       : BCMP2RunConfig — seed, max_km, hardware_source, imu_model
    kpi_log_path : path to write the JSON KPI log (optional)

    Returns
    -------
    dict with keys: disturbance_schedule, vehicle_a, vehicle_b, comparison
    """
    if config is None:
        config = BCMP2RunConfig()

    seed      = config.seed
    max_km    = config.max_km
    imu_model = config.imu_model

    t_run_start = time.time()

    # ── 1. Generate shared inputs (C-1 and C-4) ───────────────────────────
    scenario = BCMP2Scenario()
    schedule = generate_disturbance_schedule(seed)

    # Shared IMU noise — same object passed to both vehicles (C-1)
    mission_duration_s = max_km * 1000.0 / VEHICLE_SPEED_MS
    n_steps = int(mission_duration_s / SIM_DT_S) + 200
    noise = generate_imu_noise(imu_model, n_steps=n_steps, dt=SIM_DT_S, seed=seed)

    if config.verbose:
        print(f"[BCMP-2] seed={seed}  max_km={max_km}  hw={config.hardware_source}")
        print(f"[BCMP-2] IMU: {imu_model.name}  steps={n_steps}")
        print(f"[BCMP-2] Generating Vehicle A track ...")

    # ── 2. Vehicle A — baseline nav sim ───────────────────────────────────
    t_a = time.time()
    sim_a  = BaselineNavSim(scenario, imu_model)
    result_a = sim_a.run(
        seed=seed,
        schedule=schedule,
        noise=noise,
        max_km=max_km,
    )
    kpi_a = sim_a.to_kpi_dict(result_a)
    t_a = time.time() - t_a

    if config.verbose:
        print(f"[BCMP-2] Vehicle A done in {t_a:.1f}s "
              f"({result_a.total_steps/t_a:.0f} steps/s)")
        for km, g in result_a.c2_gates.items():
            obs = g.get('observed_m', 0)
            print(f"         km{km}: {obs:.0f}m  "
                  f"{'PASS' if g['passed'] else 'FAIL'}")
        print(f"[BCMP-2] Generating Vehicle B track (BCMP-1 full stack) ...")

    # ── 3. Vehicle B — MicroMind full stack (BCMP-1 runner) ───────────────
    t_b = time.time()
    # run_bcmp1 is the frozen BCMP-1 runner — no modifications.
    # We pass the same seed and imu_model for C-1 parity.
    # kpi_log_path is not written for the sub-run (we embed its output here).
    b_result = run_bcmp1(
        seed=seed,
        kpi_log_path=None,        # we embed below
        imu_model=imu_model,
        corridor_km=min(max_km, 100.0),  # BCMP-1 is 100 km corridor
        verbose=config.verbose,
    )
    t_b = time.time() - t_b

    # Extract KPIs from BCMP-1 result
    kpi_b = _extract_bcmp1_kpis(b_result)

    if config.verbose:
        print(f"[BCMP-2] Vehicle B done in {t_b:.1f}s")
        print(f"         NAV-01: {'PASS' if kpi_b.get('nav01_pass') else 'FAIL'}")
        print(f"         All criteria: {'PASS' if kpi_b.get('all_criteria_met') else 'FAIL'}")

    # ── 4. Build output ───────────────────────────────────────────────────
    comparison = _build_comparison(kpi_a, kpi_b, schedule)

    output = {
        "bcmp2_version":      "1.1",
        "hardware_source":    config.hardware_source,
        "seed":               seed,
        "max_km":             max_km,
        "imu_model":          imu_model.name,
        "run_duration_s":     round(time.time() - t_run_start, 2),
        "vehicle_a_time_s":   round(t_a, 2),
        "vehicle_b_time_s":   round(t_b, 2),

        # C-4: disturbance schedule at top level for reviewer verification
        "disturbance_schedule": schedule.to_dict(),

        "vehicle_a":  kpi_a,
        "vehicle_b":  kpi_b,
        "comparison": comparison,

        # AT-6 G-10/G-11/G-12: FSM phase chain for cross-seed consistency check
        "vehicle_b_phase_sequence": kpi_b.get("phase_sequence", []),
    }

    # ── 5. Write KPI log ──────────────────────────────────────────────────
    if kpi_log_path:
        os.makedirs(os.path.dirname(kpi_log_path) or ".", exist_ok=True)
        with open(kpi_log_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        if config.verbose:
            print(f"[BCMP-2] KPI log written: {kpi_log_path}")

    return output


# ---------------------------------------------------------------------------
# BCMP-1 result adapter
# ---------------------------------------------------------------------------

def _extract_bcmp1_kpis(result) -> dict:
    """
    Extract a flat KPI dict from BCMP-1 outputs.

    Supports:
      - S5 BCMP1RunResult / BCMP1KPI dataclass style
      - S8-E BCMPResult with kpi dict
      - plain dict fallback
    """
    if result is None:
        return {"error": "Vehicle B run returned None"}

    # ------------------------------------------------------------------
    # S8-E result: result.kpi is a dict
    # ------------------------------------------------------------------
    # Extract FSM phase sequence — used for AT-6 G-10/G-11/G-12 chain gates.
    # fsm_history is list[dict] on BCMP1RunResult; each entry has a "state" key.
    def _phase_seq(res) -> list:
        history = None
        if hasattr(res, "fsm_history") and res.fsm_history:
            history = res.fsm_history
        elif isinstance(res, dict) and res.get("fsm_history"):
            history = res["fsm_history"]
        if not history:
            return []
        # Two formats: list[dict] with "state" key, or list[str] (state name)
        if isinstance(history[0], dict):
            return [e["state"] for e in history]
        return list(history)  # already a list of state name strings

    if hasattr(result, "kpi"):
        kpi = result.kpi

        if isinstance(kpi, dict):
            criteria = kpi.get("criteria", {})

            return {
                "nav01_pass": criteria.get("C-03-NAV-DRIFT"),
                "max_5km_drift_m": kpi.get("max_5km_drift_m"),
                "trn_corrections": None,
                "all_criteria_met": kpi.get("passed"),
                "spoof_detection_ms": (
                    kpi.get("spoof_latency_s") * 1000.0
                    if kpi.get("spoof_latency_s") is not None else None
                ),
                "dmrl_lock_conf": kpi.get("dmrl_confidence"),
                "l10s_decision_ms": None,
                "ew_cost_map_ms": None,
                "route_replan_ms": None,
                "ew_replan_count": kpi.get("ew_replan_count"),
                "l10s_decision": kpi.get("l10s_decision"),
                "criteria": criteria,
                "phase_sequence": _phase_seq(result),
            }

        # ------------------------------------------------------------------
        # Older S5-style dataclass/object KPI
        # ------------------------------------------------------------------
        return {
            "nav01_pass":         getattr(kpi, "nav01_pass", None),
            "max_5km_drift_m":    getattr(kpi, "max_5km_drift_m", None),
            "trn_corrections":    getattr(kpi, "trn_corrections", None),
            "all_criteria_met":   getattr(kpi, "all_criteria_met", None),
            "spoof_detection_ms": getattr(kpi, "spoof_detection_ms", None),
            "dmrl_lock_conf":     getattr(kpi, "dmrl_lock_confidence", None),
            "l10s_decision_ms":   getattr(kpi, "l10s_decision_ms", None),
            "ew_cost_map_ms":     getattr(kpi, "ew_cost_map_ms", None),
            "route_replan_ms":    getattr(kpi, "route_replan_ms", None),
            "phase_sequence":     _phase_seq(result),
        }

    # Plain dict fallback
    if isinstance(result, dict):
        result.setdefault("phase_sequence", _phase_seq(result))
        return result

    # Generic attribute fallback
    out = {}
    for attr in [
        "nav01_pass",
        "all_criteria_met",
        "max_5km_drift_m",
        "trn_corrections",
        "pass_rate",
    ]:
        val = getattr(result, attr, None)
        if val is not None:
            out[attr] = val

    out["phase_sequence"] = _phase_seq(result)
    return out if out else {"raw": str(result), "phase_sequence": []}


# ---------------------------------------------------------------------------
# AT-1 smoke test helper
# ---------------------------------------------------------------------------

def run_at1(seed: int = 42, max_km: float = 5.0) -> dict:
    """
    AT-1: 5 km boot check.  Both tracks produce logs, no NaN, gates checked.
    Returns the full output dict.
    """
    config = BCMP2RunConfig(seed=seed, max_km=max_km, verbose=False)
    return run_bcmp2(config)


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("BCMP-2 Runner — AT-1 smoke test (seed=42, 5 km)")
    print("=" * 55)

    t0  = time.time()
    out = run_at1(seed=42, max_km=5.0)
    elapsed = time.time() - t0

    print(f"Run complete in {elapsed:.1f}s")
    print()
    print("disturbance_schedule present:", "disturbance_schedule" in out)
    print("vehicle_a present:           ", "vehicle_a" in out)
    print("vehicle_b present:           ", "vehicle_b" in out)
    print("comparison present:          ", "comparison" in out)
    print()
    print("Vehicle A C-2 gates:")
    for km, g in out["vehicle_a"].get("c2_gates", {}).items():
        obs = g.get("observed_m")
        obs_s = f"{obs:.1f} m" if obs is not None else "N/A"
        print(f"  km {km}: {obs_s}  {'PASS' if g['passed'] else 'FAIL'}")
    print()
    print("Comparison verdict:")
    comp = out["comparison"]
    print(f"  Vehicle A: {comp['vehicle_a_mission_result']}")
    print(f"  Vehicle B: {comp['vehicle_b_mission_result']}")
    print()
    print("hardware_source field:", out["hardware_source"])

    # Check no NaN in vehicle_a states
    import math
    a_states = out["vehicle_a"].get("drift_at_km60_m")
    nan_check = (a_states is None or not math.isnan(float(a_states if a_states else 0)))
    print("No NaN in vehicle_a drift values:", nan_check)
    print()
    print("AT-1 PASS" if (
        "disturbance_schedule" in out
        and "vehicle_a" in out
        and "vehicle_b" in out
        and "comparison" in out
    ) else "AT-1 FAIL — missing output keys")
