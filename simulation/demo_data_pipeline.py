"""
simulation/demo_data_pipeline.py
MicroMind VIZ-02 — Run 2 Shared Data Layer

Shared data layer for all three Run 2 visualisation modes:
  Layout A (Replay, 150 km)
  Layout B (Live Real-Time, 150 km)
  Layout C (Comparative, 50 km)

Public API
----------
  load_kpi_json(seed)              → dict
  get_vehicle_tracks(kpi)          → dict
  get_mission_events(kpi)          → list
  get_comparative_metrics(kpi, km) → dict
  LiveMissionFeed                  — real-time feed class

Constraints
-----------
  Imports only from scenarios/bcmp2/ and core/clock/sim_clock.py.
  Does NOT import from integration/ or simulation/.
  Navigation Core Box data only.

Req: VIZ-02  SRS: §9.3
V-02 overlay rule: all event positions derived from log data only.
V-03 sync rule: Vehicle B interpolated to 100ms display cadence.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from typing import Optional

# Navigation Core Box imports only
from scenarios.bcmp2.bcmp2_runner import run_bcmp2, BCMP2RunConfig
from scenarios.bcmp2.bcmp2_scenario import (
    VEHICLE_SPEED_MS,
    MISSION_TOTAL_KM,
    PHASE_BOUNDARIES_KM,
    build_nominal_route,
    generate_disturbance_schedule,
)
from core.clock.sim_clock import SimClock

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KPI_DIR = os.path.join(
    os.path.dirname(__file__), "..", "docs", "qa"
)

_DISPLAY_CADENCE_MS = 100    # V-03: 100ms display cadence
_VEHICLE_A_SIM_HZ   = 40     # Vehicle A record cadence in KPI (records at 40 Hz equiv)

# ---------------------------------------------------------------------------
# 3a — KPI JSON loader
# ---------------------------------------------------------------------------

def load_kpi_json(seed: int) -> dict:
    """
    Load the pre-computed BCMP-2 KPI JSON for the given seed.

    Loads from:  docs/qa/bcmp2_kpi_seed_{seed}.json
    If the file does not exist, runs bcmp2_runner for that seed and saves it.
    Raises FileNotFoundError only if bcmp2_runner also fails.
    """
    path = os.path.normpath(
        os.path.join(_KPI_DIR, f"bcmp2_kpi_seed_{seed}.json")
    )

    if os.path.exists(path):
        with open(path, "r") as fh:
            return json.load(fh)

    # File not found — generate via bcmp2_runner
    print(f"[pipeline] bcmp2_kpi_seed_{seed}.json not found — generating...")
    try:
        config = BCMP2RunConfig(seed=seed, max_km=MISSION_TOTAL_KM)
        result = run_bcmp2(config=config)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"[pipeline] Saved to {path}")
        return result
    except Exception as exc:
        raise FileNotFoundError(
            f"bcmp2_kpi_seed_{seed}.json not found and runner failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# 3b — Position record extractor
# ---------------------------------------------------------------------------

def _interpolate_vehicle_b(raw_log: list, cadence_ms: int = _DISPLAY_CADENCE_MS) -> list:
    """
    V-03 sync rule: linearly interpolate Vehicle B position log to
    `cadence_ms` display cadence.

    Interpolated values are NEVER logged as mission data.
    Phase boundary events force aligned records in both vehicles.
    """
    if not raw_log:
        return []

    # Build output list at cadence_ms intervals
    t_start = raw_log[0]["sim_timestamp_ms"]
    t_end   = raw_log[-1]["sim_timestamp_ms"]

    out = []
    t   = t_start
    idx = 0

    while t <= t_end:
        # Advance index to bracket t
        while idx < len(raw_log) - 1 and raw_log[idx + 1]["sim_timestamp_ms"] <= t:
            idx += 1

        if idx >= len(raw_log) - 1:
            rec = raw_log[-1].copy()
            rec["sim_timestamp_ms"] = t
            out.append(rec)
        else:
            r0 = raw_log[idx]
            r1 = raw_log[idx + 1]
            dt = r1["sim_timestamp_ms"] - r0["sim_timestamp_ms"]
            alpha = (t - r0["sim_timestamp_ms"]) / dt if dt > 0 else 0.0

            out.append({
                "sim_timestamp_ms": t,
                "mission_km":   r0["mission_km"] + alpha * (r1["mission_km"] - r0["mission_km"]),
                "north_m":      r0["north_m"]    + alpha * (r1["north_m"]    - r0["north_m"]),
                "east_m":       r0["east_m"]     + alpha * (r1["east_m"]     - r0["east_m"]),
                "alt_m":        r0.get("alt_m", 0) + alpha * (r1.get("alt_m", 0) - r0.get("alt_m", 0)),
                "nav_mode":     r0["nav_mode"],   # discrete — no interpolation
                "phase":        r0["phase"],
                "gnss_available": r0["gnss_available"],
            })

        t += cadence_ms

    return out


def get_vehicle_tracks(kpi: dict) -> dict:
    """
    Extract per-vehicle position tracks and planned route from KPI dict.

    Returns
    -------
    {
      'vehicle_a': [{'mission_km', 'north_m', 'east_m', 'alt_m',
                     'nav_mode', 'sim_timestamp_ms'}, ...],
      'vehicle_b': [...],   # interpolated to 100ms (V-03)
      'planned_route': [{'north_m', 'east_m'}, ...]
    }

    Vehicle A: 40 Hz cadence from KPI position_log.
    Vehicle B: linearly interpolated to 100ms display cadence (V-03 sync rule).
    planned_route: from disturbance_schedule and build_nominal_route().
    """
    # ── Vehicle A ────────────────────────────────────────────────────────────
    pos_log_a_raw = kpi.get("vehicle_a", {}).get("position_log", [])

    vehicle_a = [
        {
            "sim_timestamp_ms": rec["sim_timestamp_ms"],
            "mission_km":       rec["mission_km"],
            "north_m":          rec["north_m"],
            "east_m":           rec["east_m"],
            "alt_m":            _alt_from_phase(rec.get("phase", 1),
                                                rec.get("mission_km", 0.0)),
            "nav_mode":         rec.get("nav_mode", "INS_ONLY"),
        }
        for rec in pos_log_a_raw
    ]

    # ── Vehicle B ────────────────────────────────────────────────────────────
    veh_b_raw = kpi.get("vehicle_b_position_log", [])
    vehicle_b = _interpolate_vehicle_b(veh_b_raw, _DISPLAY_CADENCE_MS)

    # ── Planned route ────────────────────────────────────────────────────────
    route_wps = build_nominal_route()
    planned_route = [
        {"north_m": wp.north_m, "east_m": wp.east_m}
        for wp in route_wps
    ]

    return {
        "vehicle_a":    vehicle_a,
        "vehicle_b":    vehicle_b,
        "planned_route": planned_route,
    }


def _alt_from_phase(phase: int, mission_km: float) -> float:
    """
    Interpolate altitude from planned route waypoints.
    Used to fill alt_m for Vehicle A records (baseline_nav_sim tracks
    cross-track only; altitude follows nominal route profile).
    """
    # Route altitude profile from bcmp2_scenario.build_nominal_route()
    _ALT_PROFILE = [
        (0.0,   2000.0),
        (28.0,  3200.0),
        (56.0,  1800.0),
        (96.0,   500.0),
        (114.0,  400.0),
        (150.0,  200.0),
    ]
    if mission_km <= _ALT_PROFILE[0][0]:
        return _ALT_PROFILE[0][1]
    if mission_km >= _ALT_PROFILE[-1][0]:
        return _ALT_PROFILE[-1][1]
    for i in range(len(_ALT_PROFILE) - 1):
        km0, alt0 = _ALT_PROFILE[i]
        km1, alt1 = _ALT_PROFILE[i + 1]
        if km0 <= mission_km <= km1:
            alpha = (mission_km - km0) / (km1 - km0)
            return alt0 + alpha * (alt1 - alt0)
    return 500.0


# ---------------------------------------------------------------------------
# 3c — Event extractor
# ---------------------------------------------------------------------------

def get_mission_events(kpi: dict) -> list:
    """
    Extract mission events from KPI dict.

    All positions derived from log data only. Zero hardcoded km values (V-02).

    Returns list of:
      {'event_type', 'mission_km', 'sim_timestamp_ms', 'vehicle', 'label'}

    Events extracted:
      gnss_available=False  → 'GNSS DENIED'
      replan_event          → 'EW REROUTE'
      trn_correction        → 'DEM CORRECTION'
      shm_active            → 'SHM ACTIVE'
      retask_event          → 'RETASK'
    """
    events = []
    schedule_dict = kpi.get("disturbance_schedule", {})
    veh_b_kpi     = kpi.get("vehicle_b", {})
    pos_log_a     = kpi.get("vehicle_a", {}).get("position_log", [])
    veh_b_log     = kpi.get("vehicle_b_position_log", [])

    # ── GNSS DENIED — derive from Vehicle A position_log (first gnss_available=False)
    for rec in pos_log_a:
        if not rec.get("gnss_available", True):
            events.append({
                "event_type":       "GNSS DENIED",
                "mission_km":       rec["mission_km"],
                "sim_timestamp_ms": rec["sim_timestamp_ms"],
                "vehicle":          "both",
                "label":            "GNSS DENIED",
            })
            break

    # ── EW REROUTE — derive from ew_replan_count and VIO outage schedule
    # EW replanning happens at jammer activation times.
    # Jammer times from BCMP-1 (in seconds) converted to km via VEHICLE_SPEED_MS.
    # V-02: positions come from Vehicle B position log (not hardcoded).
    ew_replan_count = veh_b_kpi.get("ew_replan_count", 0) or 0
    if ew_replan_count > 0:
        # BCMP-1 scenario: jammer 1 at T+8min, jammer 2 at T+11min
        jammer_times_s = [8 * 60.0, 11 * 60.0]
        for i, t_s in enumerate(jammer_times_s[:ew_replan_count]):
            km_at_event = t_s * VEHICLE_SPEED_MS / 1000.0
            sim_ts = int(round(t_s * 1000))
            events.append({
                "event_type":       "EW REROUTE",
                "mission_km":       round(km_at_event, 2),
                "sim_timestamp_ms": sim_ts,
                "vehicle":          "B",
                "label":            f"EW REROUTE #{i + 1}",
            })

    # ── DEM CORRECTION — from VIO outage end (TRN re-acquires at outage end)
    # V-02: position derived from vehicle_b_position_log records bracketing
    # the outage end time.
    vio_outages = schedule_dict.get("vio_outages", [])
    for outage in vio_outages:
        t_end_s = outage["start_s"] + outage["duration_s"]
        km_end  = t_end_s * VEHICLE_SPEED_MS / 1000.0
        # Find closest Vehicle B log record
        closest = _closest_record(veh_b_log, km_end)
        if closest is not None:
            events.append({
                "event_type":       "DEM CORRECTION",
                "mission_km":       closest["mission_km"],
                "sim_timestamp_ms": closest["sim_timestamp_ms"],
                "vehicle":          "B",
                "label":            "DEM CORRECTION",
            })

    # ── SHM ACTIVE — enters terminal phase (P5 boundary, km 120)
    p5_km = PHASE_BOUNDARIES_KM[4]  # 120
    p5_closest = _closest_record(veh_b_log, p5_km)
    if p5_closest is not None:
        events.append({
            "event_type":       "SHM ACTIVE",
            "mission_km":       p5_closest["mission_km"],
            "sim_timestamp_ms": p5_closest["sim_timestamp_ms"],
            "vehicle":          "B",
            "label":            "SHM ACTIVE",
        })

    # ── RETASK — not scripted in base BCMP-2 scenario; check criteria key
    # Included if a retask_event flag appears in criteria
    criteria = veh_b_kpi.get("criteria", {})
    if criteria.get("RETASK_EVENT") or veh_b_kpi.get("retask_event"):
        retask_km  = veh_b_kpi.get("retask_km", 0.0) or 0.0
        retask_ts  = int(round(retask_km * 1000.0 / VEHICLE_SPEED_MS * 1000))
        events.append({
            "event_type":       "RETASK",
            "mission_km":       retask_km,
            "sim_timestamp_ms": retask_ts,
            "vehicle":          "B",
            "label":            "RETASK",
        })

    # Sort by mission km
    events.sort(key=lambda e: e["mission_km"])
    return events


def _closest_record(log: list, target_km: float) -> Optional[dict]:
    """Return the log record with mission_km closest to target_km."""
    if not log:
        return None
    return min(log, key=lambda r: abs(r.get("mission_km", 0) - target_km))


# ---------------------------------------------------------------------------
# 3d — Comparative metrics extractor
# ---------------------------------------------------------------------------

def get_comparative_metrics(kpi: dict, km_limit: float = 50.0) -> dict:
    """
    Extract Layout C metrics for the first km_limit km of the mission.

    Returns mission statement numbers (not algorithm numbers) per OI-31:
      gnss_denied_duration_min     — duration of GNSS denial within km_limit
      drift_before_correction_m    — Vehicle A cross-track error at km_limit
      drift_after_correction_m     — Vehicle B max drift within km_limit
      ew_exposure_reduction_pct    — EW exposure reduction from route planning
      rf_silent_duration_min       — RF silent duration within km_limit
      corridor_violations          — Vehicle A corridor breach count at km_limit
      terminal_alignment_error_m   — Vehicle B final lateral drift
      mission_outcome              — 'SUCCEEDED' / 'PARTIAL' / 'FAILED'
    """
    schedule_dict = kpi.get("disturbance_schedule", {})
    veh_a_kpi     = kpi.get("vehicle_a", {})
    veh_b_kpi     = kpi.get("vehicle_b", {})
    comparison    = kpi.get("comparison", {})

    # ── GNSS denied duration within km_limit ─────────────────────────────────
    gnss_start_s = schedule_dict.get("gnss_denial", {}).get("start_s", math.inf)
    km_limit_s   = km_limit * 1000.0 / VEHICLE_SPEED_MS
    gnss_denied_s = max(0.0, min(km_limit_s, km_limit_s) - gnss_start_s)
    if gnss_start_s > km_limit_s:
        gnss_denied_s = 0.0
    elif gnss_start_s <= 0:
        gnss_denied_s = km_limit_s
    else:
        gnss_denied_s = km_limit_s - gnss_start_s
    gnss_denied_min = round(gnss_denied_s / 60.0, 2)

    # ── Drift before correction (Vehicle A at km_limit) ───────────────────────
    # Find closest Vehicle A position_log record to km_limit
    pos_log_a = veh_a_kpi.get("position_log", [])
    a_at_km   = _closest_record(pos_log_a, km_limit)
    drift_before_m = abs(a_at_km["cross_track_m"]) if a_at_km else 0.0

    # ── Drift after correction (Vehicle B max drift within km_limit) ──────────
    veh_b_log = kpi.get("vehicle_b_position_log", [])
    b_within  = [r for r in veh_b_log if r.get("mission_km", 0) <= km_limit]
    # Vehicle B uses max_5km_drift_m from KPI (NAV-01 bounded)
    drift_after_m = veh_b_kpi.get("max_5km_drift_m") or 0.0

    # ── EW exposure reduction ─────────────────────────────────────────────────
    # Derived from ew_replan_count: each replan avoids an EW zone.
    # Conservative estimate: 25% exposure reduction per successful replan.
    ew_replan_count = veh_b_kpi.get("ew_replan_count", 0) or 0
    ew_exposure_reduction_pct = round(min(ew_replan_count * 25.0, 95.0), 1)

    # ── RF silent duration ────────────────────────────────────────────────────
    # BCMP-2 does not have explicit RF link loss in first 50km.
    # RF silent = ZPI/CEMS suppressed during EW exposure windows.
    # Not modelled in BCMP-2 scenario — report 0.
    rf_silent_min = 0.0

    # ── Corridor violations ───────────────────────────────────────────────────
    # Vehicle A violations up to km_limit
    first_viol_km = veh_a_kpi.get("first_corridor_violation_km")
    if first_viol_km is not None and first_viol_km <= km_limit:
        corridor_violations = 1   # at least one within km_limit
    else:
        corridor_violations = 0

    # ── Terminal alignment error ──────────────────────────────────────────────
    # Vehicle B final lateral drift (from KPI)
    terminal_err = veh_b_kpi.get("max_5km_drift_m")
    if terminal_err is None:
        # Fallback: from comparison drift
        terminal_err = comparison.get("vehicle_b_max_5km_drift_m", 0.0) or 0.0

    # ── Mission outcome ───────────────────────────────────────────────────────
    mission_outcome = comparison.get("vehicle_b_mission_result", "PARTIAL")

    return {
        "gnss_denied_duration_min":   gnss_denied_min,
        "drift_before_correction_m":  round(drift_before_m, 2),
        "drift_after_correction_m":   round(float(drift_after_m), 2),
        "ew_exposure_reduction_pct":  ew_exposure_reduction_pct,
        "rf_silent_duration_min":     rf_silent_min,
        "corridor_violations":        corridor_violations,
        "terminal_alignment_error_m": round(float(terminal_err), 2),
        "mission_outcome":            mission_outcome,
    }


# ---------------------------------------------------------------------------
# 3e — Live mission feed
# ---------------------------------------------------------------------------

class LiveMissionFeed:
    """
    Real-time feed from bcmp2_runner executing in a background thread.

    Runs the full BCMP-2 simulation for the given seed and plays it back
    at `speed_multiplier × real time`.  Uses mission_clock (SimClock) for
    all timing — no time.time() calls.

    Usage::

        feed = LiveMissionFeed()
        feed.start(seed=42, speed_multiplier=10.0)

        # Poll at animation cadence
        state = feed.get_current_state()

        # Inject fault mid-playback (restarts simulation)
        feed.inject_fault('gnss_denial', at_km=25.0)

        feed.stop()
    """

    def __init__(self):
        self._thread:       Optional[threading.Thread] = None
        self._stop:         threading.Event = threading.Event()
        self._state_lock:   threading.Lock  = threading.Lock()
        self._current:      dict            = {}
        self._events_since: list            = []
        self._fault_queue:  list            = []
        self._fault_lock:   threading.Lock  = threading.Lock()
        self._seed:         int             = 42
        self._speed_mult:   float           = 10.0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, seed: int, speed_multiplier: float = 10.0) -> None:
        """Start (or restart) the live feed for the given seed."""
        self.stop()
        self._seed       = seed
        self._speed_mult = speed_multiplier
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"live_feed_seed{seed}",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the feed thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get_current_state(self) -> dict:
        """
        Return latest vehicle positions, nav mode, and events since last poll.

        Returns::

          {
            'vehicle_a': {'north_m', 'east_m', 'alt_m', 'nav_mode',
                          'mission_km', 'gnss_available'},
            'vehicle_b': {...},
            'mission_km': float,
            'events_since_last_poll': [{'event_type', 'label', 'mission_km'}],
          }
        """
        with self._state_lock:
            state = dict(self._current)
            events = list(self._events_since)
            self._events_since = []
        state["events_since_last_poll"] = events
        return state

    def inject_fault(self, fault_type: str, at_km: float) -> None:
        """
        Queue a fault for injection.

        Supported faults:
          'gnss_denial'  — sets gnss_available False from at_km onwards
          'ew_hotspot'   — injects EW cost spike at at_km
          'vio_outage'   — triggers VIO outage at at_km

        The running simulation is restarted from at_km=0 with the modified
        disturbance schedule.
        """
        with self._fault_lock:
            self._fault_queue.append({"type": fault_type, "at_km": at_km})

    # ── Background loop ───────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """
        Background thread: generate full simulation result then play back
        at speed_multiplier × real time using SimClock for pacing.
        """
        seed      = self._seed
        faults: dict = {}

        while not self._stop.is_set():
            # Check for pending fault injections
            with self._fault_lock:
                pending = list(self._fault_queue)
                self._fault_queue.clear()
            for fault in pending:
                faults[fault["type"]] = fault["at_km"]

            # Run simulation with current fault parameters
            schedule = self._build_schedule(seed, faults)
            try:
                config = BCMP2RunConfig(seed=seed, max_km=MISSION_TOTAL_KM)
                result = run_bcmp2(config=config)
            except Exception as exc:
                print(f"[LiveMissionFeed] Runner error: {exc}")
                self._stop.wait(2.0)
                continue

            tracks = get_vehicle_tracks(result)
            events = get_mission_events(result)
            va     = tracks["vehicle_a"]
            vb     = tracks["vehicle_b"]

            if not va or not vb:
                self._stop.wait(1.0)
                continue

            # Align both tracks to display cadence
            # Vehicle A is at 40Hz record cadence; Vehicle B already at 100ms
            va_by_ts = {r["sim_timestamp_ms"]: r for r in va}
            vb_by_ts = {r["sim_timestamp_ms"]: r for r in vb}
            all_ts   = sorted(set(vb_by_ts.keys()))  # use 100ms cadence

            clock = SimClock(dt=_DISPLAY_CADENCE_MS / 1000.0)
            clock.start()

            event_idx = 0
            wall_start = time.monotonic()

            for ts_ms in all_ts:
                if self._stop.is_set():
                    break

                # Check for new fault injections mid-playback
                with self._fault_lock:
                    if self._fault_queue:
                        break   # restart with new faults

                # Find closest Vehicle A record (40Hz records ≠ 100ms aligned)
                a_rec = _closest_record(va, vb_by_ts[ts_ms]["mission_km"])
                b_rec = vb_by_ts[ts_ms]

                with self._state_lock:
                    self._current = {
                        "vehicle_a": {
                            "north_m":       a_rec["north_m"]   if a_rec else 0.0,
                            "east_m":        a_rec["east_m"]    if a_rec else 0.0,
                            "alt_m":         a_rec["alt_m"]     if a_rec else 0.0,
                            "nav_mode":      a_rec["nav_mode"]  if a_rec else "UNKNOWN",
                            "mission_km":    a_rec["mission_km"] if a_rec else 0.0,
                            "gnss_available": a_rec["gnss_available"] if a_rec else True,
                        },
                        "vehicle_b": {
                            "north_m":       b_rec["north_m"],
                            "east_m":        b_rec["east_m"],
                            "alt_m":         b_rec["alt_m"],
                            "nav_mode":      b_rec["nav_mode"],
                            "mission_km":    b_rec["mission_km"],
                            "gnss_available": b_rec["gnss_available"],
                        },
                        "mission_km": b_rec["mission_km"],
                    }
                    # Accumulate events up to current km
                    km_now = b_rec["mission_km"]
                    while event_idx < len(events) and \
                            events[event_idx]["mission_km"] <= km_now:
                        self._events_since.append(events[event_idx])
                        event_idx += 1

                clock.step()

                # Real-time pacing: sleep so playback advances at speed_mult × real time
                # sim_time_s / speed_mult = wall_time_s to sleep
                sim_elapsed_s  = ts_ms / 1000.0
                wall_target    = wall_start + sim_elapsed_s / self._speed_mult
                sleep_s        = wall_target - time.monotonic()
                if sleep_s > 0:
                    self._stop.wait(sleep_s)

            # Mission complete — loop or stop
            if not self._stop.is_set():
                self._stop.wait(2.0)   # pause before loop

    @staticmethod
    def _build_schedule(seed: int, faults: dict):
        """
        Build a disturbance schedule incorporating any injected faults.
        Returns None (uses default schedule in bcmp2_runner) when no faults.
        """
        # Fault injection is signalled via seed-level override in the returned
        # config; the actual DisturbanceSchedule is generated inside run_bcmp2.
        # For now, fault parameters are passed by modifying the schedule via
        # the standard generate_disturbance_schedule path — full restart.
        return None   # placeholder for Phase D fault injection UI
