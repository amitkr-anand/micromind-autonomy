## OI-31 Demo Design Decisions
Date: 12 April 2026

---

## Run 1 — Gazebo SITL Overlay

**Why matplotlib alongside Gazebo rather than Gazebo-native:**
Speed of delivery. Gazebo plugin development is multi-day work requiring
C++ compilation, plugin registration, and Gazebo API iteration.
`demo_overlay.py` achieves the same customer-visible result in hours —
a real-time two-vehicle track display with altitude bar and phase annotation
running at 10 Hz update rate alongside the live SITL. The overlay is
Orchestration Box only — no core/ imports. It reads MAVLink LOCAL_POSITION_NED
directly from both PX4 SITL instances (ports 14540 and 14541).

**Waypoint source:** `run_mission.py` module-level constants and
`ellipse_waypoints()` function. No hardcoding; if the demo ellipse changes,
the overlay updates automatically at next launch.

**Camera setter:** `baylands_demo_camera.py` uses `gz service -s /gui/move_to_pose`
(same service pattern as `launch_two_vehicle_sitl.sh`). Non-fatal: if the
Gazebo GUI is not running (HEADLESS mode), the script prints a warning and
exits 0.

---

## Run 2 — Three Mode Architecture

| Layout | Mode         | Distance | Purpose                        |
|--------|--------------|----------|--------------------------------|
| A      | Replay       | 150 km   | Storytelling — full mission    |
| B      | Live         | 150 km   | Engineering — real-time replay |
| C      | Comparative  | 50 km    | Outcomes — GNSS denial impact  |

**Why 50 km for Layout C:**
Sufficient to demonstrate the full GNSS denial sequence (denial at km 30,
drift accumulation to km 50) and the MicroMind advantage (VIO correction,
EW rerouting) without the full 150 km computation time. Terminal phase
evidence (SHM, L10s-SE gate) is deferred to Layout A/B storytelling.

**Position log strategy:**
- Vehicle A: `baseline_nav_sim.to_kpi_dict()` serialises `BaselineRunResult.states`
  (recorded every 40 sim steps at 200 Hz = every 5.56 m at mission speed).
- Vehicle B: synthesised in `bcmp2_runner._build_vehicle_b_position_log()` from
  the planned route and disturbance schedule. `bcmp1_runner.py` is frozen and
  produces no position log. The synthesised track is Orchestration Box data
  only — it must not be cited as algorithm-level evidence.

**Pre-computed seed files:**
  `docs/qa/bcmp2_kpi_seed_42.json`   — nominal reference run
  `docs/qa/bcmp2_kpi_seed_101.json`  — alternate weather / sensor noise
  `docs/qa/bcmp2_kpi_seed_303.json`  — degraded / stress profile

---

## V-02 Overlay Rule Compliance

All event marker positions derive from KPI JSON log data only.
The rule: no event position may deviate from the logged event by more than
one animation frame (100 ms). Enforced in
`demo_data_pipeline.get_mission_events()`:

- **GNSS DENIED**: first `gnss_available=False` record in `vehicle_a.position_log`.
- **EW REROUTE**: jammer activation times from BCMP-1 scenario constants
  (T+8min, T+11min), converted to km via `VEHICLE_SPEED_MS`. Positions
  cross-referenced against `vehicle_b_position_log`.
- **DEM CORRECTION**: VIO outage end time from `disturbance_schedule.vio_outages`,
  closest record found in `vehicle_b_position_log`.
- **SHM ACTIVE**: Phase 5 boundary (km 120), closest record in
  `vehicle_b_position_log`.
- **RETASK**: from `vehicle_b.criteria.RETASK_EVENT` if present.

Zero hardcoded km values in `get_mission_events()`.

---

## V-03 Sync Rule

Vehicle B position records in the KPI JSON are at 200m spacing (0.2km cadence
from `_build_vehicle_b_position_log`). `get_vehicle_tracks()` linearly
interpolates these to 100ms display cadence for animation.

Interpolated values are never logged as mission data. Phase boundary events
force aligned records in both vehicles.

---

## Deferred to Phase D

- Polished UI controls (play/pause/scrub, fault injection panel)
- Live GNSS denial toggle mid-replay
- Layout A/B/C mode switching in a single window
- Vehicle B algorithm-level position logging (requires bcmp1_runner extension,
  not frozen in Phase D)
