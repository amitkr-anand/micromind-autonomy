# simulation/TECHNICAL_NOTES.md
**Module:** simulation (run_mission.py, run_demo.sh)
**Last updated:** 10 April 2026

---

## OODA loop rationale

The simulation module is the demo integration layer. It is NOT part of
the onboard autonomy stack. It drives PX4 SITL via MAVLink to exercise
the two-vehicle GPS denial scenario for demonstration and verification
purposes.

**Observe:** MAVLink `LOCAL_POSITION_NED` at 20 Hz per vehicle via
`_read_local_pos()`. Heartbeat receipt confirms vehicle liveness.

**Orient:** Waypoint advance criterion (Euclidean distance < 15 m to
current NED setpoint) and GPS denial state machine (elapsed time vs
`args.gps_denial_time`).

**Decide:** FSM implicit in each mission thread: CONNECT → EKF2_WAIT →
BARRIER → OFFBOARD → CLIMB → LAP_LOOP → COMPLETE.

**Act:** `SET_POSITION_TARGET_LOCAL_NED` at 20 Hz, `MAV_CMD_DO_SET_MODE`
for OFFBOARD engagement, `EKF2_GPS_CTRL` PARAM_SET for GPS denial.

---

## Known magic numbers (pending config governance)

| Constant | Value | Location | OI |
|---|---|---|---|
| `MISSION_TIMEOUT_S` | 300 s | `main()` line 644 | OI-37 |

**OI-37 note:** `MISSION_TIMEOUT_S = 300` is a hardcoded default. It
covers the 2-lap GPS denial scenario with margin (observed lap time ~107 s
per vehicle in live SITL, QA-014). No `mission_timeout` key exists in any
config file at time of writing. This constant must be moved to config and
documented in the mission_config schema before external demo builds.

---

## OI-36 fix (10 April 2026, commit TBD)

**Problem:** `t_a.join()` and `t_b.join()` in `main()` had no timeout.
If a vehicle thread hung (e.g., Vehicle A OFFBOARD failsafe causes landing,
waypoint distance never < 15 m, lap_count never increments), `main()` would
block indefinitely and MISSION PASS/FAIL/ABORT would never be reached.

**Fix:** `t_a.join(timeout=MISSION_TIMEOUT_S)` followed by
`t_a.is_alive()` check. If alive after timeout, prints
`[MISSION] ABORT — Vehicle A thread did not complete within timeout.`
and calls `os._exit(2)`. Same pattern for `t_b`. Exit code 2 distinguishes
timeout abort from EKF2/arming abort (exit 1) and PASS (exit 0).

**Note:** `os._exit(2)` bypasses Python's atexit/finalizer phase.
Required because `_hb_thread` daemon threads inside each mission function
hold open pymavlink UDP sockets. `sys.exit()` would block on socket cleanup
(same root cause as EF-02 / commit 7ed5a8e).
