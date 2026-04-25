# SRS v1.4 Delta — §16 Corridor Violation Ownership Row
Authority: Deputy 1 (Architect Lead)
Date: 26 April 2026
Status: CLOSED — OI-40

## New §16 Row: Corridor Violation (Predicted)

| Role | Owner | Module | Evidence |
|---|---|---|---|
| Detects | Navigation Manager | baseline_nav_sim.py / live NavigationManager — computes cross_track_error_m vs corridor boundary, sets corridor_violation=True on SystemInputs | SystemInputs.cross_track_error_m added at 3e79805 |
| Decides | Mission Manager (NanoCorteXFSM) | core/state_machine/state_machine.py — unconditional ABORT from 5 states: NOMINAL (297), EW_AWARE (320), GNSS_DENIED (361), NAV_TRN_ONLY (399), SILENT_INGRESS (440) | state_machine.py |
| Executes | Mission Manager (NanoCorteXFSM) | _transition(NCState.ABORT, "CORRIDOR_VIOLATION") | state_machine.py |
| Logs | Mission Manager (NanoCorteXFSM) | _log_corridor_violation_event() emits LogCategory.SYSTEM_ALERT MissionLogEntry with event, active_state, trigger, mission_km, bim_state | 9d99a75 |
| Consumes | Navigation Manager (hold position), PX4 Bridge (ABORT setpoint), All modules (log and halt) | Standard ABORT consumers | All modules |

## OI-55 Resolution
cross_track_error_m added to SystemInputs at 3e79805.
The structured CORRIDOR_VIOLATION log event payload now
includes cross_track_error_m.

## SRS v1.4 Incorporation Status
This row is hereby incorporated by reference pending
formal SRS v1.4 document revision. This file is the
authoritative record until SRS is republished.

OI-40: CLOSED.
OI-55: CLOSED (resolved at 3e79805).
