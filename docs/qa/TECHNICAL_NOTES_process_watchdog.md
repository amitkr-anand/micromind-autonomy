# Technical Notes — core/watchdog/process_watchdog.py
Authority: Deputy 1 | Date: 26 April 2026 | Req: RS-03, E-03

## 1. NOT_RESTARTABLE Path Design

**Decision:** On NOT_RESTARTABLE classification: log
ESKF_CORE_FAILURE (CRITICAL), call abort_fn(), return
immediately. No restart attempt under any circumstance.

**Why immediate abort, no retry:** The ESKF (Error-State
Kalman Filter) is the navigation state estimator. A failed
ESKF produces corrupted position/velocity/attitude estimates.
Any mission continuation with corrupted navigation state
risks uncontrolled flight into civilian areas or friendly
assets. The correct response is immediate mission abort,
not retry. A failed ESKF cannot be restarted safely while
the vehicle is airborne — the state is lost.

**Why ESKF_CORE_FAILURE at CRITICAL severity:** All other
events are WARNING. CRITICAL is reserved for conditions
that require immediate mission termination with no recovery
path. ESKF failure is the only such condition in the
current implementation.

**Safety invariant:** restart_fn must NEVER be called when
restartability_class == NOT_RESTARTABLE. This invariant is
tested explicitly (test confirms restart_fn call_count == 0).

## 2. PROCESS_REGISTRY Classification Rationale

| Process | Class | Rationale |
|---|---|---|
| ESKFCore | NOT_RESTARTABLE | Navigation state is lost on failure. No safe restart while airborne. |
| NavigationManager | RESTARTABLE_WITH_SHM | Holds nav state. Checkpoint restore recovers state. SHM required to preserve mission context. |
| MissionManager | RESTARTABLE_WITH_SHM | Holds mission state (phase, waypoint, envelope). Checkpoint restore recovers. |
| PX4Bridge | RESTARTABLE_WITH_SHM | Holds OFFBOARD link state. Restart requires SHM to preserve mission phase so D8a gate re-evaluates correctly. |
| EWManager | RESTARTABLE_WITHOUT_SHM | EW map is rebuilt from observations — no persistent state requires checkpoint. Restart is stateless. |
| LogBus | RESTARTABLE_WITHOUT_SHM | Logging infrastructure. Restart loses buffered logs only — not mission-critical state. |

## 3. Daemon Thread Dispatch

**Decision:** on_process_failure() dispatches _handle_failure()
as a daemon thread. Returns immediately to caller.

**Why daemon thread:** The caller (watchdog monitor loop) must
not block. If restart_fn takes 2s per attempt and max_retries=3,
blocking would freeze the monitor loop for 6s. Daemon thread
ensures the monitor loop continues observing other processes
during recovery.

**Why daemon=True:** If the main process exits, daemon threads
are killed automatically. A non-daemon thread would prevent
clean shutdown during abort sequences.

## 4. SIL Scope Boundary

**What UT-RS-03 tests:** The decision logic — given a process
failure event and a restartability_class, does the watchdog
select the correct recovery path? This is fully SIL-testable
with mock restart_fn and checkpoint_restore_fn.

**What UT-RS-03 does NOT test:** Real SIGKILL delivery to a
real process. Real process restart via OS exec(). Real
checkpoint restore from disk after process restart. These
require a production watchdog daemon — Phase D.

**Why this boundary is correct:** The decision logic is the
safety-critical component. The OS-level restart mechanism is
infrastructure. Testing the decision logic in SIL with mocks
is standard practice for safety-critical systems (analogous
to testing an autopilot flight law in simulation before HIL).

## 5. Relationship to Unified State Vector

**Important boundary:** ProcessWatchdog does NOT consume the
Unified State Vector. It receives process failure events via
on_process_failure() callback — a direct call from the watchdog
monitor, not mediated through the USV.

**Why:** The USV is a navigation/state data structure. Process
health is an infrastructure concern, not a navigation concern.
Routing process failure through the USV would violate the
AD-01 architectural separation between navigation state and
system health.

**What this means for Deputy 2's audit:** The ProcessWatchdog
correctly does NOT react to MAVLink-level events. It reacts
to process-level failure signals. This is the intended design.
