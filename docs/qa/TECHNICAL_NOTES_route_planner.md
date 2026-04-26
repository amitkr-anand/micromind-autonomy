# Technical Notes — core/route_planner/route_planner.py
Authority: Deputy 1 | Date: 26 April 2026 | Req: PLN-02

## 1. ETA Rollback Design (R-03, OI-56)

**Decision:** _eta_s added as live attribute to RoutePlanner,
computed after every successful route: route_length_m /
_cruise_speed_ms. Snapshotted before retask, restored in
_rollback() as action 5.

**Why not computed from route on demand:** SRS Appendix B
ROLLBACK action (5) explicitly requires "Restore ETA" as a
named action. Derivability from route is irrelevant to
compliance — the SRS requires explicit restoration.

**Why constructor parameter not config dict:** RoutePlanner
has no config dict. Constructor parameter injection is the
existing pattern (consistent with other modules). Default
CRUISE_SPEED_MS_DEFAULT = 27.78 m/s = AVP-02 100 km/h.

**snap_target = None:** RoutePlanner does not hold a
_current_target attribute. Target is passed as a parameter
to retask(). The RETASK_ROLLBACK payload field is present
(SRS compliance) but value is None (honest deviation).
Accepted by Deputy 1 — field exists, value is accurate.

## 2. TERRAIN_GEN_FAIL / COMMIT_FAIL Try/Except Pattern

**Decision:** Both _terrain_regen_fn() and _px4_upload_fn()
wrapped in try/except Exception. On failure: specific event
logged (RETASK_TERRAIN_GEN_FAILED or RETASK_COMMIT_FAILED),
_rollback(snap_eta_s) called, RETASK_ROLLBACK emitted,
_cleanup_route_fragments() called, return False.

**Why try/except Exception not specific exception types:**
The callback functions are injected — their exception types
are not known to RoutePlanner. Catching Exception is the
correct pattern for injected callbacks.

**Why BaseException not caught:** SIGKILL, KeyboardInterrupt,
SystemExit must propagate. Catching BaseException would
suppress process-level signals. Exception is the correct
boundary.

## 3. RETASK_ROLLBACK Payload — 10 Fields

Fields: event, req_id, severity, module_name, timestamp_ms,
eta_s_restored, reason, previous_target, restored_ew_map_age_ms,
restored_terrain_phase.

**Why 10 fields vs original 6:** SRS Appendix B ROLLBACK
state requires reason, previous_target, restored_ew_map_age_ms,
restored_terrain_phase. These were absent from the original
implementation. Added at 17330fa to achieve SRS compliance.

**reason values:** "TIMEOUT" | "DEAD_END" | "TERRAIN_GEN_FAIL"
| "COMMIT_FAIL" — one per call site, no ambiguity.

## 4. SR-01 Compliance — threading.Event not self._clock.now()

**SR-01:** No while-loop or polling construct using
self._clock.now() (simulation clock) may be placed inside any
synchronous method.

**Why:** self._clock.now() in a synchronous context returns
a fixed value within the call stack — a loop condition using
it never becomes False, producing an infinite busy-wait.
This was the actual failure mode in W1-P07 (reverted before
commit). threading.Event().wait(timeout) uses the OS clock
and is not affected by simulation clock state.

**Application in RoutePlanner:** The retask() method uses
the simulation clock only for elapsed time measurement, not
as a loop wait condition. The constraint applies to wait loops,
not to elapsed time checks that terminate on timeout.

## 5. _rollback() as Nested Function

**Decision:** _rollback() is a nested function inside retask(),
capturing snapshot variables by Python closure.

**Why nested not method:** _rollback() has no meaning outside
retask() — the snapshot variables only exist during a retask
attempt. Making it a method would expose it publicly and
require passing all snapshot variables as parameters. The
nested pattern is architecturally clean: rollback cannot be
called from outside the retask context.

**Governance compliance:** CGM §2.3 requires functions with
side effects to be explicitly scoped. Nested function scope
is the strictest available in Python.
