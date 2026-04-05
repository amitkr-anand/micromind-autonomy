# Sprint D — Architecture Specification
Date: 2026-04-05
Closes: OI-16 (RC-11), OI-17 (RC-7), OI-18 (RC-8)
Enables: CP-3 declaration
Regression baseline entering Sprint D: 283 tests

## Four Deliverables — Strict Dependency Order

1. integration/pipeline/setpoint_coordinator.py
   External coordinator wiring LivePipeline.setpoint_queue
   to MAVLinkBridge.update_setpoint(). RC-11c prerequisite.
   Must NOT modify LivePipeline or MAVLinkBridge.

2. tests/test_prehil_rc11.py
   RC-11a through RC-11e.

3. tests/test_prehil_rc7.py
   RC-7 timestamp monotonicity injection.

4. tests/test_prehil_rc8.py
   RC-8 logger non-blocking 200 Hz 60 s drop-rate test.

## SetpointCoordinator Design

File: integration/pipeline/setpoint_coordinator.py

Class: SetpointCoordinator
Constructor: __init__(self, pipeline, bridge, poll_hz=50.0)
Methods: start(), stop()
Properties: setpoints_forwarded (int), setpoints_dropped (int)

Internal loop — runs in daemon thread:
  Poll setpoint_queue at poll_hz.
  Drain all pending entries — keep only most recent.
  Call bridge.update_setpoint(sp.x_m, sp.y_m, sp.z_m).
  Increment setpoints_forwarded on success.
  Increment setpoints_dropped for each stale entry discarded.

File header must state:
  Closes RC-11c wiring gap (Sprint D code review 4972110).
  Does not modify LivePipeline or MAVLinkBridge.
  External coordinator pattern.

## RC-11 Test Specifications

### RC-11a — OUTAGE detection latency (SD-01)
Stimulus:
  Start LivePipeline with OfflineVIODriver at 25 Hz.
  After 2 s nominal, stop frame delivery.
  Poll vio_mode.current_mode at 50 Hz for up to 1 s.
Assertions:
  current_mode == VIOMode.OUTAGE within 500 ms.
  n_outage_events == 1.
  Log record containing VIO_OUTAGE_DETECTED present.
  Use logging.handlers.MemoryHandler to capture log output.

### RC-11b — ESKF numerical stability (SD-02) — HIGHEST RISK
Stimulus:
  Start LivePipeline with OfflineVIODriver.
  After 2 s nominal, stop frames (induce OUTAGE).
  Hold OUTAGE for 30 s — IMU-only propagation at 200 Hz.
  Read ESKF state externally at each propagation step.
Assertions:
  np.isfinite(state.p).all() at every step.
  np.isfinite(state.v).all() at every step.
  np.isfinite(state.q).all() at every step.
  np.isfinite(state.ba).all() at every step.
  np.isfinite(state.bg).all() at every step.
  Zero NaN events across 6000 steps.
  If any NaN: FAIL immediately, record step and component.
Mandatory caveat in test output:
  RC-11b validated on Ryzen 7 9700X at 200 Hz SIL.
  Jetson Orin timing margins not characterised (OI-25).

### RC-11c — Setpoint continuity during OUTAGE (SD-03)
Prerequisite: SetpointCoordinator wired and running.
Stimulus:
  Start LivePipeline + MAVLinkBridge (SITL mode).
  Start SetpointCoordinator.
  After 2 s nominal, inject OUTAGE.
  Hold OUTAGE for 15 s.
  Monitor bridge setpoint values at 10 Hz.
Assertions:
  setpoints_forwarded > 0 during OUTAGE.
  Bridge setpoint values finite at every sample.
  Bridge setpoint values change between samples
    (not frozen at last VIO position).
  Setpoint rate >= 20 Hz
    (forwarded count / duration >= 20).

### RC-11d — RESUMPTION correctness (SD-04)
Stimulus:
  Continue from RC-11c state (OUTAGE active at 15 s).
  Resume VIO frame delivery.
  Poll vio_mode.current_mode at 50 Hz for up to 5 s.
Assertions:
  Transition OUTAGE to RESUMPTION within 500 ms
    of first resumed frame.
  Transition RESUMPTION to NOMINAL within 2 s.
  Log record VIO_RESUMPTION_STARTED present.
  Log record VIO_NOMINAL_RESTORED present.
  ESKF position change at resumption <= 50 m.
  np.isfinite(state.p).all() throughout.

### RC-11e — Regression baseline unaffected (SD-05)
Not a new test. Run after RC-11a through RC-11d pass:
  run_s5_tests.py    must be 119/119.
  run_s8_tests.py    must be 68/68.
  run_bcmp2_tests.py must be 90/90.
  test_s5_l10s_se_adversarial.py must be 6/6.

## RC-7 Test Specification (SD-06)

File: tests/test_prehil_rc7.py

Stimulus:
  Instantiate OfflineVIODriver.
  Deliver 10 valid frames with monotonically
    increasing timestamps (t=0.0, 0.04, ... 0.36).
  Inject ONE non-monotonic frame: t_bad = t_last - 0.01.
  Deliver 10 more valid frames after the bad one.
Assertions:
  Non-monotonic frame is rejected.
  driver.ifm01_rejections == 1 after injection.
  driver.ifm01_rejections == 0 before injection.
  Frames after bad timestamp accepted normally.
  Log record containing IFM-01 or monotonicity present.
Mandatory in test docstring:
  mark_send confirmed natively integrated at
  mavlink_bridge.py lines 358-359 (Sprint D code
  review). CP-2 asterisk withdrawn. RC-7 tests
  IFM-01 guard directly, not setpoint latency.

## RC-8 Test Specification (SD-07)

File: tests/test_prehil_rc8.py

Stimulus:
  Instantiate logger (check fusion_logger.py).
  Start logger.
  Submit log entries at 200 Hz for 60 s (12000 total).
  Each entry is a minimal valid log record.
  Measure wall-clock time per submission call.
  Measure total entries written vs submitted.
Assertions:
  log_completeness = written / submitted >= 0.99.
  No single submission call blocks for > 5 ms.
  drop_count == 0 or within 1% tolerance.
Mandatory caveat in test output:
  RC-8 validated on Ryzen 7 9700X. Jetson Orin
  renice +10 / 2-core taskset variant not
  characterised (OI-25). This result does not
  constitute Jetson clearance.

## Nine Acceptance Gates

SD-01: RC-11a OUTAGE detected within 500 ms, log present
SD-02: RC-11b zero NaN across 30 s x 200 Hz
SD-03: RC-11c setpoints forwarded, finite, rate >= 20 Hz
SD-04: RC-11d NOMINAL restored within 2 s, no jump > 50 m
SD-05: RC-11e 283 existing tests unchanged
SD-06: RC-7 IFM-01 rejects non-monotonic, counter == 1
SD-07: RC-8 completeness >= 0.99, no call > 5 ms
SD-08: SetpointCoordinator does not touch frozen files
SD-09: Jetson caveat in RC-11b and RC-8 output

## Do Not Touch
- error_state_ekf.py (NaN guard is in the test)
- mavlink_bridge.py (wiring is external coordinator)
- live_pipeline.py (queue already producing)
- Any existing test file
- vio_mode_FROZEN_BASELINE.py backup
