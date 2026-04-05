# Sprint D — Pre-Specification Code Review
**Date:** 2026-04-05  
**Type:** Code review — read only. No changes, no commits from this session except this document.  
**Purpose:** Answer QA agent questions before Sprint D specification is written.  
**Covers:** OI-16 (RC-11), OI-17 (RC-7), OI-18 (RC-8)

---

## Question Set 1 — VIONavigationMode and OUTAGE path

**File:** `core/fusion/vio_mode.py` (frozen)

### 1a. States defined by VIONavigationMode

Three states, defined as `VIOMode` enum at line 79:

```python
class VIOMode(Enum):
    NOMINAL    = auto()    # VIO active, position strongly observed
    OUTAGE     = auto()    # VIO absent, position confidence degrading
    RESUMPTION = auto()    # VIO returned, stabilisation in progress
```

### 1b. NOMINAL → OUTAGE transition path

Triggered in `tick(dt)` at lines 158–163:

```python
if (self._mode is VIOMode.NOMINAL
        and self._dt_since_vio >= self._outage_threshold_s):
    self._mode = VIOMode.OUTAGE
    self._in_outage = True
    self._n_outage_events += 1
```

- `_dt_since_vio` accumulates via `tick()` on each IMU step.
- Default threshold: `VIO_OUTAGE_THRESHOLD_S = 2.0` s (line 54).
- Transition is instantaneous at the first `tick()` call where the threshold is exceeded.
- **No explicit latency value is specified** beyond the threshold itself — latency is bounded by the IMU step rate × threshold, approximately 2.0 s at default.

### 1c. OUTAGE → RESUMPTION → NOMINAL (recovery path)

Handled in `on_vio_update(accepted, innov_mag)` at lines 202–228:

1. **OUTAGE → RESUMPTION** (lines 202–212): on first `accepted=True` update after outage, mode transitions to RESUMPTION. `_dt_since_vio` reset to 0.0. Innovation spike checked.
2. **RESUMPTION → NOMINAL** (lines 214–223): on each subsequent `accepted=True` update, `_resumption_count` increments. When `_resumption_count >= VIO_RESUMPTION_CYCLES` (default 1), mode returns to NOMINAL.

Default path: **one accepted VIO update** in RESUMPTION → back to NOMINAL.

### 1d. Timeout parameter

`VIO_OUTAGE_THRESHOLD_S = 2.0` seconds (line 54).  
Source: S-NEP-07 L-07.  
Configurable at construction via `outage_threshold_s` parameter.

### 1e. Logging on OUTAGE detection / RESUMPTION

**⚠️ SURPRISE — CRITICAL FINDING:**  
`vio_mode.py` contains **zero logging calls**. No `print()`, no `logging.*`, no event emission of any kind. The transition at lines 158–163 increments `_n_outage_events` (an internal counter) but emits nothing. RESUMPTION and NOMINAL recovery at lines 209–223 are also silent.

The module exposes diagnostic properties (`n_outage_events`, `n_spike_alerts`, `max_dt_since_vio`, `max_drift_envelope_m`) for callers to poll, but the module itself is completely silent on transitions.

Logging is the caller's responsibility. In `inject_outage.py`, the demo script observes mode via `pipeline.health().vio_mode` and calls `log_event()` explicitly. No standing log contract exists.

**Implication for Sprint D:** RC-11 test requires observing OUTAGE/RESUMPTION transitions. Tests must instrument the caller, not the module.

---

## Question Set 2 — ESKF state during OUTAGE

**File:** `core/ekf/error_state_ekf.py` (frozen)

### 2a. ESKF state vector structure (15 states)

```
[0:3]   δp   — position error (m)
[3:6]   δv   — velocity error (m/s)
[6:9]   δθ   — attitude error (rad, small-angle)
[9:12]  δba  — accelerometer bias error (m/s²)
[12:15] δbg  — gyroscope bias error (rad/s)
```

Covariance: `P` is 15×15. Initial diagonals at lines 71–75:  
pos=1.0 m², vel=0.1 (m/s)², att=(1e-3)² rad², acc_bias=(0.1)² (m/s²)², gyro_bias=(0.01)² (rad/s)².

### 2b. Finite-value assertions

**No `np.isfinite` or equivalent assertion anywhere in `error_state_ekf.py`.**

The demo script `inject_outage.py` has a NaN check at line 176 of that file:
```python
if not all(math.isfinite(float(x)) for x in self._state.p):
    _nan_detected = True
```
But this is in the patched nav loop of the demo script, **not in the ESKF file itself**.

There is also no NaN guard in `update_vio()` (lines 208–235), `propagate()` (lines 118–130), or `inject()` (lines 160–183).

**Implication for Sprint D:** RC-11 must either add a finite-value guard or test that the existing code path does not produce NaN. There is currently no assertion catching this.

### 2c. IMU-only propagation method

`propagate(self, state, acc_body, dt)` at line 118:

```python
def propagate(self, state, acc_body, dt):
    F = self._build_F(state, acc_body, dt)
    Q = self._build_Q(dt)
    self.x = F @ self.x
    self.P = F @ self.P @ F.T + Q
```

Called every IMU step. Does not require VIO — runs independently.

### 2d. OUTAGE-aware code path in ESKF

**None.** The ESKF has no concept of VIO modes. During OUTAGE, the ESKF simply does not receive `update_vio()` calls — the calling code skips VIO updates when the driver returns `valid=False` or when `_vio_paused=True` (as in the demo). The ESKF continues IMU-only propagation via `propagate()` with covariance growing according to `_Q`. No OUTAGE-specific branch exists in the ESKF file.

---

## Question Set 3 — MAVLink bridge setpoint loop

**File:** `integration/bridge/mavlink_bridge.py`

### 3a. Setpoint publishing method/thread and rate

`_setpoint_loop()` method, runs as **T-SP** thread (line 323).  
Target rate: **20 Hz** (`_SP_INTERVAL_S = 0.05` s, line 85).  
Time-driven (FM-4 compliance): loop advances by `_SP_INTERVAL_S` regardless of execution time.

### 3b. VIO mode / ESKF state check before publishing

**None.** T-SP reads `_setpoint_x_m/_y_m/_z_m` under `_setpoint_lock` and publishes unconditionally (lines 336–356). It does not inspect VIO mode, ESKF state, or bridge state. It will send whatever values are currently in the setpoint fields, whether stale or NaN.

One guard exists: `self._sp_paused.is_set()` (line 341) — if set, the loop sleeps and skips sending. This is a manual pause mechanism (not VIO-mode driven).

### 3c. Data path from ESKF to MAVLink

```
ErrorStateEKF.inject(state)
    → state.p updated (position corrections applied)
LivePipeline._enqueue_setpoint()    [live_pipeline.py:330]
    → Setpoint(x_m=pos[0], y_m=pos[1], z_m=-pos[2])
    → _setpoint_queue.put_nowait(setpoint)  ← internal bounded queue
    [DISCONNECTION: no automatic consumer connecting queue → bridge]
External caller (not yet wired)
    → bridge.update_setpoint(x_m, y_m, z_m)
    → MAVLinkBridge._setpoint_x_m/_y_m/_z_m (under _setpoint_lock)
MAVLinkBridge._setpoint_loop()
    → mav.set_position_target_local_ned_send(...)
```

**⚠️ SURPRISE — ARCHITECTURE FINDING:**  
`LivePipeline._setpoint_queue` and `MAVLinkBridge._setpoint_x_m/_y_m/_z_m` are **not automatically connected**. The pipeline enqueues `Setpoint` dataclasses to an internal `queue.Queue`. T-SP reads from separate private floats updated by `update_setpoint()`. In the demo `inject_outage.py`, `bridge.update_setpoint()` is called manually at hard-coded positions — it does not read from the pipeline queue. An external consumer loop connecting the two does not exist in the current codebase.

### 3d. mark_send integration

**Yes — integrated.** Lines 358–359:

```python
if self._latency_monitor is not None:
    self._latency_monitor.mark_send()
```

Called immediately after each successful `set_position_target_local_ned_send()`.

**⚠️ CONFLICT WITH OI-21:** The context file Open Items states: *"mark_send not natively integrated into mavlink_bridge setpoint loop — CP-2 latency result has asterisk."* This is **contradicted by the current code**. `mark_send` is natively integrated at lines 358–359. OI-21 appears to be stale or was already resolved. QA agent should verify and close OI-21 if the code is correct.

### 3e. NaN guard in setpoint loop

**No guard.** T-SP reads `x_m, y_m, z_m` directly and passes them to `set_position_target_local_ned_send()` with no finite-value check. If `state.p` contains NaN (e.g., ESKF divergence), it will propagate through `_enqueue_setpoint()` → `update_setpoint()` → T-SP → PX4. No exception is expected from pymavlink for NaN float values — it will transmit silently.

**Implication for Sprint D:** RC-11b (setpoint continuity under VIO OUTAGE) must verify that the last-known finite setpoint is held rather than propagating a NaN or zero-reset value.

---

## Question Set 4 — Logger architecture

**Files:** `integration/bridge/bridge_logger.py`, `integration/pipeline/live_pipeline.py`

### 4a. Logger class name and file path

- **`BridgeLogger`** — `integration/bridge/bridge_logger.py`  
  Logs MAVLink bridge events (heartbeat TX/RX, setpoints, commands, mode transitions, staleness).

- The navigation loop (`LivePipeline`) does not use `BridgeLogger` directly. It produces to `_setpoint_queue` and tracks `_queue_drop_count` internally.

### 4b. Synchronous or asynchronous

**Asynchronous (non-blocking).**  
`log()` calls `self._queue.put_nowait(entry)` at line 126 — returns immediately.  
T-LOG consumer thread (`_log_loop()`) at line 212 dequeues and writes JSON-lines to disk independently.

### 4c. Logging rate and 200Hz path

`BridgeLogger` logs bridge events at their natural rate (2Hz heartbeats, 20Hz setpoints from T-SP).  
**There is no 200Hz logging path.** T-NAV runs at 200Hz but does not call `BridgeLogger`. The 200Hz loop metrics are tracked in `LivePipeline._loop_count` and `_queue_drop_count` (internal counters), not logged to disk per-cycle.

**Implication for Sprint D RC-8:** A formal 200Hz non-blocking logger does not currently exist for the navigation loop. RC-8 requires a 60-second non-blocking test at 200Hz. The BridgeLogger infrastructure (queue + T-LOG) could be the basis, but it is not currently wired to T-NAV.

### 4d. Drop-rate counter

**Yes — exists.** `BridgeLogger._drop_count` (line 77), incremented at line 128 on `queue.Full`. Exposed as read-only property `drop_count` (line 199).

`LivePipeline._queue_drop_count` (line 163), incremented at line 343 on `queue.Full` in `_enqueue_setpoint()`. Exposed via `health().queue_drop_count`.

Two separate drop counters for two separate queues.

### 4e. Behaviour when log write is slow

T-LOG writes inside `open(path, 'w')` with `fh.write(json.dumps(d) + '\n')` at line 221. If disk I/O is slow, the T-LOG thread blocks but the calling threads do not — they continue calling `log()` which puts to the queue. Items accumulate in the bounded queue (maxsize=10,000 default). If the queue fills, `_drop_count` increments silently. **The calling threads (T-HB, T-SP, T-MON) are never blocked by slow disk I/O.**

---

## Question Set 5 — Timestamp monotonicity

**Files:** `integration/drivers/vio_driver.py`, `integration/bridge/mavlink_bridge.py`

### 5a. IFM-01 guard location

**`integration/drivers/vio_driver.py`**, class `_MonotonicityGuard`, method `check(t: float)` at **line 76**.

```python
def check(self, t: float) -> bool:
    if t <= self._last_t:
        event_id = f"IFM01-{self._violation_count:04d}"
        self._violations.append({...})
        self._violation_count += 1
        return False
    self._last_t = t
    return True
```

Used by `OfflineVIODriver.read()` at line 217: `valid = self._guard.check(t)`.

### 5b. Behaviour on non-monotonic timestamp

Two actions:
1. **Records** violation internally: `event_id`, `t_prev`, `t_current`, `delta` stored in `_violations` list (lines 82–88).
2. **Rejects** by returning `False` → `VIOReading.valid = False`. Caller must check `valid` before passing to ESKF. No external log call is made by the guard itself.

The ESKF is NOT called when `valid=False`. The `_MonotonicityGuard` is silent to external systems — violations are inspectable via `monotonicity_guard.violations` property only.

### 5c. Existing tests for IFM-01 under simulated timing

**Partial — yes, but not under real timing injection.**

`integration/tests/test_prehil_vio_driver.py` contains:
- `test_G_VIO_12_ifm01_rejects_non_monotonic` (line 144): forces violation via `d._guard._last_t = 999.0` — direct state mutation, not real timing.
- `test_G_VIO_13_ifm01_records_event_fields` (line 151): verifies violation record fields.
- `test_G_VIO_14_ifm01_violation_count` (line 165): verifies violation_count increments.
- `test_G_VIO_15_valid_frames_continue_after_violation` (line 176): verifies recovery.

**No test simulates IFM-01 under live clock timing** (e.g., time.sleep jitter, rapid re-injection, same-timestamp collision). All existing tests use direct `_guard._last_t` mutation. RC-7 (timestamp monotonicity injection under live timing) is therefore genuinely untested.

### 5d. Current timestamp source for VIO measurements

`OfflineVIODriver`: `_mission_t` initialised to 0.0, incremented by `_dt_s` per `read()` call (lines 221–222). Monotonically increasing by construction under normal use.

`LiveVIODriver`: stub — raises `DriverReadError`. No timestamp source defined.

The bridge (`mavlink_bridge.py`) uses `TimeReference.time_boot_ms()` for MAVLink message timestamps (FM-5), not for VIO measurement timestamps. These are independent clocks.

---

## Question Set 6 — Existing RC test infrastructure

### 6a. Files referencing RC-7, RC-8, RC-11, or the target strings

```
grep results across tests/ and integration/tests/:
```

| File | Match | Context |
|---|---|---|
| `tests/test_bcmp2_at1.py` | `vio_outages` (line 66, 68, 69) | Field name in scenario disturbance schedule dict — not an RC test |
| `integration/tests/test_prehil_vio_driver.py` | `monotonicity_guard` (line 156, 168, 171, 174) | IFM-01 gate tests (G-VIO-12 through G-VIO-15) |

**No files reference RC-7, RC-8, or RC-11 by name.**  
**No files reference `drop_rate`.**  
The string `vio_outage` (single word, no 's') appears in `test_bcmp2_at1.py` only as a dictionary key, not as a test of VIO outage behaviour.

### 6b. Integration test files under integration/tests/

| File | Gates | Subject |
|---|---|---|
| `test_prehil_bridge.py` | 9 (G-TREF-01 to G-TREF-09) | TimeReference |
| `test_prehil_drivers.py` | 10 (G-BASE-01 to G-BASE-10) | SensorDriver ABC conformance |
| `test_prehil_latency.py` | 10 (G-LAT-01 to G-LAT-10) | LatencyMonitor |
| `test_prehil_pipeline.py` | 15 (G-PIPE-01 to G-PIPE-15) | LivePipeline T-NAV |
| `test_prehil_vio_driver.py` | 22 (G-VIO-01 to G-VIO-22) | OfflineVIODriver + IFM-01 |

Total: 66 integration gates.

### 6c. Existing VIO OUTAGE test fixtures

**No reusable test fixture for VIO OUTAGE exists in the formal test suites.**

`inject_outage.py` (repo root) is a demo script that injects VIO outage by patching `_vio_paused = True` into the nav loop via `types.MethodType`. It is not a test fixture — it requires a live MAVLink connection and PX4 simulator. It cannot be run in unit/SIL tests.

`test_bcmp2_at1.py` references `vio_outages` as a disturbance schedule field (BCMP-2 scenario data) but does not test the VIO navigation mode state machine.

---

## Summary of Findings That Require QA Attention

| Finding | Severity | Implication |
|---|---|---|
| `vio_mode.py` emits **zero logs** — no OUTAGE/RESUMPTION emission | HIGH | RC-11 tests must instrument caller, not module. Tests must observe mode via `current_mode` property or summary counters. |
| ESKF has **no np.isfinite assertion** anywhere in the file | HIGH | RC-11b (setpoint continuity) must address NaN propagation path explicitly. |
| T-SP has **no NaN guard** on setpoint values | HIGH | NaN in `state.p` transmits silently to PX4. Sprint D must specify whether guard goes in ESKF, pipeline, or bridge. |
| `LivePipeline._setpoint_queue` and `MAVLinkBridge._setpoint_x_m/_y_m/_z_m` are **not connected** | HIGH | Setpoint continuity during OUTAGE (RC-11b) is not currently testable end-to-end without the missing consumer. Sprint D spec must address wiring or scope to unit level only. |
| **OI-21 appears stale** — `mark_send` IS integrated at bridge lines 358–359 | MEDIUM | OI-21 claim contradicted by code. QA agent should verify and close OI-21. |
| BridgeLogger has **no 200Hz logging path** — RC-8 has no existing infrastructure | MEDIUM | RC-8 formal 60s non-blocking test requires new instrumentation. The queue + T-LOG pattern in BridgeLogger is the right model but is not yet wired to T-NAV. |
| IFM-01 guard tests use **direct state mutation** not live timing | MEDIUM | RC-7 tests must inject non-monotonic timestamps via driver interface or clock manipulation — existing G-VIO-12 to G-VIO-15 tests do not cover this. |
| `OfflineVIODriver._mission_t` is monotonically increasing by construction — **IFM-01 is never triggered in normal operation** | LOW | The guard is tested but structurally inert during normal SIL replay. Only triggered via direct `_guard._last_t` mutation or actual clock anomaly. |
