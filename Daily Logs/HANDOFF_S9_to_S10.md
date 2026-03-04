# MicroMind — Sprint Handoff: S9 → S10
**Generated:** 4 March 2026
**Outgoing sprint:** S9 — TRN+ESKF Architectural Correction / NAV-01 Closure
**Incoming sprint:** S10 — TBD (scope pending TASL meeting outcome)
**Author:** Amit (amitkr-anand)
**Commit:** 7fba53c
**Tag:** s9-nav01-pass

---

## 1. Outgoing Sprint: What Was Completed

Sprint S9 was a single-objective corrective sprint: close the NAV-01 requirement
(INS drift < 100m per 5km segment over 150km+ corridors) using the STIM300 IMU.
The sprint diagnosed and resolved fundamental architectural errors in the TRN+ESKF
integration that had been present but hidden since S3.

**NAV-01 result at sprint close:**

| Corridor | max_5km_drift | final_drift | NAV-01 |
|---|---|---|---|
| 20 km | ~4 m | ~4 m | ✅ PASS |
| 50 km | 9.7 m | 3.7 m | ✅ PASS |
| 150 km | 14.5 m | 4.2 m | ✅ PASS |

Limit: < 100 m per 5 km segment.

### Changes delivered

| File | Change | Gate |
|---|---|---|
| `core/ins/trn_stub.py` | S9-1/S9-2: Removed internal Kalman filter from `TRNStub.update()`. Signature changed to scalar inputs. Returns raw NCC offset only. | ✅ |
| `core/ekf/error_state_ekf.py` | S9-4: Corrected Q-matrix bias RW values to STIM300 TS1524 rev.31. Added position process noise term `_POS_DRIFT_PSD=1.0 m/√s`. | ✅ |
| `sim/als250_nav_sim.py` | S9-3: Correct propagation order (ESKF propagate → INS propagate → TRN update → ESKF update → inject). Added `corridor_km` parameter. NAV-01 drift metric changed to horizontal 2D norm. | ✅ |
| `scenarios/bcmp1/bcmp1_runner.py` | S9-0: Fixed S5 dispatcher routing (regression fix — 111/111 restored). | ✅ |

### Regression state at sprint close

```
python run_s5_tests.py              → 111/111  PASS ✅
python tests/test_s6_zpi_cems.py   → 36/36    PASS ✅
python run_s8_tests.py             → 68/68    PASS ✅
Total: 215/215
```

---

## 2. Root Causes Diagnosed and Fixed in S9

This section is retained permanently — these were non-trivial architectural errors
that cost significant time to surface.

### RC-1: TRN internal Kalman shadowing the ESKF (S9-1/S9-2)
`TRNStub.update()` contained its own 2×2 position Kalman filter that directly mutated
`ins.north_m` / `ins.east_m` and `ins.P`. This created a duplicate estimation loop
running in parallel with the 15-state ESKF, with inconsistent noise parameters and no
covariance coupling. The two filters were fighting each other. Fix: removed the internal
Kalman entirely. `update()` now returns a raw `TRNCorrection` record; the caller
(simulation loop) passes it to the ESKF.

### RC-2: ESKF Q-matrix 61,000× too large for gyro bias (S9-4)
`_GYRO_BIAS_RW` was set to `1e-5 rad/s/√s`. The correct STIM300 value (derived from
TS1524 rev.31, 600s Allan variance) is `4.04e-8 rad/s/√s` — 247× smaller. The inflated
Q caused the filter to expect rapid bias variation, making it overweight new measurements
and preventing stable convergence.

### RC-3: Position block of Q was zero — Kalman gain → 0 (S9-4 addendum)
After correcting bias RW values, `Q[0:3, 0:3]` remained zero. With `P_pos ≈ 1e-6 m²`
and `R_north = 25 m²`, the Kalman gain `K ≈ 4e-8` — effectively zero. TRN fired 33
corrections over 50km but applied nothing. Fix: added `_POS_DRIFT_PSD = 1.0 m/√s`
as position process noise, giving `P_pos ≈ 27 m²` after 1500m, matching R and
producing `K ≈ 0.5` (healthy).

### RC-4: Propagation order violated (S9-3)
The simulation loop called `ins_propagate()` before `eskf.propagate()`, meaning the ESKF
received a stale state for covariance propagation. Fixed to: ESKF propagate (with
bias-corrected accel) → INS propagate → TRN → ESKF update/inject.

### RC-5: NAV-01 drift metric included altitude (discovery in S9)
`drift_per_seg` was computed as a 3D Euclidean norm including altitude. The ALS-250
trajectory has a sinusoidal altitude profile and the barometric channel is not corrected
by TRN (horizontal-only system). Altitude drifts at ~3.4 m/km, giving ~102m of
apparent 3D drift at 30km even when horizontal position is accurate to <5m. Fix:
changed drift metric to `np.linalg.norm(state.p[:2] - true_pos[k, :2])` (horizontal
2D norm). This is physically correct — TRN is a horizontal navigation aid.

---

## 3. Live Interfaces Changed in S9

### TRNStub.update() — NEW SIGNATURE (breaking change from S3)

**Old (S3–S8):**
```python
def update(self, ins: INSState, true_north_m, true_east_m,
           dt, ground_track_m, timestamp_s) -> TRNCorrection | None
# Side effects: mutated ins.north_m, ins.east_m, ins.P
```

**New (S9+):**
```python
def update(self,
           ins_north_m:    float,
           ins_east_m:     float,
           true_north_m:   float,
           true_east_m:    float,
           ground_track_m: float,
           timestamp_s:    float = 0.0) -> TRNCorrection | None
# No side effects. Returns TRNCorrection with .accepted, .delta_north_m, .delta_east_m
# Caller must check corr.accepted before passing to ESKF.
```

### ErrorStateEKF — New Q constants

```python
_ACC_BIAS_RW   = 9.81e-7   # m/s²/√s  — STIM300 TS1524 rev.31
_GYRO_BIAS_RW  = 4.04e-8   # rad/s/√s — STIM300 TS1524 rev.31
_POS_DRIFT_PSD = 1.0       # m/√s     — position process noise (S9 addendum)
```

### run_als250_sim() — New parameter

```python
result = run_als250_sim(
    imu_name="STIM300",
    corridor_km=150,      # NEW — overrides duration_s if provided
    seed=42,
    verbose=False,
)
# result['kpi'] keys include: trn_corrections, max_5km_drift_m (horizontal), NAV01_pass
```

### Correct propagation loop pattern (S9-3 canonical)

```python
# Inside the simulation loop — this order is mandatory
eskf.propagate(state, accel_b - state.ba, DT)          # 1. covariance propagation
state = ins_propagate(state, accel_b, gyro_b, DT, ...) # 2. nominal state advance

ground_track_m += CRUISE_SPEED_MS * DT
corr = trn.update(
    ins_north_m    = float(state.p[1]),
    ins_east_m     = float(state.p[0]),
    true_north_m   = float(true_pos[k, 1]),
    true_east_m    = float(true_pos[k, 0]),
    ground_track_m = ground_track_m,
    timestamp_s    = k * DT,
)
if corr is not None and corr.accepted:
    delta_n, delta_e = corr.delta_north_m, corr.delta_east_m
    if abs(delta_n) <= MAX_TRN_CORRECTION_M and abs(delta_e) <= MAX_TRN_CORRECTION_M:
        z = np.array([state.p[0] + delta_e, state.p[1] + delta_n, state.p[2]])
        eskf.update_gnss(state, z, trust_score=1.0)
        eskf.inject(state)
```

---

## 4. Known Issues and Deferred Items

| Item | Status | Notes |
|---|---|---|
| `dashboard/als250_drift_chart.py` (S8-D) | 🔲 Still deferred | Overnight `.npy` run was never completed. Needs a full 250km × 3 IMU run. |
| V7 spec update — STIM300 ARW | ⚠ Still pending | Must update spec floor to ≤ 0.2°/√hr before TASL meeting. |
| NCC simulation speed | ⚠ Known | 150km run took 3.5 hours (pure Python NCC). Acceptable for SIL; not for overnight multi-model runs. If full 250km × 3 IMU runs are needed, NCC must be vectorised (scipy.signal.correlate2d) or run in parallel. |
| S9 test suite | 🔲 Not written | S9 changes are validated by 50km/150km sim results and the full 215/215 regression, but there is no dedicated `test_s9_*.py`. If required, S10 should add regression tests for: (a) TRN correction count > 0, (b) NAV-01 pass at 50km, (c) drift metric is 2D horizontal. |
| `_POS_DRIFT_PSD` value | ⚠ Engineering approximation | 1.0 m/√s is a physically motivated value for tactical INS but has not been derived from Allan variance. If a more precise value is required for TASL, derive from STIM300 velocity random walk: ARW = 0.15°/√hr → position diffusion over time. |

---

## 5. Decisions Made in S9 (Carry Forward)

| Decision | Rationale |
|---|---|
| TRNStub no longer owns a Kalman filter | Single source of truth: the 15-state ESKF. TRN is an observation source only. |
| NAV-01 drift metric is 2D horizontal | TRN is a horizontal navigation aid. Altitude is a separate channel (barometric). 3D norm incorrectly penalises a correctly functioning system. |
| `_POS_DRIFT_PSD = 1.0 m/√s` | Gives P_pos ≈ 27 m² over 1500m correction interval, comparable to TRN measurement noise R = 25 m². Produces healthy Kalman gain K ≈ 0.5. |
| Sanity gate MAX_TRN_CORRECTION_M = 300m | Rejects NCC mis-correlations. At 1500m correction intervals with drift rate ~1m/km, maximum plausible correction is ~1.5m. 300m is conservative; could be tightened to 50m once system is proven. |
| `corridor_km` parameter added to `run_als250_sim` | Avoids computing duration_s manually. Overrides `duration_s` when provided. |

---

## 6. Incoming Sprint: S10 Scope Options

S9 closes the NAV-01 gap. The programme is now in a strong position for the TASL meeting.
S10 scope depends entirely on TASL outcome and timing.

| Fork | Description | Readiness |
|---|---|---|
| A — S8-D completion | ALS-250 three-curve drift chart (STIM300 vs ADIS vs BASELINE) for TASL presentation. Fast — one session if NCC speed is acceptable. | Ready now |
| B — Cybersecurity hardening | `core/cybersec/` — key loading, signed mission envelope, PQC-ready stack (FR-109–112) | No blockers |
| C — NCC vectorisation | Replace pure-Python NCC loop with `scipy.signal.correlate2d`. Target: 150km run in <10 minutes instead of 3.5 hours. Unblocks overnight multi-model runs. | No blockers |
| D — HIL integration prep | ROS2 node wrappers, PX4 SITL skeleton — blocked on TASL platform decision | Blocked |
| E — S9 test suite | Write dedicated regression tests for TRN+ESKF integration | No blockers — low priority |

**Recommended first action for S10:** Run S8-D (three-curve chart) — it is the most visible
deliverable for the TASL presentation and can be done quickly now that NAV-01 is closed.

---

## 7. Session Start Checklist for S10

```bash
# 1. Sync repo
git checkout main && git pull origin main
git log --oneline main | head -5
# Expected: 7fba53c at top (S9 commit)

# 2. Full regression — must be 215/215
python run_s5_tests.py               # 111/111
python tests/test_s6_zpi_cems.py     # 36/36
python run_s8_tests.py               # 68/68

# 3. Confirm session goal with Amit before writing any code
```

---

## 8. End of Sprint Reminder (for whoever closes S10)

At the end of Sprint S10:
1. Save handoff as `Daily Logs/HANDOFF_S10_to_S11.md`
2. Commit and push to `main`
3. Upload to Claude Project knowledge
4. Update `SPRINT_STATUS.md` and re-upload
5. Update Project Instructions if any new interfaces or file paths were added
