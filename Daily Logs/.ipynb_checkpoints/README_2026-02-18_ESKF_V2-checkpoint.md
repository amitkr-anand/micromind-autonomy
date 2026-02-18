# MicroMind Autonomy — Daily Log
## ESKF V2 Baseline (LOCKED)
**Date:** 18 February 2026  
**Stage:** Foundational Autonomy — ESKF Bug Fix + GNSS Update  
**Status:** ✅ WORKING & LOCKED

---

## 1. What Was Fixed (V1 → V2)

### Bug 1 — IMU simulation: missing gravity modelling
**Root cause:** V1 computed `acc_meas = bias + noise` — no gravity.  
A real IMU measures **specific force** in body frame: `f = R^T(a_true - g)`.  
For a stationary vehicle, `a_true = 0`, so the IMU reads `-R^T * g` — gravity felt as upward force.  
V1 was modelling an IMU in free-fall. Result: 15,862 km drift in 30 minutes.

**Fix:** `_simulate_imu()` now correctly rotates world gravity into body frame and computes the specific force the accelerometer actually measures before adding bias and noise.

### Bug 2 — EKF bias estimates never fed back into INS
**Root cause:** V1 computed `b_a_est = ekf.x[9:12]` but then discarded it. INS ran with `state.ba = zeros` throughout — true bias was never corrected.

**Fix:** `ekf.inject(state)` is called after each GNSS update. Corrections to `p, v, q, ba, bg` are written back into the nominal INS state, closing the feedback loop.

---

## 2. New Capability: GNSS Update with BIM Trust Score

`ErrorStateEKF.update_gnss(state, gnss_pos, trust_score)`

| trust_score | BIM State | Behaviour |
|---|---|---|
| 1.0 | Green | Full GNSS weight (R = R_nominal) |
| 0.4 | Amber | Reduced weight (R = R_nominal / 0.4) |
| < 0.1 | Red | Update skipped — pure inertial |

This is the integration hook for the BIM module (Sprint S2).

---

## 3. Verified Results

| Scenario | Duration | Final Drift |
|---|---|---|
| GNSS aided (trust=1.0) | 5 min | 3.0 m ✅ |
| GNSS denied (trust=0.0) | 1 min | 67.6 m ✅ (bias-induced) |
| Amber state (trust=0.4) | 5 min | 4.2 m ✅ |

---

## 4. Repository State

```
core/ekf/error_state_ekf.py   ← V2: full F matrix, update_gnss, inject
core/ins/mechanisation.py     ← unchanged (correct)
core/ins/state.py             ← unchanged (correct)
core/math/quaternion.py       ← unchanged (correct)
core/constants.py             ← unchanged (correct)
sim/eskf_simulation.py        ← V2: correct IMU sim, GNSS update, 5 returns
```

---

## 5. What Comes Next

**Sprint S2 — BIM Module** (`core/bim/bim.py`)
- Inputs: GNSS raw (PDOP, SNR, C/N0), multi-constellation delta, Doppler deviation, EW events
- Output: `trust_score` (0.0–1.0) → feeds directly into `ekf.update_gnss()`
- G/A/R state with 3-sample hysteresis
- Target: GNSS spoof injection via injector script → BIM flips Red within 250 ms

**The ESKF ↔ BIM interface is already live.** BIM just needs to produce a `trust_score`.

---

## 6. Architecture Note

The error-state EKF is **not** a MicroMind IP boundary.  
It is the estimation substrate — equivalent to PX4's EKF2 in the control stack.

**MicroMind IP lives above it:**
- BIM: what degrades the trust score and when
- State machine: what mode transitions trust_score triggers  
- EW Engine: what feeds the EW impact component of BIM
- DMRL: terminal guidance decision logic

The ESKF is the foundation. BIM is the first brick of the IP layer.
