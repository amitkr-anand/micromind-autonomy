# HANDOFF — S-NEP-04 Closure
**Date:** 22 March 2026
**Status:** S-NEP-04 COMPLETE — all steps closed
**From:** S-NEP-04 execution session (22 March 2026)
**To:** Next programme phase session

---

## Critical Architecture Reminder

Two repositories. Both updated in S-NEP-04.

| Repo | Latest commit | Role |
|---|---|---|
| `micromind-autonomy` | ea1fe3e | Navigation codebase — update_vio() + core/fusion/ added |
| `nep-vio-sandbox` | 154c860 | Evaluation pipeline — fusion runners + logs |

---

## Programme Invariants (Do Not Violate)

1. **Frozen constants are immutable**
   - `_ACC_BIAS_RW`, `_GYRO_BIAS_RW`, `_POS_DRIFT_PSD`, `_GNSS_R_NOMINAL`
   - Any modification requires explicit Technical Director approval

2. **EKF interface stability**
   - `update_vio()` is additive only
   - No changes to `propagate()`, `inject()`, or `update_gnss()` without approval

3. **Reproducibility is mandatory**
   - All evaluations must be runnable via committed scripts
   - All results must map to committed logs

4. **NIS interpretation constraint**
   NIS band [0.5, 8.0] remains valid for its intended estimator regime. PF-03 establishes that the current operating regime (high-rate VIO) does not satisfy the assumptions required for this band. Therefore:
   - NIS must NOT be tuned to satisfy the band
   - NIS must NOT be reinterpreted
   - NIS must be treated as a diagnostic signal only under current architecture
   - Any modification to R, measurement model, or NIS applicability requires explicit Technical Director approval

5. **Baseline anchoring**
   - All future evaluations must reference S-NEP-04 logs (04-B, 04-C, 04-D)
   - Any behavioural change must be compared against these baselines

---

## Repo State at Closure

### micromind-autonomy

```
Latest commits:
  ea1fe3e  S-NEP-04 04-A: add update_vio() to ErrorStateEKF (missed from fe6e359)
  fe6e359  S-NEP-04 04-A: VIO fusion interface — update_vio(), core/fusion/, T-01+T-02 gates (19/19)
  140be0b  docs: Add system baseline report for micromind-node01

Branch: main (clean)
Test suite: 222/222 (S0–S10 regression gates) + 19/19 (S-NEP-04 04-A interface tests)
Total: 241 tests passing
```

**New modules added in S-NEP-04:**
```
core/
  ekf/
    error_state_ekf.py    ← update_vio() method added at line ~208 (additive only)
  fusion/
    __init__.py
    vio_covariance_error.py   ← VIOCovarianceError (IFM-04)
    frame_utils.py            ← R_ENU_TO_NED, rotate_pos_enu_to_ned(), rotate_cov_enu_to_ned(),
                                  extract_vio_position_cov()
    fusion_logger.py          ← FusionLogger (O-01 through O-08)
tests/
  test_s_nep_04a_interface.py  ← T-01 (7 assertions) + T-02 (8 assertions) + smoke (2)
```

**Frozen constants — unchanged:**
- `_ACC_BIAS_RW = 9.81e-7`
- `_GYRO_BIAS_RW = 4.04e-8`
- `_POS_DRIFT_PSD = 1.0`
- `_GNSS_R_NOMINAL`
- `update_gnss()`, `inject()`, `propagate()` — not modified

### nep-vio-sandbox

```
Latest commits:
  154c860  S-NEP-04 04-C: IMU+VIO fusion — ATE 0.0795m PASS, NIS structural limitation identified
  c875356  S-NEP-04 04-D: rate-response validation — stride 6/12, NIS monotonic confirmed, EC-01 PASS
  b7adf11  S-NEP-04 04-B: offline replay validation — MH_01 x3, ATE 0.0865m, EC-01 PASS, zero IFM

Branch: main (clean)
Test suite: 443/443
```

**New fusion artifacts:**
```
fusion/
  run_04b_offline.py       ← offline replay (Stage-2 .npy arrays → ESKF)
  run_04c_imu_vio.py       ← IMU+VIO fusion (200Hz IMU + 124Hz VIO → ESKF)
  run_04d_downsampled.py   ← stride-based VIO downsampling (rate-response test)
  logs/
    mh01_run1.json          ← 04-B run 1
    mh01_run2.json          ← 04-B run 2
    mh01_run3.json          ← 04-B run 3
    mh01_run1_04c.json      ← 04-C run 1
    mh01_run2_04c.json      ← 04-C run 2
    mh01_run3_04c.json      ← 04-C run 3
    mh01_s6_run1_04d.json   ← 04-D stride-6 run 1
    mh01_s6_run2_04d.json   ← 04-D stride-6 run 2
    mh01_s6_run3_04d.json   ← 04-D stride-6 run 3
    mh01_s12_run1_04d.json  ← 04-D stride-12 run 1
    mh01_s12_run2_04d.json  ← 04-D stride-12 run 2
    mh01_s12_run3_04d.json  ← 04-D stride-12 run 3
```

---

## S-NEP-04 Results Summary

### 04-A — Interface Implementation (micromind-autonomy)

| Deliverable | Status |
|---|---|
| update_vio() in ErrorStateEKF | ✅ Committed ea1fe3e |
| core/fusion/ package | ✅ Committed fe6e359 |
| T-01 Frame Sanity (7 assertions) | ✅ PASS |
| T-02 Covariance Extraction (8 assertions) | ✅ PASS |
| Full regression suite (222 gates) | ✅ Unchanged |

### 04-B — Offline Replay Validation

| Metric | Value | Verdict |
|---|---|---|
| ATE RMSE (MH_01 ×3) | 0.0865 m | ✅ EC-01 PASS (limit 0.174m) |
| std(ATE) | 0.000 m | ✅ EC-06 PASS |
| IFM events | 0 | ✅ EC-05 PASS |
| NIS | Not evaluable | No propagation — structural artifact |

### 04-C — IMU+VIO Fusion (124Hz VIO)

| Metric | Value | Verdict |
|---|---|---|
| ATE RMSE steady (MH_01 ×3) | 0.0795 m | ✅ EC-01 PASS |
| IFM events | 0 | ✅ EC-04 PASS |
| NaN / divergence | None | ✅ EC-05 PASS |
| std(ATE) | 0.000 m | ✅ EC-06 PASS |
| NIS mean steady | 0.000353 | NOT APPLICABLE — measurement model limitation |

### 04-D — Rate-Response Validation

| Config | VIO rate | NIS mean | NIS vs 04-C | ATE steady | EC-01 |
|---|---|---|---|---|---|
| Stride 6 | ~21 Hz | 0.016983 | 48× higher | 0.0805 m | ✅ PASS |
| Stride 12 | ~10 Hz | 0.033515 | 95× higher | 0.0856 m | ✅ PASS |

**Primary finding:** NIS increases monotonically with stride — correct estimator behaviour confirmed.

### Closure Statement

S-NEP-04 establishes a verified fusion baseline.

The estimator has been validated across:
- measurement-dominant regime (high-rate VIO)
- prediction-influenced regime (downsampled VIO)

All core behaviours (propagation, update, covariance evolution, stability, determinism)
have been confirmed through controlled experiments.

No open defects remain.

---

## Key Programme Finding — PF-03

**NIS measurement model limitation** (new finding, S-NEP-04):

The ESKF NIS evaluation band [0.5, 8.0] assumes innovations reflect accumulated state uncertainty between corrections. At 124Hz VIO, innovations reflect only inter-frame relative motion (~2mm), which is ~40× smaller than the absolute measurement noise (σ ≈ 87mm). NIS is structurally suppressed in this operating regime.

NIS responds correctly to rate changes (monotonic increase confirmed in 04-D):
- 124Hz → NIS 0.000353
- 21Hz  → NIS 0.017 (48× increase)
- 10Hz  → NIS 0.034 (95× increase, ~2× per stride doubling)

**PF-03 is a measurement model limitation, not an estimator defect.**

No corrective action is to be taken within the current architecture.

Any change to R definition, measurement model, or NIS interpretation must be treated as a future architectural decision, not a sprint-level fix.

---

## Environment Notes for Next Session

**micromind-node01 specifics:**
- Python: `python3` → Python 3.12.3 (system)
- conda not installed — use `python3` directly
- Docker container: `openvins_humble` — available but not needed for offline evaluation
- Repo paths: `~/micromind/repos/micromind-autonomy` and `~/micromind/repos/nep-vio-sandbox`
- All fusion runners execute from `~/micromind/repos/micromind-autonomy` as working directory

**IMU data path:**
```
~/micromind/repos/nep-vio-sandbox/datasets/data/euroc/machine_hall/MH_01_easy/mav0/imu0/data.csv
```
Format: `timestamp[ns], w_x, w_y, w_z, a_x, a_y, a_z` (gyro first, acc second)

**VIO data:** Stage-2 .npy arrays in `~/micromind/repos/nep-vio-sandbox/benchmark/`

---

## Entry Checklist for Next Session

```bash
cd ~/micromind/repos/micromind-autonomy
git log --oneline main | head -3
# Expect ea1fe3e at top

grep -n "def update_vio" core/ekf/error_state_ekf.py
# Expect line ~208

python3 -m pytest tests/ -q 2>&1 | tail -3
# Expect 241 passed (222 S0-S10 + 19 S-NEP-04-A)

cd ~/micromind/repos/nep-vio-sandbox
git log --oneline main | head -3
# Expect 154c860 at top
```

---

## Next Phase — S-NEP-05 (Mission-Scale Validation)

**Transition principle:** S-NEP-04 validated correctness. S-NEP-05 must validate behaviour under scale. No architectural changes are permitted.

### Objective

Evaluate long-duration estimator behaviour under sustained operation — stability over time, drift accumulation, consistency under continuous execution.

### Scope

Must: extend beyond short EuRoC segments, operate over multi-minute continuous runs, preserve full IMU + VIO pipeline (same as 04-C baseline).

Must NOT: modify Q, R, or bias models; alter `update_vio()` or EKF structure; introduce new sensors or constraints; reinterpret PF-03.

### Required Evaluations

**A. Drift Accumulation** — Does position error grow, stabilise, or oscillate? Error vs time curve, drift rate (m/min), regime classification (bounded / linear / accelerating).

**B. Correction Effectiveness** — Do VIO updates continue to correct accumulated INS drift? Pre/post-update error behaviour, innovation magnitude trend over time.

**C. Covariance Behaviour** — Does P remain meaningful over extended operation? No long-term collapse, no unbounded growth, stable propagate–update pattern throughout.

**D. Stability** — No NaN, no resets, no divergence, no degradation in update rate across full duration.

**E. Consistency** — ≥3 independent runs, identical outputs (std(metrics) ≈ 0).

### Success Criteria

| ID | Condition | Expectation |
|---|---|---|
| EC-01 | Stability | No NaN, reset, or divergence |
| EC-02 | Drift behaviour | Bounded or linear (no acceleration) |
| EC-03 | Correction effectiveness | VIO updates remain effective throughout |
| EC-04 | Covariance | Stable grow/contract pattern over full duration |
| EC-05 | Repeatability | Deterministic across runs |
| EC-06 | ATE | Remains within S-NEP-04 envelope |

### Metric Emphasis

| Metric | Role |
|---|---|
| Error vs time | **Primary** |
| Drift rate | **Primary** |
| ATE | Secondary (end-point only) |
| Innovation trend | Diagnostic |
| Covariance trace | Structural validation |
| NIS | Diagnostic only (PF-03 applies) |

### Logging Requirements

Minimum signals per step: timestamp, position error vs GT, innovation magnitude, trace(P), step type. Logs must allow reconstruction of error vs time, P vs time, and innovation vs time.

### Failure Handling

Flag explicitly → do not reinterpret metrics → do not modify parameters → diagnose using logs → escalate before proceeding.

### Deliverable

Clear classification of drift behaviour, evidence of correction effectiveness (or limitation), statement on estimator suitability for mission-scale operation.

**Status:**
```
S-NEP-05: TO BE DEFINED — EXECUTION PLAN REQUIRED BEFORE IMPLEMENTATION
```
